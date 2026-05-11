import argparse
import json
import os

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Export activations into alternate layouts.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--format', choices=['nested_by_suffix_layer'], required=True)
    parser.add_argument('--num-prompts', type=int, default=100)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.input_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    tensors = [torch.load(os.path.join(args.input_dir, c['path']), map_location='cpu') for c in manifest['chunks']]
    activations = torch.cat(tensors, dim=0)
    num_examples, num_layers, d_model = activations.shape
    if num_examples % args.num_prompts != 0:
        raise ValueError('num_examples must be divisible by num_prompts')
    num_suffixes = num_examples // args.num_prompts
    activations = activations.view(args.num_prompts, num_suffixes, num_layers, d_model)
    nested = {
        suffix_id: {
            layer_id: [activations[prompt_id, suffix_id, layer_id, :].clone() for prompt_id in range(args.num_prompts)]
            for layer_id in range(num_layers)
        }
        for suffix_id in range(num_suffixes)
    }
    torch.save(nested, os.path.join(args.output_dir, 'nested_by_suffix_layer.pt'))
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
        json.dump({'artifact_format': 'nested_by_suffix_layer', 'num_prompts': args.num_prompts, 'num_suffixes': num_suffixes, 'num_layers': num_layers}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
