import argparse
import json
import os

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Split a tensor into canonical_tensor_chunks.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-chunks', type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tensor = torch.load(args.input, map_location='cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {'artifact_format': 'canonical_tensor_chunks', 'shape': list(tensor.shape), 'chunks': []}
    for i, chunk in enumerate(torch.chunk(tensor, args.num_chunks, dim=0)):
        name = f'tensor_chunk_{i:05d}.pt'
        path = os.path.join(args.output_dir, name)
        torch.save(chunk, path)
        manifest['chunks'].append({'path': name, 'shape': list(chunk.shape)})
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
