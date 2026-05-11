import argparse
import json
import os

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Combine canonical_tensor_chunks into one tensor file.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(os.path.join(args.input_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    tensors = [torch.load(os.path.join(args.input_dir, c['path']), map_location='cpu') for c in manifest['chunks']]
    torch.save(torch.cat(tensors, dim=0), args.output)


if __name__ == '__main__':
    main()
