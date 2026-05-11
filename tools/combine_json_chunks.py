import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Combine manifest-backed JSON chunks into one JSON array.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    records = []
    for chunk in manifest['chunks']:
        records.extend(json.loads((root / chunk['path']).read_text()))
    Path(args.output).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
