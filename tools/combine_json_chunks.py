import argparse
import json
from pathlib import Path

from pipeline.artifacts import load_manifest_records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Combine manifest-backed JSON chunks into one JSON array.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _, records = load_manifest_records(args.manifest)
    Path(args.output).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
