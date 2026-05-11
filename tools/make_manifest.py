import argparse
import json
from pathlib import Path

from pipeline.artifacts import sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Create a simple manifest for files under a directory.')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-format', default='file_manifest')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root)
    output = Path(args.output)
    files = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        if path.resolve() == output.resolve():
            continue
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    output.write_text(json.dumps({'artifact_format': args.artifact_format, 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    main()
