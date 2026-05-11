import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description='Create a simple manifest for files under a directory.')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact-format', default='file_manifest')
    return parser.parse_args()


def main():
    args = parse_args()
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
