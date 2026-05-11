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
    parser = argparse.ArgumentParser(description='Verify chunk checksums in a manifest.')
    parser.add_argument('--manifest', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for chunk in manifest['chunks']:
        path = root / chunk['path']
        actual = sha256_file(path)
        if actual != chunk['sha256']:
            raise SystemExit(f'Checksum mismatch: {path}')
    print(f'OK: {manifest_path}')


if __name__ == '__main__':
    main()
