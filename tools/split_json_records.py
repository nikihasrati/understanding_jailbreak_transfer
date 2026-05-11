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
    parser = argparse.ArgumentParser(description='Split a JSON array into manifest-backed chunks.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--records-per-chunk', type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)
    chunks_dir = out / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text())
    if not isinstance(data, list):
        data = [data]
    chunks = []
    for i in range(0, len(data), args.records_per_chunk):
        chunk = data[i:i + args.records_per_chunk]
        path = chunks_dir / f'chunk_{len(chunks):05d}.json'
        path.write_text(json.dumps(chunk, separators=(',', ':')) + '\n')
        chunks.append({'path': str(path.relative_to(out)), 'records': len(chunk), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    manifest = {'artifact_format': 'chunked_json', 'source': str(src), 'total_records': len(data), 'total_chunks': len(chunks), 'chunks': chunks}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
