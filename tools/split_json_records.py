import argparse
import json
from pathlib import Path

from pipeline.artifacts import sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Split a JSON array into manifest-backed chunks.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--records-per-chunk', type=int, default=5000)
    parser.add_argument('--manifest-source', default=None, help='Source string to record in manifest.json.')
    return parser.parse_args(argv)


def default_manifest_source(src: Path) -> str:
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    try:
        return str(src.resolve().relative_to(data_dir))
    except ValueError:
        return str(src)


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    out = Path(args.output)
    chunks_dir = out / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for stale in chunks_dir.glob('chunk_*.json'):
        stale.unlink()
    data = json.loads(src.read_text())
    if not isinstance(data, list):
        data = [data]
    chunks = []
    for i in range(0, len(data), args.records_per_chunk):
        chunk = data[i:i + args.records_per_chunk]
        path = chunks_dir / f'chunk_{len(chunks):05d}.json'
        path.write_text(json.dumps(chunk, separators=(',', ':')) + '\n')
        chunks.append(
            {
                'path': str(path.relative_to(out)),
                'records': len(chunk),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }
        )
    manifest = {
        'artifact_format': 'chunked_json',
        'source': args.manifest_source or default_manifest_source(src),
        'original_bytes': src.stat().st_size,
        'original_sha256': sha256_file(src),
        'top_level': 'array',
        'total_records': len(data),
        'total_chunks': len(chunks),
        'chunks': chunks,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
