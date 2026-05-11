import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Create temporary generation_chunks from canonical manifest chunks.')
    parser.add_argument('--artifact-dir', required=True, help='Artifact directory containing manifest.json and canonical chunks/.')
    parser.add_argument('--manifest', default=None, help='Manifest path. Defaults to <artifact-dir>/manifest.json.')
    parser.add_argument('--output-subdir', default='generation_chunks')
    parser.add_argument('--num-output-chunks', type=int, default=None, help='Defaults to the manifest chunk count.')
    parser.add_argument('--input-column', default='jailbreak', help='Column expected by generation.')
    parser.add_argument('--allow-missing-input-column', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_manifest_records(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    records = []
    for chunk in manifest['chunks']:
        data = json.loads((root / chunk['path']).read_text())
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return manifest, records


def write_chunks(records, output_dir: Path, num_chunks: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('chunk_*.json'):
        stale.unlink()
    chunk_size = len(records) // num_chunks
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = (chunk_id + 1) * chunk_size if chunk_id != num_chunks - 1 else len(records)
        (output_dir / f'chunk_{chunk_id:05d}.json').write_text(json.dumps(records[start:end], indent=2) + '\n')


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    manifest_path = Path(args.manifest) if args.manifest else artifact_dir / 'manifest.json'
    manifest, records = load_manifest_records(manifest_path)
    if records and not args.allow_missing_input_column and args.input_column not in records[0]:
        raise SystemExit(f'Missing expected input column {args.input_column!r}. Use --input-column or --allow-missing-input-column.')
    num_chunks = args.num_output_chunks or int(manifest.get('total_chunks') or len(manifest['chunks']))
    write_chunks(records, artifact_dir / args.output_subdir, num_chunks)
    print(f'Wrote {len(records)} records to {artifact_dir / args.output_subdir} as {num_chunks} generation chunks')


if __name__ == '__main__':
    main()
