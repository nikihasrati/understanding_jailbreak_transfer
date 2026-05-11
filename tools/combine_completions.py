import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def populated(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return not (isinstance(value, str) and value.strip() == '')


def parse_args():
    parser = argparse.ArgumentParser(description='Combine generated completion shards and re-split them for evaluation or publication.')
    parser.add_argument('--input-dir', required=True, help='Artifact directory containing generation_chunks/ or evaluation_chunks/.')
    parser.add_argument('--input-subdir', default='generation_chunks')
    parser.add_argument('--output-dir', default=None, help='Artifact directory to write to. Defaults to --input-dir.')
    parser.add_argument('--output-subdir', default='evaluation_chunks')
    parser.add_argument('--num-output-chunks', type=int, required=True)
    parser.add_argument('--check-stage', choices=['generation', 'evaluation', 'none'], default='generation')
    parser.add_argument('--combined-output', default=None, help='Optional path for a full combined JSON array.')
    parser.add_argument('--write-manifest', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--manifest-source', default=None)
    parser.add_argument('--allow-missing', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_records(chunks_dir: Path):
    files = sorted(chunks_dir.glob('chunk_*.json'))
    if not files:
        files = sorted(chunks_dir.glob('*.json'))
    if not files:
        raise SystemExit(f'No JSON chunks found in {chunks_dir}')

    records = []
    for path in files:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


def required_fields(stage: str):
    if stage == 'generation':
        return ['response']
    if stage == 'evaluation':
        return ['response', 'jailbroken']
    return []


def check_records(records, stage: str, allow_missing: bool):
    missing = {field: 0 for field in required_fields(stage)}
    for record in records:
        for field in missing:
            if not populated(record.get(field)):
                missing[field] += 1
    for field, count in missing.items():
        print(f'{field}: {len(records) - count}/{len(records)} populated')
    bad = {field: count for field, count in missing.items() if count}
    if bad and not allow_missing:
        details = ', '.join(f'{field} missing in {count} records' for field, count in bad.items())
        raise SystemExit(f'Completeness check failed: {details}')


def write_chunks(records, output_dir: Path, num_chunks: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('chunk_*.json'):
        stale.unlink()

    chunk_size = len(records) // num_chunks
    chunks = []
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = (chunk_id + 1) * chunk_size if chunk_id != num_chunks - 1 else len(records)
        path = output_dir / f'chunk_{chunk_id:05d}.json'
        path.write_text(json.dumps(records[start:end], separators=(',', ':')) + '\n')
        chunks.append({'path': str(path), 'records': end - start, 'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
    return chunks


def write_manifest(artifact_dir: Path, output_subdir: str, chunks, records, source: str | None):
    previous = {}
    previous_path = artifact_dir / 'manifest.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
    rel_chunks = []
    for chunk in chunks:
        path = Path(chunk['path'])
        item = dict(chunk)
        item['path'] = str(path.relative_to(artifact_dir))
        rel_chunks.append(item)
    manifest = {
        'artifact_format': 'chunked_json',
        'source': source or previous.get('source') or output_subdir,
        'top_level': 'array',
        'total_records': len(records),
        'total_chunks': len(rel_chunks),
        'chunks': rel_chunks,
    }
    if 'metadata' in previous:
        manifest['metadata'] = previous['metadata']
    (artifact_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    input_root = input_dir / args.input_subdir
    output_root = output_dir / args.output_subdir

    records = load_records(input_root)
    check_records(records, args.check_stage, args.allow_missing)
    chunks = write_chunks(records, output_root, args.num_output_chunks)

    if args.combined_output:
        Path(args.combined_output).write_text(json.dumps(records, indent=2) + '\n')
    if args.write_manifest:
        write_manifest(output_dir, args.output_subdir, chunks, records, args.manifest_source)

    print(f'Wrote {len(records)} records to {output_root} as {args.num_output_chunks} chunks')


if __name__ == '__main__':
    main()
