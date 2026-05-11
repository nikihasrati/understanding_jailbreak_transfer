import argparse
from pathlib import Path

from pipeline.artifacts import load_manifest_records, write_json_chunks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Create temporary generation_chunks from canonical manifest chunks.')
    parser.add_argument('--artifact-dir', required=True, help='Artifact directory containing manifest.json and canonical chunks/.')
    parser.add_argument('--manifest', default=None, help='Manifest path. Defaults to <artifact-dir>/manifest.json.')
    parser.add_argument('--output-subdir', default='generation_chunks')
    parser.add_argument('--num-output-chunks', type=int, default=None, help='Defaults to the manifest chunk count.')
    parser.add_argument('--input-column', default='jailbreak', help='Column expected by generation.')
    parser.add_argument('--allow-missing-input-column', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    manifest_path = Path(args.manifest) if args.manifest else artifact_dir / 'manifest.json'
    manifest, records = load_manifest_records(manifest_path)
    if records and not args.allow_missing_input_column and args.input_column not in records[0]:
        raise SystemExit(f'Missing expected input column {args.input_column!r}. Use --input-column or --allow-missing-input-column.')
    num_chunks = args.num_output_chunks or int(manifest.get('total_chunks') or len(manifest['chunks']))
    write_json_chunks(records, artifact_dir / args.output_subdir, num_chunks, indent=2)
    print(f'Wrote {len(records)} records to {artifact_dir / args.output_subdir} as {num_chunks} generation chunks')


if __name__ == '__main__':
    main()
