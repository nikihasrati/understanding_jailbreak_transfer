import argparse
from pathlib import Path

from pipeline.artifacts import iter_chunk_records, populated


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Check generated responses and jailbreak-judge evaluations in workflow chunks.')
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--subdir', required=True, help='Usually generation_chunks or evaluation_chunks.')
    parser.add_argument('--stage', choices=['generation', 'evaluation'], required=True)
    parser.add_argument('--max-examples', type=int, default=20, help='Maximum missing-record examples to print.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    chunks_dir = Path(args.artifact_dir) / args.subdir
    required = ['response'] if args.stage == 'generation' else ['response', 'jailbroken']
    totals = {field: 0 for field in required}
    missing = {field: [] for field in required}
    total = 0

    try:
        for path, index, row in iter_chunk_records(chunks_dir):
            total += 1
            for field in required:
                if populated(row.get(field)):
                    totals[field] += 1
                elif len(missing[field]) < args.max_examples:
                    label = {k: row.get(k) for k in ('prompt_id', 'suffix_id', 'seed') if k in row}
                    missing[field].append((path.name, index, label))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f'Checked {total} records in {chunks_dir}')
    failed = False
    for field in required:
        count = totals[field]
        print(f'{field}: {count}/{total} populated')
        if count != total:
            failed = True
            for filename, index, label in missing[field]:
                print(f'  missing {field}: {filename} record {index} {label}')
    if failed:
        raise SystemExit('Missing required fields; rerun the previous step before continuing.')


if __name__ == '__main__':
    main()
