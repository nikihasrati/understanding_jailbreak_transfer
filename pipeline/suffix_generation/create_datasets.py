import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Create and check multi-seed datasets from raw GCG results.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--results-dir', default='data/gcg_results/raw')
    parser.add_argument('--output-dir', default='outputs/multiple_seed_results')
    parser.add_argument('--expected-prompts', type=int, default=100)
    parser.add_argument('--expected-seeds', type=int, default=100)
    parser.add_argument('--check', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def model_alias(model_path: str) -> str:
    return os.path.basename(model_path).lower()


def create_no_transfer_dataset(raw_model_dir: Path) -> list[dict]:
    results = []
    suffix_index = 0
    for path in sorted(raw_model_dir.glob('index-*/seed-*/results.json')):
        with path.open('r') as f:
            data = json.load(f)
        record = {
            'prompt_id': data['index'],
            'suffix_id': suffix_index,
            'seed': data['seed'],
            'prompt': data['goal'],
            'suffix': data['best_string'],
            'jailbreak': data['goal'] + data['best_string'],
            'loss': data['best_loss'],
            'num_steps': data['strings'].index(data['best_string']) + 1,
        }
        suffix_index += 1
        results.append(record)
    return results


def create_transfer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prompt_meta = df[['prompt_id', 'prompt']].drop_duplicates().reset_index(drop=True)
    suffix_meta = df.drop(columns=['prompt_id', 'prompt', 'jailbreak']).drop_duplicates().reset_index(drop=True)
    combos = prompt_meta.merge(suffix_meta, how='cross')
    combos['jailbreak'] = combos['prompt'] + combos['suffix']
    original_pairs = df[['prompt_id', 'suffix_id']].drop_duplicates().assign(transfer=False)
    combos = combos.merge(original_pairs, on=['prompt_id', 'suffix_id'], how='left')
    combos['transfer'] = combos['transfer'].isna()
    return combos[['prompt_id', 'suffix_id', 'seed', 'prompt', 'suffix', 'jailbreak', 'loss', 'num_steps', 'transfer']]


def check_dataset(df: pd.DataFrame, expected_prompts: int, expected_seeds: int) -> None:
    expected = set(range(expected_prompts))
    actual = set(df['prompt_id'].unique())
    print(f'Missing prompt IDs: {sorted(expected - actual)}')
    seed_counts = df.groupby('prompt_id')['seed'].nunique()
    missing = seed_counts[seed_counts != expected_seeds].index.tolist()
    print(f'Prompt IDs without {expected_seeds} distinct seeds: {missing}')


def main():
    args = parse_args()
    alias = model_alias(args.model_path)
    raw_model_dir = Path(args.results_dir) / os.path.basename(args.model_path)
    if not raw_model_dir.exists():
        raw_model_dir = Path(args.results_dir) / alias
    if not raw_model_dir.exists():
        raise FileNotFoundError(f'Could not find raw GCG results for {args.model_path} under {args.results_dir}')

    out_root = Path(args.output_dir) / alias
    no_transfer_dir = out_root / 'no_transfer'
    transfer_dir = out_root / 'transfer'
    no_transfer_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir.mkdir(parents=True, exist_ok=True)

    records = create_no_transfer_dataset(raw_model_dir)
    no_transfer_path = no_transfer_dir / f'{alias}_multiple_seed_results_no_transfer.json'
    transfer_path = transfer_dir / f'{alias}_multiple_seed_results_transfer.json'

    with no_transfer_path.open('w') as f:
        json.dump(records, f, indent=2)

    df = pd.DataFrame(records)
    create_transfer_dataset(df).to_json(transfer_path, orient='records', indent=2)

    print(f'Wrote {no_transfer_path}')
    print(f'Wrote {transfer_path}')
    if args.check:
        check_dataset(df, args.expected_prompts, args.expected_seeds)


if __name__ == '__main__':
    main()
