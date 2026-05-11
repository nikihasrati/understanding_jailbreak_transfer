from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.artifacts import write_json_chunks
from pipeline.gcg_push import config as gcg_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create chunked GCG-push artifacts from raw coefficient-modified GCG runs.')
    parser.add_argument('--config', default=str(gcg_config.DEFAULT_CONFIG))
    parser.add_argument('--intervention', choices=['suffix_push', 'orth_shift'], default=None)
    parser.add_argument('--coeff', default=None)
    parser.add_argument('--raw-results-root', default=None)
    parser.add_argument('--artifact-root', default=None)
    parser.add_argument('--check', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def raw_result_path(config: dict[str, Any], intervention: str, coeff: str, index: int, seed: int, raw_root: Path | None) -> Path:
    if raw_root is None:
        return gcg_config.raw_result_path(config, intervention, coeff, index, seed)
    return raw_root / gcg_config.model_alias(config) / gcg_config.experiment_name(config) / intervention / gcg_config.coefficient_dir(coeff) / f'index-{index:04d}' / f'seed-{seed:04d}' / 'results.json'


def artifact_dir(config: dict[str, Any], intervention: str, coeff: str, split: str, artifact_root: Path | None) -> Path:
    if artifact_root is None:
        return gcg_config.artifact_dir(config, intervention, coeff, split)
    return artifact_root / gcg_config.model_alias(config) / intervention / gcg_config.coefficient_dir(coeff) / split


def result_to_record(data: dict[str, Any], suffix_id: int) -> dict[str, Any]:
    best_string = data['best_string']
    strings = data.get('strings') or []
    try:
        num_steps = strings.index(best_string) + 1
    except ValueError:
        num_steps = data.get('num_steps')
    return {
        'prompt_id': int(data['index']),
        'suffix_id': suffix_id,
        'seed': int(data['seed']),
        'prompt': data['goal'],
        'suffix': best_string,
        'jailbreak': data['goal'] + best_string,
        'loss': data['best_loss'],
        'num_steps': num_steps,
    }


def load_no_transfer_records(config: dict[str, Any], intervention: str, coeff: str, raw_root: Path | None, strict: bool) -> list[dict[str, Any]]:
    records = []
    missing = []
    suffix_id = 0
    start_seed, end_seed = gcg_config.seed_range(config, intervention)
    for index in gcg_config.indices(config):
        for seed in range(start_seed, end_seed + 1):
            path = raw_result_path(config, intervention, coeff, index, seed, raw_root)
            if not path.exists():
                missing.append(path)
                continue
            data = json.loads(path.read_text())
            records.append(result_to_record(data, suffix_id))
            suffix_id += 1
    if missing:
        print(f'Missing {len(missing)} raw result files for {intervention} coeff={coeff}')
        for path in missing[:20]:
            print(f'  missing: {path}')
        if len(missing) > 20:
            print(f'  ... {len(missing) - 20} more')
        if strict:
            raise SystemExit('Missing raw GCG-push result files.')
    return records


def create_transfer_dataset(no_transfer_df: pd.DataFrame) -> pd.DataFrame:
    prompt_meta = no_transfer_df[['prompt_id', 'prompt']].drop_duplicates().reset_index(drop=True)
    suffix_meta = no_transfer_df.drop(columns=['prompt_id', 'prompt', 'jailbreak']).drop_duplicates().reset_index(drop=True)
    combos = prompt_meta.merge(suffix_meta, how='cross')
    combos['jailbreak'] = combos['prompt'] + combos['suffix']
    original_pairs = no_transfer_df[['prompt_id', 'suffix_id']].drop_duplicates().assign(transfer=False)
    combos = combos.merge(original_pairs, on=['prompt_id', 'suffix_id'], how='left')
    combos['transfer'] = combos['transfer'].isna()
    return combos[['prompt_id', 'suffix_id', 'seed', 'prompt', 'suffix', 'jailbreak', 'loss', 'num_steps', 'transfer']]


def write_chunked_json(records: list[dict[str, Any]], artifact_path: Path, num_chunks: int, source: str, metadata: dict[str, Any]) -> None:
    if num_chunks < 1:
        raise ValueError('num_chunks must be >= 1')
    chunks = write_json_chunks(records, artifact_path / 'chunks', num_chunks, artifact_dir=artifact_path)
    manifest = {
        'artifact_format': 'chunked_json',
        'source': source,
        'top_level': 'array',
        'total_records': len(records),
        'total_chunks': len(chunks),
        'metadata': metadata,
        'chunks': chunks,
    }
    (artifact_path / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(records)} records to {artifact_path}')


def check_records(df: pd.DataFrame, config: dict[str, Any], intervention: str, split: str) -> None:
    expected_indices = set(gcg_config.indices(config))
    actual_indices = set(int(x) for x in df['prompt_id'].unique()) if not df.empty else set()
    print(f'{split}: prompts={len(actual_indices)}, records={len(df)}')
    if expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        print(f'{split}: missing configured prompt IDs: {missing}')
        print(f'{split}: extra prompt IDs: {extra}')
    if 'seed' in df.columns and not df.empty:
        seed_counts = df.groupby('prompt_id')['seed'].nunique()
        print(f'{split}: seed-count range per prompt: {seed_counts.min()}..{seed_counts.max()}')


def process_one(config: dict[str, Any], intervention: str, coeff: str, args: argparse.Namespace) -> None:
    raw_root = gcg_config.repo_path(args.raw_results_root) if args.raw_results_root else None
    artifact_root = gcg_config.repo_path(args.artifact_root) if args.artifact_root else None
    no_transfer_records = load_no_transfer_records(config, intervention, coeff, raw_root, args.strict)
    no_transfer_df = pd.DataFrame(no_transfer_records)
    transfer_df = create_transfer_dataset(no_transfer_df) if not no_transfer_df.empty else pd.DataFrame()

    if args.check:
        check_records(no_transfer_df, config, intervention, 'no_transfer')
        check_records(transfer_df, config, intervention, 'transfer')

    metadata = {
        'model_alias': gcg_config.model_alias(config),
        'model_id': gcg_config.model_id(config),
        'experiment': gcg_config.experiment_name(config),
        'intervention': intervention,
        'coeff': coeff,
        'indices': gcg_config.indices(config),
        'seed_start': gcg_config.seed_range(config, intervention)[0],
        'seed_end': gcg_config.seed_range(config, intervention)[1],
        'num_steps': gcg_config.num_steps(config),
    }
    source = f'raw GCG-push results generated by pipeline.gcg_push.run_gcg ({intervention}, coeff={coeff})'
    write_chunked_json(
        no_transfer_records,
        artifact_dir(config, intervention, coeff, 'no_transfer', artifact_root),
        gcg_config.artifact_chunks(config, intervention, 'no_transfer'),
        source,
        {**metadata, 'split': 'no_transfer'},
    )
    write_chunked_json(
        transfer_df.to_dict(orient='records'),
        artifact_dir(config, intervention, coeff, 'transfer', artifact_root),
        gcg_config.artifact_chunks(config, intervention, 'transfer'),
        source,
        {**metadata, 'split': 'transfer'},
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    config = gcg_config.load_config(args.config)
    for intervention in gcg_config.selected_interventions(config, args.intervention):
        for coeff in gcg_config.selected_coefficients(config, intervention, args.coeff):
            process_one(config, intervention, coeff, args)


if __name__ == '__main__':
    main()
