from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.artifacts import load_json_records, load_manifest_records, populated
from pipeline.gcg_push import config as gcg_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze GCG-push artifacts across interventions and coefficients.')
    parser.add_argument('--config', default=str(gcg_config.DEFAULT_CONFIG))
    parser.add_argument('--model_path', '--model-path', dest='model_path', default=None, help='Optional legacy override for the model path.')
    parser.add_argument('--coeff', default=None, help='Optional single coefficient to analyze.')
    parser.add_argument('--intervention', choices=['suffix_push', 'orth_shift'], default=None, help='Optional single intervention to analyze.')
    parser.add_argument('--suffix-push', '--suffix_push', dest='suffix_push', action=argparse.BooleanOptionalAction, help='Analyze only suffix-push results.')
    parser.add_argument('--orth-shift', '--orth_shift', dest='orth_shift', action=argparse.BooleanOptionalAction, help='Analyze only orthogonal-shift results.')
    parser.add_argument('--output-dir', default='outputs/gcg_push_analysis')
    parser.add_argument('--no-save', action='store_true', help='Print results without writing output files.')
    return parser.parse_args(argv)


parse_arguments = parse_args


def load_artifact_df(artifact_dir: Path) -> pd.DataFrame:
    manifest = artifact_dir / 'manifest.json'
    if not manifest.exists():
        combined = artifact_dir / 'combined.json'
        if combined.exists():
            return pd.DataFrame(load_json_records(combined))
        raise FileNotFoundError(f'No manifest.json or combined.json found in {artifact_dir}')
    _, records = load_manifest_records(manifest)
    return pd.DataFrame(records)


def load_manifest_tree(root: Path) -> pd.DataFrame:
    manifests = sorted(path for path in root.rglob('manifest.json') if 'generation_chunks' not in path.parts and 'evaluation_chunks' not in path.parts)
    if not manifests:
        raise FileNotFoundError(f'No manifest.json files found under {root}')
    frames = [pd.DataFrame(load_manifest_records(path)[1]) for path in manifests]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {'true', '1', 'yes'})


def get_num_jailbroken(df: pd.DataFrame) -> int:
    if df.empty or 'jailbroken' not in df.columns:
        return 0
    return int(bool_series(df['jailbroken']).sum())


def get_asr(df: pd.DataFrame) -> float:
    return get_num_jailbroken(df) / len(df) if len(df) else 0.0


def get_filtered_zero_df(zero_df: pd.DataFrame, coeff_df: pd.DataFrame) -> pd.DataFrame:
    num_prompts = coeff_df['prompt_id'].nunique()
    num_seeds = coeff_df['seed'].nunique()
    prompts = sorted(coeff_df['prompt_id'].unique())
    seeds = sorted(coeff_df['seed'].unique())
    filtered_df = zero_df[zero_df['prompt_id'].isin(prompts) & zero_df['seed'].isin(seeds)].copy()
    rows_per_prompt = num_prompts * num_seeds
    return filtered_df.groupby('prompt_id', group_keys=False).head(rows_per_prompt).reset_index(drop=True)


def no_suffix_path(config: dict[str, Any]) -> Path:
    alias = gcg_config.model_alias(config)
    return gcg_config.REPO_ROOT / 'data' / 'no_suffix_generations' / alias


def baseline_transfer_dir(config: dict[str, Any]) -> Path:
    return gcg_config.REPO_ROOT / 'data' / 'intra_model_transfer' / 'multi_seed' / gcg_config.model_alias(config) / 'transfer'


def analyze_one(config: dict[str, Any], zero_df: pd.DataFrame, no_suffix_df: pd.DataFrame, intervention: str, coeff: str) -> dict[str, Any]:
    artifact = gcg_config.artifact_dir(config, intervention, coeff, 'transfer')
    coeff_df = load_artifact_df(artifact)
    for column in ['response', 'jailbroken']:
        missing = sum(not populated(value) for value in coeff_df.get(column, pd.Series(dtype=object)).tolist())
        if missing:
            raise RuntimeError(f'{artifact} has {missing} records missing {column!r}')

    filtered_zero = get_filtered_zero_df(zero_df, coeff_df)
    if len(filtered_zero) != len(coeff_df):
        raise RuntimeError(
            f'Filtered zero-coefficient baseline has {len(filtered_zero)} records but {artifact} has {len(coeff_df)} records.'
        )
    prompt_ids = set(coeff_df['prompt_id'].unique())
    filtered_no_suffix = no_suffix_df[no_suffix_df['prompt_id'].isin(prompt_ids)] if not no_suffix_df.empty else pd.DataFrame()
    return {
        'model_alias': gcg_config.model_alias(config),
        'intervention': intervention,
        'coeff': coeff,
        'records': len(coeff_df),
        'prompts': coeff_df['prompt_id'].nunique(),
        'suffixes': coeff_df['suffix_id'].nunique(),
        'seeds': coeff_df['seed'].nunique(),
        'zero_jailbroken': get_num_jailbroken(filtered_zero),
        'zero_asr': get_asr(filtered_zero),
        'intervention_jailbroken': get_num_jailbroken(coeff_df),
        'intervention_asr': get_asr(coeff_df),
        'delta_asr': get_asr(coeff_df) - get_asr(filtered_zero),
        'no_suffix_jailbroken': get_num_jailbroken(filtered_no_suffix),
        'no_suffix_asr': get_asr(filtered_no_suffix),
    }


def selected_intervention(args: argparse.Namespace) -> str | None:
    flag_selected = None
    if args.suffix_push and args.orth_shift:
        raise ValueError('Choose at most one of --suffix_push and --orth_shift.')
    if args.suffix_push:
        flag_selected = 'suffix_push'
    if args.orth_shift:
        flag_selected = 'orth_shift'
    if args.intervention and flag_selected and args.intervention != flag_selected:
        raise ValueError('--intervention conflicts with --suffix_push/--orth_shift.')
    return args.intervention or flag_selected


def print_table(df: pd.DataFrame) -> None:
    columns = ['intervention', 'coeff', 'records', 'prompts', 'suffixes', 'seeds', 'zero_asr', 'intervention_asr', 'delta_asr', 'no_suffix_asr']
    table = df[columns].copy()
    for column in ['zero_asr', 'intervention_asr', 'delta_asr', 'no_suffix_asr']:
        table[column] = table[column].map(lambda value: f'{value:.4f}')
    print(table.to_string(index=False))


def main(argv=None) -> None:
    args = parse_args(argv)
    config = gcg_config.load_config(args.config)
    if args.model_path:
        config.setdefault('experiment', {})['model_id'] = args.model_path
        config['experiment']['model_alias'] = Path(args.model_path).name.lower()

    zero_df = load_manifest_tree(baseline_transfer_dir(config))
    no_suffix_df = load_artifact_df(no_suffix_path(config))
    rows = []
    for intervention in gcg_config.selected_interventions(config, selected_intervention(args)):
        for coeff in gcg_config.selected_coefficients(config, intervention, args.coeff):
            rows.append(analyze_one(config, zero_df, no_suffix_df, intervention, coeff))
    results = pd.DataFrame(rows)
    print_table(results)

    if not args.no_save:
        output_dir = gcg_config.repo_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / f'{gcg_config.model_alias(config)}_gcg_push_summary.csv', index=False)
        results.to_json(output_dir / f'{gcg_config.model_alias(config)}_gcg_push_summary.json', orient='records', indent=2)
        print(f'Wrote {output_dir}')


if __name__ == '__main__':
    main()
