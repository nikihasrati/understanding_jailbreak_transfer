import argparse
import pandas as pd
import os

import numpy as np

from pipeline.config import Config

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--coeff', type=str, required=True, help='Coefficient for GCG push completions')
    parser.add_argument('--suffix_push', action=argparse.BooleanOptionalAction, help='Whether to process suffix push results or not.')
    parser.add_argument('--orth_shift', action=argparse.BooleanOptionalAction, help='Whether to process orthogonal shift results or not.')
    return parser.parse_args()

def get_filtered_zero_df(zero_df, coeff_df):
    num_prompts = coeff_df['prompt_id'].nunique()
    num_seeds = coeff_df['seed'].nunique()
    prompts = coeff_df['prompt_id'].unique()

    filtered_df = zero_df[
        zero_df['prompt_id'].isin(prompts) &
        zero_df['seed'].isin(range(num_seeds))
    ]

    # Get the first num_seeds * num_prompts entries for each unique prompt_id
    filtered_df = filtered_df.groupby('prompt_id').apply(lambda x: x.head(num_seeds * num_prompts)).reset_index(drop=True)

    return filtered_df

def get_num_jailbroken(df):
    return df['jailbroken'].sum()

def get_asr(df):
    total_responses = len(df)
    total_jailbroken = get_num_jailbroken(df)
    return total_jailbroken / total_responses if total_responses > 0 else 0

def create_df(num_prompts, num_seeds):
    num_suffixes = num_prompts * num_seeds

    df = pd.DataFrame({
        'prompt_id': np.repeat(np.arange(num_prompts), num_suffixes),
        'suffix_id': np.tile(np.arange(num_suffixes), num_prompts)
    })
    df['seed'] = df['suffix_id'] % num_seeds

    return df[['prompt_id', 'seed', 'suffix_id']]

def test1_get_zero_asr():
    big_df = create_df(num_prompts=3, num_seeds=3)
    small_df = create_df(num_prompts=2, num_seeds=2)

    answer_df = pd.DataFrame({
        'prompt_id': [0, 0, 0, 0, 1, 1, 1, 1],
        'seed': [0, 1, 0, 1, 0, 1, 0, 1],
        'suffix_id': [0, 1, 3, 4, 0, 1, 3, 4]
    })

    filtered_df = get_filtered_zero_df(big_df, small_df)
    assert filtered_df.equals(answer_df), "Filtered DataFrame does not match the expected answer DataFrame"


def test2_get_zero_asr():
    big_df = create_df(num_prompts=4, num_seeds=5)
    small_df = create_df(num_prompts=3, num_seeds=2)

    answer_df = pd.DataFrame({
        'prompt_id': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'seed': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'suffix_id': [0, 1, 5, 6, 10, 11, 0, 1, 5, 6, 10, 11, 0, 1, 5, 6, 10, 11]
    })

    filtered_df = get_filtered_zero_df(big_df, small_df)

    assert filtered_df.equals(answer_df), "Filtered DataFrame does not match the expected answer DataFrame"

if __name__ == "__main__":
    args = parse_arguments()
    coeff = args.coeff
    cfg = Config(model_path=args.model_path)
    from pipeline.utils import utils

    test1_get_zero_asr()
    test2_get_zero_asr()

    if args.suffix_push:
        coeff_dir = cfg.gcg_push_suffix_push_transfer_path(coeff)
    elif args.orth_shift:    
        coeff_dir = cfg.gcg_push_orth_shift_transfer_path(coeff)

    coeff_df = pd.read_json(coeff_dir)

    zero_dir = cfg.multi_seed_generations_transfer_dir()
    zero_df = utils.concat_json_files_in_dir(zero_dir)

    filtered_zero_df = get_filtered_zero_df(zero_df, coeff_df)
    assert len(filtered_zero_df) == len(coeff_df), "Filtered zero DataFrame length does not match coefficient DataFrame length"

    zero_asr = get_asr(filtered_zero_df)
    print(f"Zero coefficient ASR: {zero_asr:.4f}")
    print(f"Number of jailbroken responses for zero coefficient: {get_num_jailbroken(filtered_zero_df)} out of {len(filtered_zero_df)}")

    coeff_asr = get_asr(coeff_df)
    print(f"Coefficient ASR for coefficient {coeff}: {coeff_asr:.4f}")
    print(f"Number of jailbroken responses for coefficient {coeff}: {get_num_jailbroken(coeff_df)} out of {len(coeff_df)}")

    no_suffix_df = pd.read_json(cfg.no_suffix_generations_path())
    filtered_no_suffix_df = no_suffix_df[no_suffix_df['prompt_id'].isin(coeff_df['prompt_id'])]
    no_suffix_asr = get_asr(filtered_no_suffix_df)
    print(f"No suffix ASR: {no_suffix_asr:.4f}")