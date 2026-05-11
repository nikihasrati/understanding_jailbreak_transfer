import argparse
import os

import pandas as pd

from pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--generations', action=argparse.BooleanOptionalAction, help='Whether to check for missing generations.')
    parser.add_argument('--evaluations', action=argparse.BooleanOptionalAction, help='Whether to check for missing evaluations.')
    return parser.parse_args()

def check_missing_generations(df):
    print('Number of generations ', len(df['response']))
    print('Number of missing generations ', df['response'].isna().sum())
    null_indices = df[df['response'].isna()].index.tolist()
    print("Indices with null values in 'response':", null_indices)
    return df['response'].isna().any()

def check_missing_evaluations(df):
    print('Number of evaluations ', len(df['jailbroken']))
    print('Number of missing evaluations ', df['jailbroken'].isna().sum())
    null_indices = df[df['jailbroken'].isna()].index.tolist()
    print("Indices with null values in 'jailbroken':", null_indices)
    return df['jailbroken'].isna().any()

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path)
    target_cfg = Config(model_path=args.target_model_path)

    print("Source model path:", args.source_model_path)
    print("Target model path:", args.target_model_path)

    from pipeline.utils import utils
    df_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    df = utils.concat_json_files_in_dir(df_dir)

    if args.generations:
        is_missing_gens = check_missing_generations(df)
        if is_missing_gens:
            print(f"Missing generations found.")
        else:
            print("No missing generations found.")
    
    if args.evaluations:
        is_missing_evals = check_missing_evaluations(df)
        if is_missing_evals:
            print(f"Missing evaluations found.")
        else:
            print("No missing evaluations found.")

if __name__ == "__main__":
    main()