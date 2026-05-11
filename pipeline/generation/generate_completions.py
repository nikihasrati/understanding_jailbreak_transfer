import argparse
import os

import pandas as pd
from tqdm import tqdm

from pipeline.artifacts import assert_column_populated, workflow_chunk_path
from pipeline.config import Config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate completions for transferability datasets.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--multi-seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-id', type=int, default=0)
    parser.add_argument('--no-transfer', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-suffix-completions', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rephrasings', action=argparse.BooleanOptionalAction)
    parser.add_argument('--gcg-push', action=argparse.BooleanOptionalAction)
    parser.add_argument('--coeff', default=None)
    parser.add_argument('--suffix-push', action=argparse.BooleanOptionalAction)
    parser.add_argument('--orth-shift', action=argparse.BooleanOptionalAction)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--batch-size', type=int, default=100)
    return parser.parse_args(argv)


def chunk_path(path: str, chunk_id: int) -> str:
    return workflow_chunk_path(path, chunk_id, 'generation')


def generate_for_path(model, path: str, input_column: str = 'jailbreak', chunk_id: int | None = None, resume: bool = False, batch_size: int = 100, num_chunks: int = 1):
    if chunk_id is not None:
        path = chunk_path(path, chunk_id)
    df = pd.read_json(path)
    if input_column not in df.columns:
        raise ValueError(f'{path} is missing required input column {input_column!r}')
    if not resume:
        inputs = df[input_column].to_list()
        df['response'] = model.generate_completions(inputs, max_new_tokens=200)
        df.to_json(path, orient='records', indent=2)
        assert_column_populated(df, 'response', path)
        return

    if 'response' not in df.columns:
        df['response'] = None
    start = df['response'].notna().sum()
    print(f'Resuming at index {start}/{len(df)}')
    for batch_start in tqdm(range(start, len(df), batch_size)):
        batch_end = min(batch_start + batch_size, len(df))
        texts = df[input_column].iloc[batch_start:batch_end].to_list()
        completions = model.generate_completions(texts, max_new_tokens=200)
        for i, completion in enumerate(completions):
            df.loc[batch_start + i, 'response'] = completion
        df.to_json(path, orient='records', indent=2)
    assert_column_populated(df, 'response', path)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(args.model_path)
    from pipeline.model_utils.model_factory import construct_model_base
    model = construct_model_base(cfg.model_path)

    if args.no_suffix_completions:
        prompts = pd.read_json(cfg.prompts_path())
        prompts['response'] = model.generate_completions(prompts['prompt'].to_list(), max_new_tokens=200)
        os.makedirs(os.path.dirname(cfg.no_suffix_generations_path()), exist_ok=True)
        prompts.to_json(cfg.no_suffix_generations_path(), orient='records', indent=2)
    elif args.rephrasings:
        generate_for_path(model, cfg.prompt_rephrasings_path(), chunk_id=args.chunk_id, resume=args.resume, batch_size=args.batch_size)
    elif args.gcg_push:
        if args.coeff is None:
            raise ValueError('--coeff is required with --gcg-push')
        if args.suffix_push:
            path = cfg.gcg_push_suffix_push_transfer_path(args.coeff)
        elif args.orth_shift:
            path = cfg.gcg_push_orth_shift_transfer_path(args.coeff)
        else:
            raise ValueError('Use --suffix-push or --orth-shift with --gcg-push')
        generate_for_path(model, path, chunk_id=args.chunk_id, resume=args.resume, batch_size=args.batch_size)
    elif args.multi_seed:
        path = cfg.multi_seed_generations_no_transfer_path() if args.no_transfer else cfg.multi_seed_generations_transfer_path()
        generate_for_path(model, path, chunk_id=None if args.no_transfer else args.chunk_id, resume=args.resume, batch_size=args.batch_size)
    else:
        prompts = pd.read_json(cfg.prompts_path()).drop(columns=['category'], errors='ignore')
        suffixes = pd.read_json(cfg.suffixes_path())
        df = prompts.merge(suffixes, how='cross')
        df['jailbreak'] = df['prompt'] + df['suffix']
        df['response'] = model.generate_completions(df['jailbreak'].to_list(), max_new_tokens=200)
        os.makedirs(os.path.dirname(cfg.cross_prompt_transfer_generations_path()), exist_ok=True)
        df.to_json(cfg.cross_prompt_transfer_generations_path(), orient='records', indent=2)


if __name__ == '__main__':
    main()
