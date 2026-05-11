import argparse
import pandas as pd
import os

from tqdm import tqdm

from pipeline.config import Config

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--num_gpus', type=int, required=False, help='The number of GPUs available')
    parser.add_argument('--multi_seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--chunk_id', required=False, type=int, default=None, help='Chunk ID to process for sharded artifacts')
    parser.add_argument('--no_suffix_completions', action=argparse.BooleanOptionalAction, help='Whether to evaluate no suffix completions or not')
    parser.add_argument('--rephrasings', action=argparse.BooleanOptionalAction, help='Whether to evaluate rephrasing completions or not')
    parser.add_argument('--gcg_push', action=argparse.BooleanOptionalAction, help='Whether to evaluate GCG push completions or not')
    parser.add_argument('--coeff', type=str, default=None, help='Coefficient for GCG push completions')
    parser.add_argument('--suffix_push', action=argparse.BooleanOptionalAction, help='Whether to evaluate suffix push completions or not. Only for --gcg_push flag.')
    parser.add_argument('--orth_shift', action=argparse.BooleanOptionalAction, help='Whether to evaluate orthogonal shift completions or not. Only for --gcg_push flag.')
    return parser.parse_args()


def chunk_path(path: str, chunk_id: int) -> str:
    directory, filename = os.path.split(path)
    evaluation_chunk = os.path.join(directory, 'evaluation_chunks', f'chunk_{chunk_id:05d}.json')
    if os.path.exists(evaluation_chunk):
        return evaluation_chunk
    legacy_chunk = os.path.join(directory, f'{chunk_id}_{filename}')
    if os.path.exists(legacy_chunk):
        return legacy_chunk
    generation_chunk = os.path.join(directory, 'generation_chunks', f'chunk_{chunk_id:05d}.json')
    if os.path.exists(generation_chunk):
        raise FileNotFoundError(
            f'Found generation chunk {generation_chunk}, but evaluation reads temporary evaluation_chunks/. '
            'Run tools/combine_completions.py to re-shard generation chunks for evaluation.'
        )
    canonical_chunk = os.path.join(directory, 'chunks', f'chunk_{chunk_id:05d}.json')
    if os.path.exists(canonical_chunk):
        raise FileNotFoundError(
            f'Found canonical chunk {canonical_chunk}, but evaluation reads temporary evaluation_chunks/. '
            'Create evaluation_chunks/ before running the jailbreak judge.'
        )
    raise FileNotFoundError(f'No evaluation chunk found for chunk_id={chunk_id} under {directory}')


def _populated(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(value, str) and value.strip() == '')


def assert_column_populated(df, column: str, path: str):
    if column not in df.columns:
        raise ValueError(f'{path} is missing required column {column!r}')
    missing = sum(not _populated(value) for value in df[column].tolist())
    print(f'{column}: {len(df) - missing}/{len(df)} populated in {path}')
    if missing:
        raise RuntimeError(f'{missing} records in {path} are missing {column!r}')

def evaluate_generations(jailbreak_judge, data_path, batch_size=100, chunk_id=None):
    if chunk_id is not None:
        print("Chunk ID", chunk_id)
        data_path = chunk_path(data_path, chunk_id)
    
    generations_df = pd.read_json(data_path)
    n = len(generations_df)

    assert_column_populated(generations_df, 'response', data_path)

    if 'jailbroken' not in generations_df.columns:
        generations_df['jailbroken'] = None
        generations_df.to_json(data_path, orient='records', indent=4)

    start = generations_df['jailbroken'].notna().sum()
    print(f"Resuming at index {start}/{n}")

    for batch_start in tqdm(range(start, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        prompts = generations_df['prompt'].iloc[batch_start:batch_end].to_list()
        responses = generations_df['response'].iloc[batch_start:batch_end].to_list()
        classifications = jailbreak_judge.classify_responses(prompts, responses)

        for i, classification in enumerate(classifications):
            generations_df.loc[batch_start + i, 'jailbroken'] = classification

        generations_df.to_json(data_path, orient='records', indent=4)

    assert_column_populated(generations_df, 'jailbroken', data_path)

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path)
    from pipeline.submodules.jailbreak_judge import Llama3JailbreakJudge
    jailbreak_judge = Llama3JailbreakJudge(num_gpus=args.num_gpus)
    
    if args.no_suffix_completions:
        print("Evaluating no suffix completions")
        evaluate_generations(jailbreak_judge, data_path=cfg.no_suffix_generations_path(), chunk_id=None)
    elif args.multi_seed:
        print("Evaluating multi-seed generations")
        # Evaluate multi-seed generations
        # evaluate_generations(jailbreak_judge, data_path=cfg.multi_seed_generations_no_transfer_path(), chunk_id=None)
        evaluate_generations(jailbreak_judge, data_path=cfg.multi_seed_generations_transfer_path(), chunk_id=args.chunk_id)
    elif args.rephrasings:
        print("Evaluating rephrasing completions")
        # Evaluate rephrasing completions
        evaluate_generations(jailbreak_judge, data_path=cfg.prompt_rephrasings_path(), chunk_id=args.chunk_id)
    elif args.gcg_push:
        print("Evaluating GCG push completions transfer")
        if args.coeff is None:
            raise ValueError("Coefficient must be provided for GCG push completions.")
        if not (args.suffix_push or args.orth_shift):
            raise ValueError("Either --suffix_push or --orth_shift must be specified for GCG push completions.")
        
        if args.suffix_push:
            data_path = cfg.gcg_push_suffix_push_transfer_path(args.coeff)
        if args.orth_shift:
            data_path = cfg.gcg_push_orth_shift_transfer_path(args.coeff)

        evaluate_generations(jailbreak_judge, data_path=data_path, chunk_id=args.chunk_id)
    else:
        print("Evaluating cross prompt transfer generations")
        # Evaluate cross prompt transfer generations
        evaluate_generations(jailbreak_judge, data_path=cfg.cross_prompt_transfer_generations_path(), chunk_id=None)
