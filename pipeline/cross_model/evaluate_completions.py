import argparse
import os 

import pandas as pd

from tqdm import tqdm

from pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID to process')
    parser.add_argument('--num_gpus', type=int, required=False, help='The number of GPUs available')
    return parser.parse_args()

def evaluate_completions(jailbreak_judge, source_cfg, target_cfg, chunk_id, batch_size=100):
    print("Chunk ID", chunk_id)

    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    eval_chunk = os.path.join(save_dir, 'evaluation_chunks', f'chunk_{chunk_id:05d}.json')
    if os.path.exists(eval_chunk):
        save_path = eval_chunk
    else:
        generation_chunk = os.path.join(save_dir, 'generation_chunks', f'chunk_{chunk_id:05d}.json')
        if os.path.exists(generation_chunk):
            raise FileNotFoundError('Found generation_chunks, but cross-model evaluation expects evaluation_chunks. Run tools/combine_completions.py first.')
        file_name = f'{chunk_id}_cross_model_transfer_generations.json'
        save_path = os.path.join(save_dir, file_name)

    chunk_df = pd.read_json(save_path)
    n = len(chunk_df)

    missing_responses = chunk_df['response'].isna().sum() + (chunk_df['response'].astype(str).str.strip() == '').sum()
    print(f"response: {len(chunk_df) - missing_responses}/{len(chunk_df)} populated in {save_path}")
    if missing_responses:
        raise RuntimeError(f"{missing_responses} records in {save_path} are missing response")

    if 'jailbroken' not in chunk_df.columns:
        chunk_df['jailbroken'] = None
        chunk_df.to_json(save_path, orient='records', indent=4)

    # Start from the first index where response is not None
    start = chunk_df['jailbroken'].notna().sum()
    print(f"Resuming at index {start}/{n}")

    for batch_start in tqdm(range(start, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        prompts = chunk_df['prompt'].iloc[batch_start:batch_end].to_list()
        responses = chunk_df['response'].iloc[batch_start:batch_end].to_list()
        classifications = jailbreak_judge.classify_responses(prompts, responses)

        for i, classification in enumerate(classifications):
            chunk_df.loc[batch_start + i, 'jailbroken'] = classification

        chunk_df.to_json(save_path, orient='records', indent=4)

    missing_evals = chunk_df['jailbroken'].isna().sum()
    print(f"jailbroken: {len(chunk_df) - missing_evals}/{len(chunk_df)} populated in {save_path}")
    if missing_evals:
        raise RuntimeError(f"{missing_evals} records in {save_path} are missing jailbroken")

def main():
    args = parse_args()
    source_cfg = Config(model_path=args.source_model_path)
    target_cfg = Config(model_path=args.target_model_path)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    from pipeline.submodules.jailbreak_judge import Llama3JailbreakJudge
    jailbreak_judge = Llama3JailbreakJudge(num_gpus=args.num_gpus)
    evaluate_completions(jailbreak_judge, source_cfg, target_cfg, args.chunk_id)

if __name__ == "__main__":
    main()