import argparse
import os 

import pandas as pd

from tqdm import tqdm

from pipeline.artifacts import assert_column_populated, workflow_chunk_path
from pipeline.config import Config

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Set up data sets for cross-model analysis.')
    parser.add_argument('--source-model-path', '--source_model_path', dest='source_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--target-model-path', '--target_model_path', dest='target_model_path', type=str, required=True, help='Path to the target model')
    parser.add_argument('--chunk-id', '--chunk_id', dest='chunk_id', type=int, required=True, help='Chunk ID to process')
    return parser.parse_args(argv)

def generate_completions(target_model, source_cfg, target_cfg, chunk_id, batch_size=100):
    print("Chunk ID", chunk_id)

    save_dir = os.path.join(target_cfg.cross_model_transfer_generations_dir(), f'{source_cfg.model_alias}_to_{target_cfg.model_alias}')
    save_path = workflow_chunk_path(os.path.join(save_dir, 'cross_model_transfer_generations.json'), chunk_id, 'generation')

    chunk_df = pd.read_json(save_path)
    n = len(chunk_df)

    if 'response' not in chunk_df.columns:
        chunk_df['response'] = None
        chunk_df.to_json(save_path, orient='records', indent=4)

    # Start from the first index where response is not None
    start = chunk_df['response'].notna().sum()
    print(f"Resuming at index {start}/{n}")

    for batch_start in tqdm(range(start, n, batch_size)):
        batch_end = min(batch_start + batch_size, n)
        texts = chunk_df['jailbreak'].iloc[batch_start:batch_end].tolist()
        completions = target_model.generate_completions(texts, max_new_tokens=200)

        for i, completion in enumerate(completions):
            chunk_df.loc[batch_start + i, "response"] = completion
        
        chunk_df.to_json(save_path, orient='records', indent=4)

    assert_column_populated(chunk_df, 'response', save_path)

def main(argv=None):
    args = parse_args(argv)
    source_cfg = Config(model_path=args.source_model_path)
    target_cfg = Config(model_path=args.target_model_path)
    
    print("Source model path", args.source_model_path)
    print("Target model path", args.target_model_path)

    from pipeline.model_utils.model_factory import construct_model_base
    target_model = construct_model_base(target_cfg.model_path)
    generate_completions(target_model, source_cfg, target_cfg, args.chunk_id)

if __name__ == "__main__":
    main()
