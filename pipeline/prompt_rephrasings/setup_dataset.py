import argparse
import os

import pandas as pd

from pipeline.config import Config

def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Parse arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--input-path', default=None, help='Unprocessed rephrasing JSON. Defaults to the model-specific path under data/prompt_rephrasings/.')
    parser.add_argument('--output-path', default=None, help='Output JSON. Defaults to the model-specific prompt rephrasing combined.json path.')
    return parser.parse_args()

def setup_dataset(rephrasings_df, suffixes_df, output_path):
    cross_df = rephrasings_df.merge(suffixes_df, how='cross')
    cross_df['jailbreak'] = cross_df['prompt'] + cross_df['suffix']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cross_df.to_json(output_path, orient='records', indent=4)

def process_rephrasings(unprocessed_rephrasings_df):
    long_df = (
        pd.wide_to_long(
            unprocessed_rephrasings_df,
            stubnames=['paraphrase', 'dot_product', 'similarity'],
            i=['prompt_id', 'ori_dot_product'],
            j='variant_index',
            sep='_',
            suffix=r'\d+'
        )
        .reset_index()
        .dropna(subset=['paraphrase']) # keep only rows with paraphrases
    )

    long_df = long_df.rename(
        columns={
            'paraphrase': 'prompt',
            'prompt_id': 'original_prompt_id',
            'variant_index': 'prompt_rephrasing_number',
            'prompt': 'original_prompt'
        }
    )

    long_df.insert(0, 'rephrased_prompt_id', range(len(long_df)))

    result = long_df[
        [
            'rephrased_prompt_id',
            'original_prompt_id',
            'prompt',
            'original_prompt',
            'ori_dot_product',
            'dot_product',
            'similarity'
        ]
    ]

    return result

if __name__ == "__main__":
    args = parse_arguments()
    cfg = Config(model_path=args.model_path)
    
    input_path = args.input_path or cfg.unprocessed_prompt_rephrasings_path()
    output_path = args.output_path or cfg.prompt_rephrasings_path()
    unprocessed_rephrasings_df = pd.read_json(input_path)
    rephrasings_df = process_rephrasings(unprocessed_rephrasings_df)
    suffixes_df = pd.read_json(cfg.suffixes_path())

    # Setup the dataset
    setup_dataset(rephrasings_df, suffixes_df, output_path)