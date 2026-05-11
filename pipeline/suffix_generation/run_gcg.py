import argparse
import json
import os

import jailbreakbench as jbb
import nanogcg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Run GCG suffix generation for one prompt index.')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--num-steps', type=int, default=500)
    parser.add_argument('--end-seed', type=int, default=99)
    parser.add_argument('--results-dir', default='data/gcg_results/raw')
    return parser.parse_args()


def run_one(args, seed):
    config = nanogcg.GCGConfig(num_steps=args.num_steps, seed=seed, early_stop=True)
    result = nanogcg.run(args.model, args.tokenizer, args.goal, args.target, config)
    return {
        'index': args.index,
        'seed': seed,
        'goal': args.goal,
        'target': args.target,
        'behavior': args.behavior,
        'category': args.category,
        'num_steps': args.num_steps,
        'best_loss': result.best_loss,
        'best_string': result.best_string,
        'losses': result.losses,
        'strings': result.strings,
    }


def main():
    args = parse_args()
    if not 0 <= args.index <= 99:
        raise ValueError('index must be between 0 and 99')
    if not 0 <= args.end_seed <= 99:
        raise ValueError('end-seed must be between 0 and 99')

    args.model_name = args.model_name or os.path.basename(args.model_id)
    args.model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to('cuda:0')
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    dataset = jbb.read_dataset()
    args.behavior = dataset.behaviors[args.index]
    args.goal = dataset.goals[args.index]
    args.target = dataset.targets[args.index]
    args.category = dataset.categories[args.index]

    for seed in range(args.end_seed + 1):
        out_dir = os.path.join(args.results_dir, args.model_name, f'index-{args.index:04d}', f'seed-{seed:04d}')
        out_path = os.path.join(out_dir, 'results.json')
        if os.path.exists(out_path):
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f'Running GCG for index={args.index}, seed={seed}')
        with open(out_path, 'w') as f:
            json.dump(run_one(args, seed), f, indent=2)


if __name__ == '__main__':
    main()
