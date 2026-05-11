from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jailbreakbench as jbb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.gcg_push import config as gcg_config
from pipeline.gcg_push.gcg_adapted import GCGConfig, run


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run coefficient-modified GCG for one JailbreakBench prompt index.')
    parser.add_argument('--config', default=str(gcg_config.DEFAULT_CONFIG))
    parser.add_argument('--intervention', required=True, choices=['suffix_push', 'orth_shift'])
    parser.add_argument('--coeff', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--seed', type=int, default=None, help='Run one seed. Defaults to the configured seed range.')
    parser.add_argument('--seed-start', type=int, default=None)
    parser.add_argument('--seed-end', type=int, default=None)
    parser.add_argument('--model-id', default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--raw-results-root', default=None)
    parser.add_argument('--refusal-direction-path', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--torch-dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--force', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def torch_dtype(name: str):
    return {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[name]


def jsonable(value: Any):
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def output_path(config: dict[str, Any], args: argparse.Namespace, seed: int) -> Path:
    if args.raw_results_root:
        root = gcg_config.repo_path(args.raw_results_root)
        return root / gcg_config.model_alias(config) / gcg_config.experiment_name(config) / args.intervention / gcg_config.coefficient_dir(args.coeff) / f'index-{args.index:04d}' / f'seed-{seed:04d}' / 'results.json'
    return gcg_config.raw_result_path(config, args.intervention, args.coeff, args.index, seed)


def load_refusal_direction(config: dict[str, Any], args: argparse.Namespace):
    path = Path(args.refusal_direction_path) if args.refusal_direction_path else gcg_config.refusal_direction_path(config)
    if not path.is_absolute():
        path = gcg_config.REPO_ROOT / path
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def build_gcg_config(config: dict[str, Any], args: argparse.Namespace, seed: int) -> GCGConfig:
    intervention_cfg = gcg_config.interventions(config)[args.intervention]
    suffix_push_coeff = float(args.coeff) if args.intervention == 'suffix_push' else 0.0
    orthogonal_shift_coeff = float(args.coeff) if args.intervention == 'orth_shift' else 0.0
    return GCGConfig(
        num_steps=args.num_steps or gcg_config.num_steps(config),
        seed=seed,
        early_stop=False,
        suffix_push_loss_type=gcg_config.loss_type(config, 'suffix_push') if 'suffix_push' in gcg_config.interventions(config) else 'mse',
        suffix_push_coeff=suffix_push_coeff,
        orthogonal_shift_loss_type=gcg_config.loss_type(config, 'orth_shift') if 'orth_shift' in gcg_config.interventions(config) else 'mse',
        orthogonal_shift_coeff=orthogonal_shift_coeff,
        activation_layer=intervention_cfg.get('activation_layer'),
    )


def seed_values(config: dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [args.seed]
    start, end = gcg_config.seed_range(config, args.intervention)
    if args.seed_start is not None:
        start = args.seed_start
    if args.seed_end is not None:
        end = args.seed_end
    return list(range(start, end + 1))


def main(argv=None) -> None:
    args = parse_args(argv)
    config = gcg_config.load_config(args.config)
    if args.model_id:
        config.setdefault('experiment', {})['model_id'] = args.model_id
        config['experiment']['model_alias'] = os.path.basename(args.model_id).lower()

    model_id = gcg_config.model_id(config)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype(args.torch_dtype)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    refusal_direction = load_refusal_direction(config, args)

    dataset = jbb.read_dataset()
    behavior = dataset.behaviors[args.index]
    goal = dataset.goals[args.index]
    target = dataset.targets[args.index]
    category = dataset.categories[args.index]

    for seed in seed_values(config, args):
        out_path = output_path(config, args, seed)
        if out_path.exists() and not args.force:
            print(f'Skipping existing {out_path}')
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Running {args.intervention} coeff={args.coeff} index={args.index} seed={seed}')
        result = run(model, tokenizer, goal, target, build_gcg_config(config, args, seed), refusal_direction)
        record = {
            'index': args.index,
            'seed': seed,
            'intervention': args.intervention,
            'coeff': args.coeff,
            'suffix_push_coeff': float(args.coeff) if args.intervention == 'suffix_push' else 0.0,
            'orthogonal_shift_coeff': float(args.coeff) if args.intervention == 'orth_shift' else 0.0,
            'goal': goal,
            'target': target,
            'behavior': behavior,
            'category': category,
            'num_steps': args.num_steps or gcg_config.num_steps(config),
            'best_loss': jsonable(result.best_loss),
            'best_string': result.best_string,
            'losses': jsonable(result.losses),
            'strings': jsonable(result.strings),
        }
        out_path.write_text(json.dumps(record, indent=2) + '\n')
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
