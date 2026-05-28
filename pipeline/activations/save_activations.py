from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from pipeline.artifacts import load_artifact_dataframe
from pipeline.config import Config
from pipeline.paper.common import cross_model_activation_dir, cross_model_artifact_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Regenerate activation tensors.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--input-kind', choices=['prompts', 'suffixes', 'cross_prompt_jailbreak', 'multi_seed_jailbreak', 'cross_model_jailbreak'], required=True)
    parser.add_argument('--source-model-path', default=None, help='Source model for --input-kind cross_model_jailbreak.')
    parser.add_argument('--output-format', choices=['canonical_tensor_chunks'], default='canonical_tensor_chunks')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    return parser.parse_args(argv)


def load_inputs(cfg: Config, input_kind: str, source_cfg: Config | None = None):
    if input_kind == 'prompts':
        return pd.read_json(cfg.prompts_path())['prompt'].to_list(), cfg.prompt_activations_path()
    if input_kind == 'suffixes':
        return pd.read_json(cfg.suffixes_path())['suffix'].to_list(), cfg.suffix_activations_path()
    if input_kind == 'cross_prompt_jailbreak':
        return load_artifact_dataframe(cfg.single_seed_transfer_path())['jailbreak'].to_list(), cfg.jailbreak_activations_dir()
    if input_kind == 'multi_seed_jailbreak':
        return load_artifact_dataframe(cfg.multi_seed_transfer_path())['jailbreak'].to_list(), cfg.multi_seed_jailbreak_activations_transfer_dir()
    if input_kind == 'cross_model_jailbreak':
        if source_cfg is None:
            raise ValueError('--source-model-path is required for cross_model_jailbreak activations')
        transfer_path = cross_model_artifact_path(cfg.repo_root, source_cfg.model_alias, cfg.model_alias)
        output_dir = cross_model_activation_dir(cfg.repo_root, source_cfg.model_alias, cfg.model_alias)
        return load_artifact_dataframe(transfer_path)['jailbreak'].to_list(), str(output_dir)
    raise ValueError(input_kind)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(args.model_path)
    source_cfg = Config(args.source_model_path) if args.source_model_path else None
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.generate_activations import get_activations
    model = construct_model_base(cfg.model_path)
    inputs, output_dir = load_inputs(cfg, args.input_kind, source_cfg=source_cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), args.batch_size)):
            acts = get_activations(model, inputs[i:i + args.batch_size], model.model_block_modules)
            batches.append(acts.detach().half().cpu())
    activations = torch.cat(batches, dim=0)
    chunks = torch.chunk(activations, args.num_chunks, dim=0)
    manifest = {'artifact_format': args.output_format, 'input_kind': args.input_kind, 'num_examples': len(inputs), 'num_chunks': len(chunks), 'chunks': []}
    for i, chunk in enumerate(chunks):
        path = output_dir / f'activations_chunk_{i:05d}.pt'
        torch.save(chunk, path)
        manifest['chunks'].append({'path': path.name, 'shape': list(chunk.shape)})
    with (output_dir / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
