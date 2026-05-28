from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from pipeline.config import Config
from pipeline.paper.common import (
    add_source_prompt_ids,
    bool_series,
    cross_model_activation_dir,
    cross_model_artifact_path,
    filter_transfer_rows,
    load_dataframe,
    write_dataframe,
)
from pipeline.paper.tensors import cosine_matrix, iter_tensor_chunks, load_tensor_artifact, select_layer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build refusal-connectivity, suffix-push, and orthogonal-shift datasets for paper GLMMs."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--source-model-path", "--source_model_path", dest="source_model_path", default=None)
    parser.add_argument("--dimensionality", choices=["100d", "1d"], default="100d")
    parser.add_argument("--transfer-path", default=None)
    parser.add_argument("--prompt-activations-path", default=None)
    parser.add_argument("--jailbreak-activations-path", default=None)
    parser.add_argument("--refusal-direction-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-suffixes-per-prompt", type=int, default=100)
    parser.add_argument("--include-source", action="store_true", help="Keep rows where source prompt equals target prompt.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional smoke-test cap after source-prompt filtering.")
    return parser.parse_args(argv)


parse_arguments = parse_args


def default_transfer_path(cfg: Config, dimensionality: str, source_cfg: Config | None = None) -> str:
    if source_cfg is not None:
        return str(cross_model_artifact_path(cfg.repo_root, source_cfg.model_alias, cfg.model_alias))
    return cfg.single_seed_transfer_path() if dimensionality == "1d" else cfg.multi_seed_transfer_path()


def default_jailbreak_activations_path(cfg: Config, dimensionality: str, source_cfg: Config | None = None) -> str:
    if source_cfg is not None:
        return str(cross_model_activation_dir(cfg.repo_root, source_cfg.model_alias, cfg.model_alias))
    if dimensionality == "1d":
        return cfg.jailbreak_activations_dir()
    return cfg.multi_seed_jailbreak_activations_transfer_dir()


def default_output_path(cfg: Config, dimensionality: str, source_cfg: Config | None = None) -> str:
    if source_cfg is not None:
        return str(
            Path("outputs")
            / "paper"
            / "feature_datasets"
            / f"{source_cfg.model_alias}_to_{cfg.model_alias}_{dimensionality}_features.csv"
        )
    return str(Path("outputs") / "paper" / "feature_datasets" / f"{cfg.model_alias}_{dimensionality}_features.csv")


def _projection(vectors: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    denom = torch.dot(direction, direction).clamp(min=1e-8)
    scale = torch.matmul(vectors, direction) / denom
    return scale.unsqueeze(1) * direction.unsqueeze(0)


def _chunk_features(
    records: pd.DataFrame,
    prompt_vectors: torch.Tensor,
    refusal_direction: torch.Tensor,
    jailbreak_vectors: torch.Tensor,
    semantic_sim: torch.Tensor,
) -> pd.DataFrame:
    target_ids = torch.as_tensor(records["prompt_id"].astype(int).to_numpy(), dtype=torch.long)
    source_ids = torch.as_tensor(records["source_prompt_id"].astype(int).to_numpy(), dtype=torch.long)
    base = prompt_vectors.index_select(0, target_ids).float()
    suffix = jailbreak_vectors.float()
    direction = refusal_direction.float()

    baseline_score = torch.matmul(base, direction)
    suffix_score = torch.matmul(suffix, direction)
    suffix_push = baseline_score - suffix_score
    base_orth = base - _projection(base, direction)
    suffix_orth = suffix - _projection(suffix, direction)
    orthogonal_shift = torch.linalg.vector_norm(suffix_orth - base_orth, dim=1)
    semantic_sim_model = semantic_sim[source_ids, target_ids]

    return pd.DataFrame(
        {
            "prompt_index": records["prompt_id"].astype(int).to_numpy(),
            "target_prompt_id": records["prompt_id"].astype(int).to_numpy(),
            "source_prompt_index": records["source_prompt_id"].astype(int).to_numpy(),
            "suffix_index": records["suffix_id"].astype(int).to_numpy(),
            "suffix_id": records["suffix_id"].astype(int).to_numpy(),
            "seed": records["seed"].astype(int).to_numpy() if "seed" in records.columns else -1,
            "baseline_score": baseline_score.cpu().numpy(),
            "suffix_score": suffix_score.cpu().numpy(),
            "suffix_push": suffix_push.cpu().numpy(),
            "orthogonal_shift": orthogonal_shift.cpu().numpy(),
            "effective_ratio": (suffix_push / orthogonal_shift.clamp(min=1e-8)).cpu().numpy(),
            "semantic_sim_model": semantic_sim_model.cpu().numpy(),
            "jailbreak_success": bool_series(records["jailbroken"]).astype(int).to_numpy(),
            "jailbroken": bool_series(records["jailbroken"]).astype(int).to_numpy(),
        }
    )


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    cfg = Config(args.model_path)
    source_cfg = Config(args.source_model_path) if args.source_model_path else None
    one_dimensional = args.dimensionality == "1d"
    transfer_path = args.transfer_path or default_transfer_path(cfg, args.dimensionality, source_cfg)
    transfer_df = load_dataframe(transfer_path)
    transfer_df = add_source_prompt_ids(
        transfer_df,
        suffixes_per_prompt=args.num_suffixes_per_prompt,
        one_dimensional=one_dimensional,
        output_column="source_prompt_id",
    )

    prompt_activations_path = args.prompt_activations_path or cfg.prompt_activations_path()
    prompt_vectors = select_layer(
        load_tensor_artifact(prompt_activations_path),
        cfg.arditi_et_al_refusal_direction_layer(),
    ).float()
    semantic_sim = cosine_matrix(prompt_vectors)

    refusal_path = args.refusal_direction_path or cfg.arditi_et_al_refusal_direction_path()
    refusal_direction = torch.load(refusal_path, map_location="cpu", weights_only=True).float()

    activation_path = args.jailbreak_activations_path or default_jailbreak_activations_path(cfg, args.dimensionality, source_cfg)
    frames: list[pd.DataFrame] = []
    row_offset = 0
    remaining = args.max_records

    for activation_chunk in iter_tensor_chunks(activation_path):
        chunk_rows = activation_chunk.shape[0]
        records = transfer_df.iloc[row_offset : row_offset + chunk_rows].copy()
        records["_activation_row"] = range(len(records))
        row_offset += chunk_rows
        if records.empty:
            continue

        records = filter_transfer_rows(records, exclude_source=not args.include_source)
        if remaining is not None:
            if remaining <= 0:
                break
            records = records.head(remaining).copy()
            remaining -= len(records)
        if records.empty:
            continue

        local_indices = torch.as_tensor(records["_activation_row"].to_numpy(), dtype=torch.long)
        layer_vectors = select_layer(activation_chunk, cfg.arditi_et_al_refusal_direction_layer()).index_select(0, local_indices)
        frames.append(_chunk_features(records, prompt_vectors, refusal_direction, layer_vectors, semantic_sim))

        if remaining is not None and remaining <= 0:
            break

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "model_alias", cfg.model_alias)
    result.insert(1, "dimensionality", args.dimensionality)
    if source_cfg is not None:
        result.insert(1, "source_model_alias", source_cfg.model_alias)
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(args.model_path)
    source_cfg = Config(args.source_model_path) if args.source_model_path else None
    output = args.output or default_output_path(cfg, args.dimensionality, source_cfg)
    df = build_dataset(args)
    path = write_dataframe(df, output)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
