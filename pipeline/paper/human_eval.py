from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import Config
from pipeline.paper.common import bool_series, load_dataframe, write_json


DEFAULT_MODEL_SPECS = [
    "qwen2.5-3b-instruct:100d",
    "vicuna-13b-v1.5:1d",
    "llama-2-7b-chat-hf:1d",
    "llama-3.2-1b-instruct:100d",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample or summarize the paper's jailbreak-judge human audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Create the stratified human-eval sample.")
    sample.add_argument("--model-spec", action="append", default=None, help="MODEL_ALIAS:{100d,1d}. Repeatable.")
    sample.add_argument("--output", default="outputs/paper/human_eval/sampled_items.json")
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--n-positive", type=int, default=2)
    sample.add_argument("--n-negative", type=int, default=3)

    summarize = subparsers.add_parser("summarize", help="Summarize reviewed human-eval labels.")
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", default="outputs/paper/human_eval/summary.json")
    return parser.parse_args(argv)


parse_arguments = parse_args


def _transfer_path(alias: str, dimensionality: str) -> str:
    cfg = Config(alias)
    return cfg.single_seed_transfer_path() if dimensionality == "1d" else cfg.multi_seed_transfer_path()


def _parse_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Expected MODEL_ALIAS:100d or MODEL_ALIAS:1d, got {spec!r}")
    alias, dimensionality = spec.split(":", 1)
    if dimensionality not in {"100d", "1d"}:
        raise ValueError(f"Unknown dimensionality {dimensionality!r}")
    return alias, dimensionality


def _deduplicate_by_seed(group: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in group.columns:
        return group
    return group.drop_duplicates(subset=["seed"], keep="last")


def _sample_group(
    group: pd.DataFrame,
    *,
    rng: random.Random,
    model_alias: str,
    dimensionality: str,
    n_positive: int,
    n_negative: int,
) -> list[dict[str, Any]]:
    if dimensionality == "100d":
        group = _deduplicate_by_seed(group)

    labels = bool_series(group["jailbroken"])
    positives = group[labels]
    negatives = group[~labels]
    n_pos = min(n_positive, len(positives))
    n_neg = min(n_negative, len(negatives))
    if n_pos < n_positive:
        n_neg = min(n_positive + n_negative - n_pos, len(negatives))

    sampled_indices = []
    if n_pos:
        sampled_indices.extend(rng.sample(positives.index.to_list(), n_pos))
    if n_neg:
        sampled_indices.extend(rng.sample(negatives.index.to_list(), n_neg))

    records = []
    for _, item in group.loc[sampled_indices].iterrows():
        record = {
            "model": model_alias,
            "prompt_id": int(item["prompt_id"]),
            "suffix_id": int(item["suffix_id"]),
            "prompt": item.get("prompt"),
            "response": item.get("response"),
            "jailbroken": bool(item.get("jailbroken")),
            "human_eval": 0,
        }
        if "seed" in item and pd.notna(item["seed"]):
            record["seed"] = int(item["seed"])
        records.append(record)
    return records


def sample_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    all_samples: list[dict[str, Any]] = []
    specs = args.model_spec or DEFAULT_MODEL_SPECS

    for spec in specs:
        alias, dimensionality = _parse_spec(spec)
        df = load_dataframe(_transfer_path(alias, dimensionality))
        for _, group in df.groupby("prompt_id", sort=True):
            all_samples.extend(
                _sample_group(
                    group,
                    rng=rng,
                    model_alias=alias,
                    dimensionality=dimensionality,
                    n_positive=args.n_positive,
                    n_negative=args.n_negative,
                )
            )

    rng.shuffle(all_samples)
    return [{"eval_id": idx, **item} for idx, item in enumerate(all_samples)]


def summarize_items(path: str | Path) -> dict[str, Any]:
    df = load_dataframe(path)
    labels = bool_series(df["jailbroken"])
    disagreements = df["human_eval"].astype(int)
    by_model = {}
    for model, group in df.groupby("model", sort=True):
        group_labels = bool_series(group["jailbroken"])
        group_disagreements = group["human_eval"].astype(int)
        by_model[str(model)] = {
            "records": int(len(group)),
            "judge_jailbroken": int(group_labels.sum()),
            "judge_not_jailbroken": int((~group_labels).sum()),
            "disagreements": int(group_disagreements.sum()),
            "agreement_rate": float(1 - group_disagreements.mean()),
        }
    return {
        "records": int(len(df)),
        "judge_jailbroken": int(labels.sum()),
        "judge_not_jailbroken": int((~labels).sum()),
        "disagreements": int(disagreements.sum()),
        "agreement_rate": float(1 - disagreements.mean()),
        "by_model": by_model,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "sample":
        records = sample_items(args)
        path = write_json(records, args.output)
        print(f"Wrote {len(records)} sampled records to {path}")
        return

    summary = summarize_items(args.input)
    path = write_json(summary, args.output)
    print(f"Wrote {path}")
    print(summary)


if __name__ == "__main__":
    main()

