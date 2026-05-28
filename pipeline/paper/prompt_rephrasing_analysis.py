from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import Config
from pipeline.paper.common import bool_series, load_dataframe, write_dataframe, write_json


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prompt-rephrasing ASR and refusal-dot-product changes for paper Section 5.6."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--evaluated-path", default=None)
    parser.add_argument("--baseline-transfer-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


parse_arguments = parse_args


def default_output_dir(cfg: Config) -> str:
    return str(Path("outputs") / "paper" / "prompt_rephrasings" / cfg.model_alias)


def _correlations(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 2:
        return {
            "pearson_correlation": None,
            "pearson_p_value": None,
            "spearman_correlation": None,
            "spearman_p_value": None,
        }
    from scipy import stats

    pearson = stats.pearsonr(df["asr_change"], df["dot_product_change"])
    spearman = stats.spearmanr(df["asr_change"], df["dot_product_change"])
    return {
        "pearson_correlation": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_correlation": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
    }


def build_analysis(evaluated_df: pd.DataFrame, baseline_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluated = evaluated_df.copy()
    evaluated["jailbroken_bool"] = bool_series(evaluated["jailbroken"])
    suffix_column = "old_suffix_id" if "old_suffix_id" in evaluated.columns else "suffix_id"

    rephrase_asr = (
        evaluated.groupby("rephrased_prompt_id", as_index=False)
        .agg(
            original_prompt_id=("original_prompt_id", "first"),
            prompt=("prompt", "first"),
            original_prompt=("original_prompt", "first"),
            ori_dot_product=("ori_dot_product", "first"),
            dot_product=("dot_product", "first"),
            similarity=("similarity", "first"),
            rephrase_asr=("jailbroken_bool", "mean"),
            suffixes=("suffix_id", "nunique"),
            records=("jailbroken_bool", "size"),
        )
    )

    relevant_suffixes = evaluated[["original_prompt_id", suffix_column]].drop_duplicates()
    baseline = baseline_df.copy()
    baseline["jailbroken_bool"] = bool_series(baseline["jailbroken"])
    baseline = baseline.merge(
        relevant_suffixes,
        left_on=["prompt_id", "suffix_id"],
        right_on=["original_prompt_id", suffix_column],
        how="inner",
    )
    original_asr = (
        baseline.groupby("original_prompt_id", as_index=False)
        .agg(original_asr=("jailbroken_bool", "mean"), original_records=("jailbroken_bool", "size"))
    )
    details = rephrase_asr.merge(original_asr, on="original_prompt_id", how="left")
    details["asr_change"] = details["rephrase_asr"] - details["original_asr"]
    details["dot_product_change"] = details["dot_product"] - details["ori_dot_product"]

    summary = {
        "rephrased_prompts": int(details["rephrased_prompt_id"].nunique()),
        "original_prompts": int(details["original_prompt_id"].nunique()),
        "evaluated_records": int(len(evaluated)),
        "mean_original_asr": float(details["original_asr"].mean()),
        "mean_rephrase_asr": float(details["rephrase_asr"].mean()),
        "mean_asr_change": float(details["asr_change"].mean()),
        "mean_dot_product_change": float(details["dot_product_change"].mean()),
    }
    summary.update(_correlations(details.dropna(subset=["asr_change", "dot_product_change"])))
    return details, summary


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(args.model_path)
    evaluated_path = args.evaluated_path or cfg.prompt_rephrasings_path()
    baseline_path = args.baseline_transfer_path or cfg.multi_seed_transfer_path()
    output_dir = Path(args.output_dir or default_output_dir(cfg))
    output_dir.mkdir(parents=True, exist_ok=True)

    details, summary = build_analysis(load_dataframe(evaluated_path), load_dataframe(baseline_path))
    details_path = write_dataframe(details, output_dir / "prompt_rephrasing_analysis.csv")
    summary_path = write_json(summary, output_dir / "summary.json")
    print(f"Wrote {details_path}")
    print(f"Wrote {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()

