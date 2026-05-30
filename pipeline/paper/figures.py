from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from pipeline.config import Config
from pipeline.paper.common import DISPLAY_NAMES, bool_series, ensure_parent, load_dataframe, write_dataframe
from pipeline.paper.feature_dataset import default_output_path as default_feature_dataset_path
from pipeline.paper.tensors import dot_with_direction, load_tensor_artifact, select_layer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible paper figure artifacts from migrated analysis code.")
    subparsers = parser.add_subparsers(dest="figure", required=True)

    density = subparsers.add_parser(
        "refusal-density",
        help="Plot prompt activation alignment with the stored refusal direction.",
    )
    density.add_argument("--model-path", "--model_path", dest="model_paths", action="append", required=True)
    density.add_argument("--prompt-activations-path", action="append", default=None)
    density.add_argument("--refusal-direction-path", action="append", default=None)
    density.add_argument("--output", default=None)
    density.add_argument("--csv-output", default=None)

    geometry = subparsers.add_parser(
        "suffix-geometry",
        help="Plot suffix-push versus orthogonal-shift summaries from a paper feature dataset.",
    )
    geometry.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    geometry.add_argument("--source-model-path", "--source_model_path", dest="source_model_path", default=None)
    geometry.add_argument("--dimensionality", choices=["100d", "1d"], default="100d")
    geometry.add_argument("--features-path", default=None)
    geometry.add_argument("--output", default=None)
    geometry.add_argument("--csv-output", default=None)
    return parser.parse_args(argv)


parse_arguments = parse_args


def _optional_index(values: list[str] | None, index: int, total: int) -> str | None:
    if values is None:
        return None
    if len(values) == 1 and total > 1:
        return values[0]
    if len(values) != total:
        raise ValueError(f"Expected 1 or {total} paths, got {len(values)}.")
    return values[index]


def _figure_path(name: str) -> Path:
    return Path("outputs") / "paper" / "figures" / name


def default_refusal_density_output(model_paths: list[str]) -> Path:
    if len(model_paths) == 1:
        alias = Config(model_paths[0]).model_alias
        return _figure_path(f"{alias}_refusal_density.png")
    return _figure_path("refusal_density_comparison.png")


def default_suffix_geometry_output(cfg: Config, dimensionality: str, source_cfg: Config | None = None) -> Path:
    if source_cfg is None:
        return _figure_path(f"{cfg.model_alias}_{dimensionality}_suffix_geometry.png")
    return _figure_path(f"{source_cfg.model_alias}_to_{cfg.model_alias}_{dimensionality}_suffix_geometry.png")


def build_refusal_density_frame(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    total = len(args.model_paths)
    for index, model_path in enumerate(args.model_paths):
        cfg = Config(model_path)
        prompt_activations_path = _optional_index(args.prompt_activations_path, index, total) or cfg.prompt_activations_path()
        refusal_direction_path = _optional_index(args.refusal_direction_path, index, total) or cfg.arditi_et_al_refusal_direction_path()
        layer = cfg.arditi_et_al_refusal_direction_layer()

        vectors = select_layer(load_tensor_artifact(prompt_activations_path), layer).float()
        direction = torch.load(refusal_direction_path, map_location="cpu", weights_only=True).float()
        dot = dot_with_direction(vectors, direction)
        cosine = torch.nn.functional.cosine_similarity(vectors, direction.unsqueeze(0), dim=1)
        max_abs_dot = dot.abs().max().clamp(min=1e-8)
        normalized_dot = dot / max_abs_dot
        frames.append(
            pd.DataFrame(
                {
                    "model_alias": cfg.model_alias,
                    "model": DISPLAY_NAMES.get(cfg.model_alias, cfg.model_name()),
                    "prompt_index": range(len(vectors)),
                    "layer": layer,
                    "dot_product": dot.cpu().numpy(),
                    "normalized_dot_product": normalized_dot.cpu().numpy(),
                    "cosine_with_refusal": cosine.cpu().numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_refusal_density(df: pd.DataFrame, output: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = ensure_parent(output)
    fig, (cosine_ax, dot_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (cosine_ax, "cosine_with_refusal", "Cosine similarity with refusal direction"),
        (dot_ax, "normalized_dot_product", "Normalized dot product with refusal direction"),
    ]
    for ax, column, label in panels:
        min_group_size = int(df.groupby("model")[column].nunique().min())
        if min_group_size >= 2:
            sns.kdeplot(data=df, x=column, hue="model", common_norm=False, fill=False, ax=ax)
        else:
            sns.histplot(data=df, x=column, hue="model", bins=20, element="step", stat="density", ax=ax)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    cosine_ax.set_title("Cosine")
    dot_ax.set_title("Dot Product")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def build_suffix_geometry_frame(args: argparse.Namespace) -> pd.DataFrame:
    cfg = Config(args.model_path)
    source_cfg = Config(args.source_model_path) if args.source_model_path else None
    features_path = args.features_path or default_feature_dataset_path(cfg, args.dimensionality, source_cfg)
    df = load_dataframe(features_path).copy()
    if "jailbreak_success" not in df.columns and "jailbroken" in df.columns:
        df["jailbreak_success"] = bool_series(df["jailbroken"]).astype(int)
    required = {"suffix_id", "suffix_push", "orthogonal_shift", "jailbreak_success"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{features_path} is missing required columns: {', '.join(missing)}")

    grouped = (
        df.groupby("suffix_id", as_index=False)
        .agg(
            suffix_push=("suffix_push", "mean"),
            orthogonal_shift=("orthogonal_shift", "mean"),
            jailbreak_success=("jailbreak_success", "mean"),
            records=("jailbreak_success", "size"),
        )
        .sort_values("suffix_id")
        .reset_index(drop=True)
    )
    grouped.insert(0, "model_alias", cfg.model_alias)
    grouped.insert(1, "dimensionality", args.dimensionality)
    if source_cfg is not None:
        grouped.insert(1, "source_model_alias", source_cfg.model_alias)
    return grouped


def plot_suffix_geometry(df: pd.DataFrame, output: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ensure_parent(output)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sizes = 18 + df["records"].clip(lower=1).pow(0.5) * 6
    scatter = ax.scatter(
        df["suffix_push"],
        df["orthogonal_shift"],
        c=df["jailbreak_success"],
        s=sizes,
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Mean ASR")
    ax.set_xlabel("Suffix push")
    ax.set_ylabel("Orthogonal shift")
    ax.set_title("Suffix geometry")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def run_refusal_density(args: argparse.Namespace) -> None:
    output = Path(args.output) if args.output else default_refusal_density_output(args.model_paths)
    df = build_refusal_density_frame(args)
    if args.csv_output:
        csv_path = write_dataframe(df, args.csv_output)
        print(f"Wrote {len(df)} rows to {csv_path}")
    figure_path = plot_refusal_density(df, output)
    print(f"Wrote {figure_path}")


def run_suffix_geometry(args: argparse.Namespace) -> None:
    cfg = Config(args.model_path)
    source_cfg = Config(args.source_model_path) if args.source_model_path else None
    output = Path(args.output) if args.output else default_suffix_geometry_output(cfg, args.dimensionality, source_cfg)
    df = build_suffix_geometry_frame(args)
    if args.csv_output:
        csv_path = write_dataframe(df, args.csv_output)
        print(f"Wrote {len(df)} rows to {csv_path}")
    figure_path = plot_suffix_geometry(df, output)
    print(f"Wrote {figure_path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.figure == "refusal-density":
        run_refusal_density(args)
    elif args.figure == "suffix-geometry":
        run_suffix_geometry(args)
    else:
        raise ValueError(args.figure)


if __name__ == "__main__":
    main()
