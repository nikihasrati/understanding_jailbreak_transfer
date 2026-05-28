from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.paper.common import add_source_prompt_ids, bool_series, filter_transfer_rows, load_dataframe, write_dataframe
from pipeline.paper.tensors import cosine_matrix, load_tensor_artifact, select_layer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the semantic-similarity GLMM dataset used for paper Tables 1 and 10."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--embedding", choices=["model", "independent"], default="model")
    parser.add_argument("--dimensionality", choices=["100d", "1d"], default="100d")
    parser.add_argument("--transfer-path", default=None)
    parser.add_argument("--prompt-activations-path", default=None)
    parser.add_argument(
        "--embeddings-path",
        default=None,
        help="Prompt embedding tensor/list for --embedding independent. Defaults are intentionally not guessed.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-suffixes-per-prompt", type=int, default=100)
    parser.add_argument("--include-source", action="store_true", help="Keep rows where source prompt equals target prompt.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional smoke-test cap after source-prompt filtering.")
    return parser.parse_args(argv)


parse_arguments = parse_args


def default_transfer_path(cfg: Config, dimensionality: str) -> str:
    return cfg.single_seed_transfer_path() if dimensionality == "1d" else cfg.multi_seed_transfer_path()


def default_output_path(cfg: Config, embedding: str, dimensionality: str) -> str:
    return str(Path("outputs") / "paper" / "semantic_datasets" / f"{cfg.model_alias}_{embedding}_{dimensionality}.csv")


def load_prompt_embeddings(args: argparse.Namespace, cfg: Config):
    if args.embedding == "independent":
        if not args.embeddings_path:
            raise ValueError("--embeddings-path is required for --embedding independent")
        embeddings = load_tensor_artifact(args.embeddings_path)
        return embeddings.float()

    activations_path = args.prompt_activations_path or cfg.prompt_activations_path()
    prompt_activations = load_tensor_artifact(activations_path)
    return select_layer(prompt_activations, cfg.arditi_et_al_refusal_direction_layer()).float()


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    cfg = Config(args.model_path)
    one_dimensional = args.dimensionality == "1d"
    transfer_path = args.transfer_path or default_transfer_path(cfg, args.dimensionality)
    transfer_df = load_dataframe(transfer_path)
    transfer_df = add_source_prompt_ids(
        transfer_df,
        suffixes_per_prompt=args.num_suffixes_per_prompt,
        one_dimensional=one_dimensional,
    )
    transfer_df = filter_transfer_rows(transfer_df, exclude_source=not args.include_source)
    if args.max_records is not None:
        transfer_df = transfer_df.head(args.max_records).copy()

    embeddings = load_prompt_embeddings(args, cfg)
    similarities = cosine_matrix(embeddings)

    source_ids = transfer_df["source_prompt_id"].astype(int).to_numpy()
    target_ids = transfer_df["prompt_id"].astype(int).to_numpy()
    result = pd.DataFrame(
        {
            "model_alias": cfg.model_alias,
            "embedding": args.embedding,
            "dimensionality": args.dimensionality,
            "source_prompt_id": source_ids,
            "suffix_id": transfer_df["suffix_id"].astype(int).to_numpy(),
            "target_prompt_id": target_ids,
            "cosine_similarity": similarities[source_ids, target_ids].cpu().numpy(),
            "jailbroken": bool_series(transfer_df["jailbroken"]).astype(int).to_numpy(),
        }
    )
    return result


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(args.model_path)
    output = args.output or default_output_path(cfg, args.embedding, args.dimensionality)
    df = build_dataset(args)
    path = write_dataframe(df, output)
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()

