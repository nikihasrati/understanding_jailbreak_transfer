from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pipeline.config import Config
from pipeline.paper.common import ensure_parent


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate independent sentence-transformer embeddings for JailbreakBench prompts."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--sentence-model", default="all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


parse_arguments = parse_args


def default_output_path(sentence_model: str) -> str:
    safe_name = sentence_model.replace("/", "_")
    return str(Path("outputs") / "paper" / "prompt_embeddings" / f"{safe_name}_prompt_embeddings.pt")


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for independent prompt embeddings. "
            "Install it in the reproduction environment before running this command."
        ) from exc

    import pandas as pd

    cfg = Config(args.model_path)
    prompts = pd.read_json(cfg.prompts_path())["prompt"].to_list()
    model = SentenceTransformer(args.sentence_model)
    batches = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        batches.append(model.encode(batch, convert_to_tensor=True).detach().cpu())
    embeddings = torch.cat(batches, dim=0)
    output = ensure_parent(args.output or default_output_path(args.sentence_model))
    torch.save(embeddings, output)
    print(f"Wrote embeddings with shape {tuple(embeddings.shape)} to {output}")


if __name__ == "__main__":
    main()

