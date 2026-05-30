from __future__ import annotations

import argparse
import json


RESULTS = [
    {
        "id": "figure_1",
        "paper_location": "Section 5.1",
        "description": "Intra-model transfer with one suffix per prompt.",
        "commands": ["python -m pipeline analysis paper --model-path MODEL"],
    },
    {
        "id": "figure_2",
        "paper_location": "Section 5.1",
        "description": "Multi-seed intra-model transfer matrices.",
        "commands": ["python -m pipeline analysis multi-seed --model-path MODEL"],
    },
    {
        "id": "figure_3",
        "paper_location": "Section 5.1",
        "description": "Inter-model transfer between Llama 3.2 and Qwen.",
        "commands": ["python -m pipeline cross-model analyze --source-model-path SOURCE --target-model-path TARGET"],
    },
    {
        "id": "tables_1_10",
        "paper_location": "Sections 5.2 and Appendix E.1",
        "description": "Semantic-similarity mixed-effects regressions.",
        "commands": [
            "python -m pipeline paper prompt-embeddings --model-path MODEL",
            "python -m pipeline paper semantic-dataset --model-path MODEL --embedding model --dimensionality 100d",
            "python -m pipeline paper semantic-dataset --model-path MODEL --embedding independent --embeddings-path EMBEDDINGS",
            "Rscript scripts/r/semantics_model.R --data-dir outputs/paper/semantic_datasets --results-dir outputs/paper/r/semantics",
        ],
    },
    {
        "id": "figures_4_9",
        "paper_location": "Sections 5.3-5.4 and Appendix C",
        "description": "Refusal-connectivity, cross-layer suppression, suffix-push, and orthogonal-shift figures.",
        "commands": [
            "python -m pipeline activations save --model-path MODEL --input-kind prompts",
            "python -m pipeline activations save --model-path MODEL --input-kind multi_seed_jailbreak",
            "python -m pipeline paper feature-dataset --model-path MODEL --dimensionality 100d",
            "python -m pipeline paper figures refusal-density --model-path MODEL",
            "python -m pipeline paper figures suffix-geometry --model-path MODEL --dimensionality 100d",
        ],
    },
    {
        "id": "tables_2_3_8_9_11_12_13",
        "paper_location": "Sections 5.4-5.5 and Appendices C/E",
        "description": "Single-feature, joint-effect, interaction, and semantic-extended mixed-effects regressions.",
        "commands": [
            "python -m pipeline paper feature-dataset --model-path MODEL --dimensionality 100d",
            "python -m pipeline activations save --model-path TARGET --source-model-path SOURCE --input-kind cross_model_jailbreak",
            "python -m pipeline paper feature-dataset --model-path TARGET --source-model-path SOURCE --dimensionality 1d",
            "Rscript scripts/r/full_random_effects_model.R --data-dir outputs/paper/feature_datasets --results-dir outputs/paper/r/features",
        ],
    },
    {
        "id": "tables_5_6",
        "paper_location": "Section 5.6",
        "description": "Altered GCG loss intervention ASR summaries.",
        "commands": ["python -m pipeline gcg-push analyze --config configs/gcg_push_paper.example.yaml"],
    },
    {
        "id": "table_7",
        "paper_location": "Appendix B",
        "description": "Model-selection metadata and suffix counts.",
        "commands": ["cat configs/models.example.yaml"],
    },
    {
        "id": "human_eval",
        "paper_location": "Appendix B",
        "description": "2,000-response human validation of the jailbreak judge.",
        "commands": ["python -m pipeline paper human-eval summarize --input data/human_eval/reviewed_sampled_items"],
    },
    {
        "id": "prompt_rephrasing",
        "paper_location": "Section 5.6 and Appendix C",
        "description": "Prompt rephrasing ASR-vs-refusal-dot-product intervention.",
        "commands": [
            "python -m pipeline prompt-rephrasings generate --model-path MODEL",
            "python -m pipeline prompt-rephrasings setup --model-path MODEL",
            "python -m pipeline paper prompt-rephrasing-analysis --model-path MODEL",
        ],
    },
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the paper result reproduction registry.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


parse_arguments = parse_args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.json:
        print(json.dumps(RESULTS, indent=2))
        return
    for item in RESULTS:
        print(f"{item['id']}: {item['paper_location']}")
        print(f"  {item['description']}")
        for command in item["commands"]:
            print(f"  - {command}")


if __name__ == "__main__":
    main()
