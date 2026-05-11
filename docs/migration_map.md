# Migration Map

The old repositories were treated as read-only sources. This repo is the consolidated, publishable working tree for reproducing the paper workflow. See [paper_map.md](paper_map.md) for how each migrated entry point maps to the paper.

## Table of Contents

- [Included](#included)
- [Explicitly Excluded](#explicitly-excluded)

## Included

- GCG suffix generation from `jbb-transfer`, migrated to `pipeline/suffix_generation/run_gcg.py`.
- Multi-seed dataset creation and validation, combined into `pipeline/suffix_generation/create_datasets.py`.
- Completion generation and resumable generation under `pipeline/generation/`.
- Jailbreak-judge evaluation and jailbreak-label normalization under `pipeline/evaluation/`.
- Activation regeneration commands and export formats under `pipeline/activations/`.
- Multi-seed analysis under `pipeline/analysis/`.
- Cross-model transfer setup, generation, evaluation, combination, checks, and analysis under `pipeline/cross_model/`.
- GCG-push and prompt-rephrasing workflows.
- JSON artifacts as manifest-backed chunks.
- Raw per-seed GCG result JSON files for v1.
- Refusal-direction vectors and metadata under normal Git file limits.
- Model wrapper support files for the external refusal-direction repository under `external/refusal_direction_model_utils/`.

## Explicitly Excluded

- Prompt clustering scripts.
- Predictor scripts.
- BERT prompt embedding scripts.
- Activation tensor artifacts.
- Slurm logs, symlinks, job summaries, virtual environments, caches, and personal launch wrappers.

## GCG-Push Consolidation

The old local `gcg-push` repo was treated as read-only. The reproducible pieces were consolidated into the new repo as follows:

```text
gcg-push/gcg.py
  -> pipeline/gcg_push/run_gcg.py
  -> pipeline/gcg_push/gcg_adapted.py

gcg-push/setup_datasets.py + check_suffixes.py
  -> pipeline/gcg_push/create_datasets.py

gcg-push/20_random_indices.txt
  -> configs/gcg_push_20_random_indices.txt

gcg-push/scripts/* and launch_job*.sh
  -> configs/gcg_push_paper.example.yaml
  -> pipeline/gcg_push/launch_experiment.py

gcg-push/processed_datasets and evaluated results used by the paper
  -> data/gcg_push_results/<model>/<intervention>/coeff-<coefficient>/<split>/manifest.json + chunks/
```

Slurm logs, personal script folders, and old machine-specific paths were intentionally not migrated. The new config file uses placeholders for cluster-specific settings and avoids personal names or email addresses.
