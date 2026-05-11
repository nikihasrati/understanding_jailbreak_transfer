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
