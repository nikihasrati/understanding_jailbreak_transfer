# Understanding Jailbreak Transfer

This repository contains the code and reproducibility artifacts for the paper *Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models*. It is intended to be the entry point for reproducing the paper workflow: suffix generation, dataset construction, completion generation, jailbreak-judge evaluation, activation regeneration, and analysis.

Published JSON artifacts live in Git as chunked files with manifests. Large activation tensors are not committed; the commands to regenerate activations are documented in [docs/activations.md](docs/activations.md).

For a step-by-step guide to reproducing the paper results, start with the recommended Slurm job-array workflow in [docs/slurm_workflow.md](docs/slurm_workflow.md). Slurm is the recommended way to parallelize prompt and chunk jobs for full paper-scale reproduction. [docs/workflow.md](docs/workflow.md) provides the local-command equivalent for reference, but it has not been tested end to end.

## Table of Contents

- [Start Here](#start-here)
- [Workflow Summary](#workflow-summary)
- [Repository Layout](#repository-layout)

## Start Here

Read the docs in this order if you are reproducing results for the first time:

| File | Purpose |
| --- | --- |
| [docs/slurm_workflow.md](docs/slurm_workflow.md) | Recommended full-scale workflow using Slurm job arrays to parallelize jobs. Start here for reproduction runs. |
| [docs/workflow.md](docs/workflow.md) | Local-command equivalent of the Slurm workflow. Useful as a readable reference, but not tested end to end. |
| [docs/paper_map.md](docs/paper_map.md) | Mapping from paper sections, definitions, figures, and tables to repo scripts. |
| [docs/artifacts.md](docs/artifacts.md) | Data layout, chunked JSON manifests, checksums, and how to combine chunked files. |
| [docs/activations.md](docs/activations.md) | How to regenerate activations and export alternate activation layouts. |
| [docs/refusal_directions.md](docs/refusal_directions.md) | How refusal directions are stored, inspected, and regenerated. |
| [docs/migration_map.md](docs/migration_map.md) | What was consolidated from the old repositories and what was intentionally excluded. |
| [docs/future_work.md](docs/future_work.md) | Follow-up cleanup that is intentionally out of scope for this release. |

## Workflow Summary

This table provides an overview of what to run, in what order, what checkpoint or saved output to expect, and the relevant paper sections. For the detailed section-by-section mapping between the files in the repo and the sections in the paper, use [docs/paper_map.md](docs/paper_map.md).

Path shorthand:
- `<output-dir>`: usually `data/multiple_seed_results`.
- `<artifact-dir>`: usually `<output-dir>/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer`.

| Step | Workflow stage | Paper section(s) | Main module | Save location |
| --- | --- | --- | --- | --- |
| 1 | Generate suffixes | Section 4 | `pipeline.suffix_generation.run_gcg` | Saves raw GCG results to:<br>`data/gcg_results/raw/<model>/`<br>`index-*/seed-*/results.json` |
| 2 | Build datasets | Section 4; Section 5.1 | `pipeline.suffix_generation.create_datasets --check` | Saves JSON files under:<br>`<output-dir>/$MODEL_ALIAS/no_transfer/`<br>`<output-dir>/$MODEL_ALIAS/transfer/`<br>Prints prompt/seed coverage checks. |
| 3 | Publish JSON artifacts | Reproducibility support for Sections 4-5 | `tools.split_json_records`<br>`tools.verify_manifest` | Saves canonical chunks and manifest:<br>`<artifact-dir>/chunks/chunk_*.json`<br>`<artifact-dir>/manifest.json`<br>Manifest verification should pass. |
| 4 | Prepare generation jobs | Operational support for Section 4 | `tools.prepare_generation_chunks` | Saves temporary generation shards:<br>`<artifact-dir>/generation_chunks/chunk_*.json` |
| 5 | Generate responses | Section 4 | `pipeline.generation.generate_completions` | Updates generation shards with `response`:<br>`<artifact-dir>/generation_chunks/chunk_*.json` |
| 6 | Re-shard for evaluation | Operational support for Section 4 | `tools.combine_completions` | Saves fewer judge-ready shards:<br>`<artifact-dir>/evaluation_chunks/chunk_*.json` |
| 7 | Judge responses | Section 4; Definition 1; Section 5 | `pipeline.evaluation.evaluate_completions` | Updates evaluation shards with `jailbroken`:<br>`<artifact-dir>/evaluation_chunks/chunk_*.json` |
| 8 | Publish evaluated artifacts | Reproducibility support for Section 5 | `tools.combine_completions`<br>`--write-manifest` | Replaces canonical chunks and manifest:<br>`<artifact-dir>/chunks/chunk_*.json`<br>`<artifact-dir>/manifest.json` |
| 9 | Normalize labels | Definition 1; Section 5 | `pipeline.evaluation.normalize_jailbreak_labels` | Writes to the command's `--output`, commonly:<br>`<artifact-dir>/combined.json`<br>Split back to `chunks/` before committing. |
| 10 | Generate no-suffix baseline | Section 4; Section 5.1; Section 5.6 | `pipeline.generation.generate_completions`<br>`--no-suffix-completions`<br>`pipeline.evaluation.evaluate_completions`<br>`--no_suffix_completions` | Saves and updates:<br>`data/no_suffix_generations/`<br>`${MODEL_ALIAS}_no_suffix_generations/combined.json` |
| 11 | Regenerate activations | Sections 3.2-3.3; Sections 5.2-5.5 | `pipeline.activations.save_activations` | Saves ignored tensor chunks under:<br>`outputs/activations/$MODEL_ALIAS/`<br>`<input-kind>/canonical_tensor_chunks/` |
| 12 | Analyze multi-seed transfer | Sections 5.1-5.5 | `pipeline.analysis.multi_seed_data_analysis` | Prints ASR summaries.<br>Saves figures under:<br>`figures/$MODEL_ALIAS/` |
| 13 | Analyze cross-model transfer | Section 3.1; Section 5.1; Section 5.5 | `pipeline.cross_model.*` | Saves transfer records under:<br>`data/cross_model_transfer_generations/`<br>`${SOURCE_ALIAS}_to_${TARGET_ALIAS}/`<br>Saves figures under:<br>`figures/${SOURCE_ALIAS}_to_${TARGET_ALIAS}/` |
| 14 | Run and analyze GCG-push interventions | Section 5.6; Tables 4-5 | `pipeline.gcg_push.launch_experiment`<br>`pipeline.gcg_push.run_gcg`<br>`pipeline.gcg_push.create_datasets`<br>`pipeline.gcg_push.data_analysis` | Raw rerun outputs:<br>`outputs/gcg_push/raw/$MODEL_ALIAS/`<br>Published artifacts:<br>`data/gcg_push_results/$MODEL_ALIAS/`<br>`{suffix_push,orth_shift}/coeff-$COEFF/`<br>`{no_transfer,transfer}/`<br>Analysis summaries:<br>`outputs/gcg_push_analysis/` |
| 15 | Analyze prompt rephrasings | Section 5.6; Appendix C | `pipeline.prompt_rephrasings.setup_dataset`<br>`--rephrasings` generation/evaluation | Saves setup dataset to:<br>`data/prompt_rephrasings/`<br>`${MODEL_ALIAS}_prompt_rephrasings/combined.json`<br>Generation/evaluation chunks live under the same artifact directory. |

The recommended command-by-command workflow is in [docs/slurm_workflow.md](docs/slurm_workflow.md). The local-command equivalent is in [docs/workflow.md](docs/workflow.md), but it is provided as an untested reference rather than the primary reproduction path.

## Repository Layout

```text
understanding_jailbreak_transfer/
  README.md                         Project entry point and doc index.
  environment.yml                   Conda environment used for reproduction.
  configs/                          Example model, path, and Slurm settings.
    models.example.yaml             Model IDs used in the experiments.
    paths.example.yaml              Cache/output path placeholders.
    slurm.example.yaml              Cluster setting placeholders.
    gcg_push_paper.example.yaml     Config-driven GCG-push paper experiment.
    gcg_push_20_random_indices.txt  Prompt IDs used for the paper GCG-push runs.
  data/                             Versioned artifacts used by the paper workflow.
    processed/                      JailbreakBench prompts and suffixes as chunked JSON.
    multiple_seed_results/          Multi-seed no-transfer and transfer artifacts.
                                      Commit only chunks/ and manifest.json; generation_chunks/,
                                      evaluation_chunks/, and combined.json are temporary.
    gcg_results/raw/                Raw per-prompt/per-seed GCG result JSON files.
    gcg_push_results/               Coefficient-modified GCG-push artifacts by intervention.
    refusal_directions/             Stored refusal-direction vectors and metadata.
  docs/                             Reproducibility guide and reference docs.
  external/                         Small support files for external repositories.
    refusal_direction_model_utils/  Model wrappers to copy into the external refusal-direction repo.
  pipeline/                         Python pipeline modules.
    suffix_generation/              GCG runner plus dataset creation/checking.
    generation/                     Completion generation.
    evaluation/                     Jailbreak-judge evaluation and label normalization.
    activations/                    Activation regeneration and export utilities.
    analysis/                       Multi-seed analysis.
    cross_model/                    Cross-model setup, generation, evaluation, and analysis.
    gcg_push/                       Config-driven altered-GCG runner, artifact builder, launcher, and analysis.
    prompt_rephrasings/             Prompt-rephrasing dataset setup.
    model_utils/                    Model wrappers used by this repository.
  scripts/                          Local and Slurm convenience wrappers.
  tools/                            Artifact chunking, combining, checksum, and tensor utilities.
```
