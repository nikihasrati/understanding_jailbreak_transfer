# Reproducibility Workflow

This is the local-command equivalent of the recommended Slurm reproducibility workflow. It includes environment setup, the command sequence, expected outputs, and links to supporting reference docs. It has not been tested end to end, so use it as a readable reference or for small debugging runs rather than as the primary paper-scale reproduction path. For a compact mapping from every entry point to the paper, see [paper_map.md](paper_map.md).

For full paper-scale runs, use the recommended Slurm workflow in [slurm_workflow.md](slurm_workflow.md). This file and [slurm_workflow.md](slurm_workflow.md) are intentionally almost identical: this file shows local `python` commands, while [slurm_workflow.md](slurm_workflow.md) shows the corresponding Slurm job-submission commands.

## Table of Contents

- [Chunking Convention](#chunking-convention)
- [Local Commands vs Slurm](#local-commands-vs-slurm)
- [0. Setup and Environment](#0-setup-and-environment)
- [1. Generate GCG Suffixes](#1-generate-gcg-suffixes)
- [2. Create and Check Multi-Seed Datasets](#2-create-and-check-multi-seed-datasets)
- [3. Canonical JSON Artifacts](#3-canonical-json-artifacts)
- [4. Prepare Generation Chunks](#4-prepare-generation-chunks)
- [5. Generate Completions](#5-generate-completions)
- [6. Combine Completions for Evaluation](#6-combine-completions-for-evaluation)
- [7. Evaluate Completions](#7-evaluate-completions)
- [8. Promote Evaluated Chunks Back to Canonical Artifacts](#8-promote-evaluated-chunks-back-to-canonical-artifacts)
- [9. Normalize Jailbreak Labels](#9-normalize-jailbreak-labels)
- [10. No-Suffix Baseline](#10-no-suffix-baseline)
- [11. Refusal Directions](#11-refusal-directions)
- [12. Activations](#12-activations)
- [13. Multi-Seed Analysis](#13-multi-seed-analysis)
- [14. Cross-Model Transfer](#14-cross-model-transfer)
- [15. GCG Push](#15-gcg-push)
- [16. Prompt Rephrasings](#16-prompt-rephrasings)

## Chunking Convention

This repo separates canonical artifacts from temporary job shards:

```text
chunks/              canonical published chunks committed with manifest.json
generation_chunks/   temporary chunks for completion-generation jobs
evaluation_chunks/   temporary chunks for jailbreak-judge evaluation jobs
combined.json        temporary full materialization for scripts that need all records
```

Only `chunks/` and `manifest.json` should be committed. `generation_chunks/`, `evaluation_chunks/`, and `combined.json` are ignored working files.

The completion/evaluation lifecycle is:

```text
chunks/ -> generation_chunks/ -> evaluation_chunks/ -> chunks/ + manifest.json
```

## Local Commands vs Slurm

The commands in this file are local commands. For chunked stages, each command processes one prompt index or one chunk per invocation, selected by variables such as `GCG_PROMPT_INDEX`, `GENERATION_CHUNK_ID`, `EVALUATION_CHUNK_ID`, `CROSS_MODEL_GENERATION_CHUNK_ID`, and `CROSS_MODEL_EVALUATION_CHUNK_ID`.

You can manually run multiple local commands in parallel with different IDs, but that is not recommended for full paper-scale reproduction. Use a workload manager/job scheduler, preferably Slurm job arrays on an HPC cluster, so each prompt or chunk is scheduled as an independent task. The recommended Slurm version of this same workflow is in [slurm_workflow.md](slurm_workflow.md).

## 0. Setup and Environment

Run commands from the repository root. Model IDs should match the models named in Paper Section 4 and Appendix B unless you are intentionally running an extension.

Create the conda environment once:

```bash
conda env create -f environment.yml
```

Activate the environment and define model/cache settings for each shell session:

```bash
conda activate understanding_jailbreak_transfer_env

export MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
export MODEL_ALIAS=llama-3.2-1b-instruct
export MODEL_CACHE_ROOT=/path/to/model/cache/root
export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

# Workflow paths used by commands below.
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NO_SUFFIX_ARTIFACT_DIR="data/no_suffix_generations/${MODEL_ALIAS}_no_suffix_generations"

# Workflow parameters used by commands below.
export GCG_PROMPT_INDEX=0
export GCG_NUM_STEPS=500
export GCG_END_SEED=99
export EXPECTED_PROMPTS=100
export EXPECTED_SEEDS=100
export JSON_RECORDS_PER_CHUNK=5000
export NUM_GENERATION_CHUNKS=32
export GENERATION_CHUNK_ID=0
export NUM_EVALUATION_CHUNKS=8
export EVALUATION_CHUNK_ID=0
export NUM_PUBLISHED_CHUNKS=50
export NUM_JUDGE_GPUS=4
export NUM_ACTIVATION_CHUNKS=32
export GCG_PUSH_COEFF=0.01
export NUM_CROSS_MODEL_GENERATION_CHUNKS=32
export CROSS_MODEL_GENERATION_CHUNK_ID=0
export NUM_CROSS_MODEL_EVALUATION_CHUNKS=8
export CROSS_MODEL_EVALUATION_CHUNK_ID=0
```

Use `configs/models.example.yaml` for the model IDs used in the paper. Set `HF_TOKEN` as well if any gated model checkpoints require Hugging Face authentication. Adjust the workflow parameters to match your local sharding and rerun target.

**Output:** The setup step creates the conda environment and shell variables only; it does not write paper artifacts.

## 1. Generate GCG Suffixes

**Paper connection:** Section 4, "Generation of adversarial suffixes"; Section 5.1 Figures 1-2; Appendix B model/suffix counts.

`pipeline.suffix_generation.run_gcg` runs the GCG attack for one JailbreakBench harmful prompt index and a seed range. Each seed produces a candidate adversarial suffix for the prompt. The output is the raw result tree under `data/gcg_results/raw/`.

```bash
python -m pipeline.suffix_generation.run_gcg \
  --model-id "$MODEL_ID" \
  --index "$GCG_PROMPT_INDEX" \
  --num-steps "$GCG_NUM_STEPS" \
  --end-seed "$GCG_END_SEED" \
  --results-dir data/gcg_results/raw
```

Run this for all prompt indices. On Slurm, use an array job where `SLURM_ARRAY_TASK_ID` is the prompt index. See [slurm_workflow.md](slurm_workflow.md#1-generate-gcg-suffixes).

**Output:** Raw GCG JSON files are saved under `data/gcg_results/raw/<model-or-alias>/index-*/seed-*/results.json`.

## 2. Create and Check Multi-Seed Datasets

**Paper connection:** Section 4, "Data" and "Generation of adversarial suffixes"; Section 5.1 intra-model transfer and multi-seed transfer matrices.

`pipeline.suffix_generation.create_datasets` replaces the old separate `create_dataset.py` and `check_dataset.py` flow. It reads raw GCG results, extracts prompt/suffix records, creates no-transfer and transfer datasets, and then prints validation results when `--check` is set. The validation pass reports missing prompt IDs and prompt IDs that do not have the expected number of seeds.

```bash
python -m pipeline.suffix_generation.create_datasets \
  --model-path "$MODEL_ID" \
  --results-dir data/gcg_results/raw \
  --output-dir "$OUTPUT_ROOT" \
  --expected-prompts "$EXPECTED_PROMPTS" \
  --expected-seeds "$EXPECTED_SEEDS" \
  --check
```

Use the printed check output before continuing. If prompts or seeds are missing, rerun suffix generation for the missing prompt/seed combinations.

**Output:** Initial no-transfer and transfer JSON arrays are saved under `$OUTPUT_ROOT/$MODEL_ALIAS/{no_transfer,transfer}/`. The check results are printed to stdout.

## 3. Canonical JSON Artifacts

**Paper connection:** Reproducibility support for Sections 4-5.

A manifest-backed artifact is a directory containing `manifest.json` plus canonical `chunks/`. The manifest records record counts, file sizes, and SHA256 checksums. `tools/split_json_records.py` creates the chunked release format, and `tools/verify_manifest.py` confirms that the committed chunks still match the manifest. See [artifacts.md](artifacts.md) for details.

After generating a large JSON array, split it into canonical chunks:

```bash
python tools/split_json_records.py \
  --input "$ARTIFACT_DIR/combined.json" \
  --output "$ARTIFACT_DIR" \
  --records-per-chunk "$JSON_RECORDS_PER_CHUNK"

python tools/verify_manifest.py \
  --manifest "$ARTIFACT_DIR/manifest.json"
```

**Output:** Canonical published chunks are saved to `$ARTIFACT_DIR/chunks/chunk_*.json`, and checksums/counts are saved to `$ARTIFACT_DIR/manifest.json`.

## 4. Prepare Generation Chunks

**Paper connection:** Operational setup for the Section 4 model-response generation step.

Generation jobs should not modify canonical `chunks/` directly. `tools/prepare_generation_chunks.py` reads the manifest-backed artifact, verifies that the requested input column exists, clears stale completion fields if requested, and writes temporary `generation_chunks/`. Use more chunks here when generation is cheap to parallelize across many single-GPU jobs.

```bash
python tools/prepare_generation_chunks.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --num-output-chunks "$NUM_GENERATION_CHUNKS" \
  --input-column jailbreak
```

**Output:** Temporary generation shards are saved to `$ARTIFACT_DIR/generation_chunks/chunk_*.json`.

## 5. Generate Completions

**Paper connection:** Section 4, "Evaluating jailbreak success"; Definition 1 ASR inputs.

`pipeline.generation.generate_completions` sends each `jailbreak` string to the target model and writes a `response` field into `generation_chunks/`. These model responses are not yet ASR labels; they are the raw text that the jailbreak judge evaluates in the next step.

`--chunk-id "$GENERATION_CHUNK_ID"` selects one file in `generation_chunks/`; chunk IDs map to zero-padded files such as `chunk_00000.json`. `--resume` skips rows that already have a non-empty `response`.

```bash
python -m pipeline.generation.generate_completions \
  --model-path "$MODEL_ID" \
  --multi-seed \
  --num-chunks "$NUM_GENERATION_CHUNKS" \
  --chunk-id "$GENERATION_CHUNK_ID" \
  --resume
```

After all generation jobs finish, check that every response was produced:

```bash
python tools/check_completions.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --subdir generation_chunks \
  --stage generation
```

The generation script also checks its own chunk before exiting. The full check above verifies all chunks together.

**Output:** `$ARTIFACT_DIR/generation_chunks/chunk_*.json` is updated with `response` fields. The coverage check prints missing or empty responses, if any.

## 6. Combine Completions for Evaluation

**Paper connection:** Operational setup for the Section 4 jailbreak-judge evaluation step.

Evaluation uses the Llama 3 jailbreak judge and typically needs more GPUs per job than generation, so use fewer chunks for this step. Set `NUM_JUDGE_GPUS` to the GPU count for your judge jobs. `tools/combine_completions.py` verifies that all `response` fields are populated before writing `evaluation_chunks/`. This is the replacement for the old combine step: it both checks completeness and changes the shard count for the judge workload.

```bash
python tools/combine_completions.py \
  --input-dir "$ARTIFACT_DIR" \
  --input-subdir generation_chunks \
  --output-subdir evaluation_chunks \
  --num-output-chunks "$NUM_EVALUATION_CHUNKS" \
  --check-stage generation
```

**Output:** Judge-ready temporary shards are saved to `$ARTIFACT_DIR/evaluation_chunks/chunk_*.json`. The command fails if any `response` field is missing.

## 7. Evaluate Completions

**Paper connection:** Section 4, "Evaluating jailbreak success"; Definition 1 ASR; all Section 5 transfer labels.

`pipeline.evaluation.evaluate_completions` runs the jailbreak judge on each `(prompt, response)` pair and writes a `jailbroken` boolean field into `evaluation_chunks/`. The `jailbroken` field is the success indicator used by ASR calculations, transfer matrices, and logistic-regression labels.

`--chunk_id "$EVALUATION_CHUNK_ID"` selects one file in `evaluation_chunks/`; chunk IDs map to zero-padded files such as `chunk_00000.json`.

```bash
python -m pipeline.evaluation.evaluate_completions \
  --model_path "$MODEL_ID" \
  --multi_seed \
  --chunk_id "$EVALUATION_CHUNK_ID" \
  --num_gpus "$NUM_JUDGE_GPUS"
```

After all evaluation jobs finish, check that every record was evaluated:

```bash
python tools/check_completions.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --subdir evaluation_chunks \
  --stage evaluation
```

The field is called `jailbroken`: `true` means the judge classified the response as a successful jailbreak, and `false` means it did not.

**Output:** `$ARTIFACT_DIR/evaluation_chunks/chunk_*.json` is updated with `jailbroken` boolean fields. The coverage check prints missing labels, if any.

## 8. Promote Evaluated Chunks Back to Canonical Artifacts

**Paper connection:** Reproducibility support for Section 5 analysis inputs.

After evaluation is complete, replace canonical `chunks/` with the evaluated records and update `manifest.json`. This is the step that makes the final evaluated artifact ready to commit and publish. `tools/combine_completions.py` checks that every `jailbroken` label is present before writing the canonical chunks.

```bash
python tools/combine_completions.py \
  --input-dir "$ARTIFACT_DIR" \
  --input-subdir evaluation_chunks \
  --output-subdir chunks \
  --num-output-chunks "$NUM_PUBLISHED_CHUNKS" \
  --check-stage evaluation \
  --write-manifest \
  --manifest-source multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer

python tools/verify_manifest.py \
  --manifest "$ARTIFACT_DIR/manifest.json"
```

Commit `chunks/` and `manifest.json`; do not commit `generation_chunks/`, `evaluation_chunks/`, or `combined.json`.

**Output:** Canonical evaluated chunks are saved to `$ARTIFACT_DIR/chunks/chunk_*.json`, and `$ARTIFACT_DIR/manifest.json` is updated.

## 9. Normalize Jailbreak Labels

**Paper connection:** Reproducibility support for Definition 1 ASR and Section 5 analyses.

Run `pipeline.evaluation.normalize_jailbreak_labels` when imported artifacts or older outputs encode labels as `0`/`1`, strings, or nullable values. Analysis scripts expect booleans so that `true` consistently means a successful jailbreak.

```bash
python -m pipeline.evaluation.normalize_jailbreak_labels \
  --input "$ARTIFACT_DIR/combined.json" \
  --output "$ARTIFACT_DIR/combined.json"
```

After normalizing, split the file back into canonical chunks if it is part of the release artifacts.

**Output:** The normalized JSON is written to the command's `--output` path, commonly `$ARTIFACT_DIR/combined.json`; split it back to `$ARTIFACT_DIR/chunks/` before committing.

## 10. No-Suffix Baseline

**Paper connection:** Section 4 experimental setup, Section 5.1 model susceptibility, and Section 5.6 prompt filtering for altered-loss experiments.

The no-suffix baseline asks the model to answer each harmful prompt without an adversarial suffix. This gives the base refusal or unsafe-response rate for the model, which helps distinguish suffix-driven jailbreaks from prompts that already succeed without an attack.

```bash
python -m pipeline.generation.generate_completions \
  --model-path "$MODEL_ID" \
  --no-suffix-completions

python -m pipeline.evaluation.evaluate_completions \
  --model_path "$MODEL_ID" \
  --no_suffix_completions \
  --num_gpus "$NUM_JUDGE_GPUS"
```

**Output:** No-suffix responses and `jailbroken` labels are saved under `data/no_suffix_generations/${MODEL_ALIAS}_no_suffix_generations/`, usually as `combined.json` plus any chunked release files you create from it.

## 11. Refusal Directions

**Paper connection:** Definition 2; Sections 3.2-3.3; Appendix A optimal-layer selection.

Stored refusal directions are used by activation-based analyses. The vectors are already committed under `data/refusal_directions/arditi_et_al_2024/`. `pipeline.refusal_directions.inspect_model` loads the stored direction, extracts a prompt activation at the selected layer, and prints the dot product and cosine similarity as a sanity check.

```bash
python -m pipeline.refusal_directions.inspect_model \
  --model-path "$MODEL_ID"
```

To regenerate directions from the external refusal-direction repo, see [refusal_directions.md](refusal_directions.md).

**Output:** Inspection prints the layer, dot product, and cosine similarity. Stored direction artifacts live at `data/refusal_directions/arditi_et_al_2024/$MODEL_ALIAS/direction.pt` and `direction_metadata.json`.

## 12. Activations

**Paper connection:** Definitions 2, 4, 5, and 6; Sections 5.2-5.5; Appendix C.

Activations are internal model vectors used for semantic-similarity, refusal-connectivity, suffix-push, and orthogonal-shift analyses. They are not stored in Git. `pipeline.activations.save_activations` regenerates hidden states, and `pipeline.activations.export_activations` converts the default tensor chunks into alternate layouts when an analysis needs a different indexing scheme.

```bash
python -m pipeline.activations.save_activations \
  --model-path "$MODEL_ID" \
  --input-kind multi_seed_jailbreak \
  --num-chunks "$NUM_ACTIVATION_CHUNKS" \
  --output-format canonical_tensor_chunks
```

See [activations.md](activations.md) for activation formats and export commands.

**Output:** Activation tensor chunks are saved under `outputs/activations/$MODEL_ALIAS/<input-kind>/canonical_tensor_chunks/`. These files are intentionally ignored by Git.

## 13. Multi-Seed Analysis

**Paper connection:** Section 5.1; Sections 5.2-5.5; Figures 1-2 and 4-6; Tables 1-3; Appendix C.

`pipeline.analysis.multi_seed_data_analysis` summarizes the multi-seed no-transfer and transfer datasets. It computes success matrices, aggregate attack success rates, most and least successful suffixes, and activation/refusal-direction plots when the required activation artifacts exist. It is the main intra-model analysis entry point for prompt vulnerability, suffix potency, semantic similarity, refusal connectivity, suffix push, orthogonal shift, and joint logistic-regression effects.

```bash
python -m pipeline.analysis.multi_seed_data_analysis \
  --model_path "$MODEL_ID"
```

**Output:** Summary statistics are printed to stdout, and figures are saved under `figures/$MODEL_ALIAS/`. Some activation-derived intermediates are read from or written under `outputs/activations/$MODEL_ALIAS/`.

## 14. Cross-Model Transfer

**Paper connection:** Section 3.1 inter-model transfer definition; Section 5.1 Figure 3; Section 5.5 inter-model results in Table 3; Appendix C.

Cross-model transfer asks whether suffixes generated on one source model jailbreak a different target model. The setup script builds the source-target records, generation runs the target model, evaluation labels each response with the judge, and the analysis script plots or summarizes the source-to-target success matrix.

```bash
export SOURCE_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
export TARGET_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
export SOURCE_ALIAS=llama-3.2-1b-instruct
export TARGET_ALIAS=qwen2.5-3b-instruct
export CROSS_MODEL_ARTIFACT_DIR="data/cross_model_transfer_generations/${SOURCE_ALIAS}_to_${TARGET_ALIAS}"

python -m pipeline.cross_model.set_up_dataset \
  --source_model_path "$SOURCE_MODEL_ID" \
  --target_model_path "$TARGET_MODEL_ID" \
  --num-chunks "$NUM_CROSS_MODEL_GENERATION_CHUNKS"
```

This writes `generation_chunks/` for the source-target pair. Generate responses, check them, re-shard into `evaluation_chunks/`, and evaluate:

```bash
python -m pipeline.cross_model.generate_completions \
  --source_model_path "$SOURCE_MODEL_ID" \
  --target_model_path "$TARGET_MODEL_ID" \
  --chunk_id "$CROSS_MODEL_GENERATION_CHUNK_ID"

python tools/check_completions.py \
  --artifact-dir "$CROSS_MODEL_ARTIFACT_DIR" \
  --subdir generation_chunks \
  --stage generation

python tools/combine_completions.py \
  --input-dir "$CROSS_MODEL_ARTIFACT_DIR" \
  --input-subdir generation_chunks \
  --output-subdir evaluation_chunks \
  --num-output-chunks "$NUM_CROSS_MODEL_EVALUATION_CHUNKS" \
  --check-stage generation

python -m pipeline.cross_model.evaluate_completions \
  --source_model_path "$SOURCE_MODEL_ID" \
  --target_model_path "$TARGET_MODEL_ID" \
  --chunk_id "$CROSS_MODEL_EVALUATION_CHUNK_ID" \
  --num_gpus "$NUM_JUDGE_GPUS"
```

Combine completed cross-model evaluation chunks and plot the success matrix:

```bash
python -m pipeline.cross_model.combine_dataset \
  --source-model-path "$SOURCE_MODEL_ID" \
  --target-model-path "$TARGET_MODEL_ID" \
  --num-chunks "$NUM_CROSS_MODEL_EVALUATION_CHUNKS"

python -m pipeline.cross_model.data_analysis \
  --source_model_path "$SOURCE_MODEL_ID" \
  --target_model_path "$TARGET_MODEL_ID"
```

**Output:** Cross-model records are saved under `$CROSS_MODEL_ARTIFACT_DIR/`, with temporary `generation_chunks/` and `evaluation_chunks/` during generation/evaluation. The combined evaluated dataset and figures are saved under the same cross-model artifact area and `figures/${SOURCE_ALIAS}_to_${TARGET_ALIAS}/`.

## 15. GCG Push

**Paper connection:** Section 5.6, "Altered GCG Loss"; Tables 4-5.

The local commands mirror [slurm_workflow.md](slurm_workflow.md), but full paper-scale GCG-push runs are not recommended locally. Use this section as the readable local equivalent of the recommended Slurm workflow.

Copy the paper config and edit local paths if needed:

```bash
cp configs/gcg_push_paper.example.yaml configs/gcg_push_paper.yaml
```

Run the raw altered-GCG jobs for the configured prompt indices and coefficients:

```bash
python -m pipeline.gcg_push.launch_experiment \
  --config configs/gcg_push_paper.yaml \
  --backend local \
  --stage raw
```

The command prints one local command per prompt index, coefficient, and intervention. Running all of them serially is slow; Slurm job arrays are the intended execution method. Raw outputs are saved under `outputs/gcg_push/raw/$MODEL_ALIAS/20_random_indices/{suffix_push,orth_shift}/coeff-$COEFF/index-*/seed-*/results.json`.

Build published chunked artifacts from raw outputs:

```bash
python -m pipeline.gcg_push.create_datasets \
  --config configs/gcg_push_paper.yaml \
  --check \
  --strict
```

This writes `data/gcg_push_results/$MODEL_ALIAS/{suffix_push,orth_shift}/coeff-$COEFF/{no_transfer,transfer}/chunks/` and `manifest.json`.

Generate completions, evaluate them, and promote evaluated records back to canonical chunks:

```bash
python -m pipeline.gcg_push.launch_experiment \
  --config configs/gcg_push_paper.yaml \
  --backend local \
  --stage generation

python -m pipeline.gcg_push.launch_experiment \
  --config configs/gcg_push_paper.yaml \
  --backend local \
  --stage evaluation

python -m pipeline.gcg_push.launch_experiment \
  --config configs/gcg_push_paper.yaml \
  --backend local \
  --stage publish
```

The generation stage creates temporary `generation_chunks/` and fills `response`; the evaluation stage creates `evaluation_chunks/` and fills `jailbroken`; the publish stage validates those fields and rewrites canonical `chunks/` plus `manifest.json`.

Analyze all configured coefficients and interventions:

```bash
python -m pipeline.gcg_push.data_analysis \
  --config configs/gcg_push_paper.yaml
```

**Output:** Published GCG-push artifacts live under `data/gcg_push_results/$MODEL_ALIAS/{suffix_push,orth_shift}/coeff-$COEFF/{no_transfer,transfer}/`. Analysis writes summaries to `outputs/gcg_push_analysis/` and prints ASR comparisons against the zero-coefficient baseline.

## 16. Prompt Rephrasings

**Paper connection:** Section 5.6, "Prompt rephrasing"; Appendix C prompt-rephrasing instructions.

Prompt rephrasing tests whether changing the wording of a harmful prompt changes its alignment with the refusal direction and, in turn, changes transfer success. `pipeline.prompt_rephrasings.setup_dataset` converts generated paraphrases into evaluation records. The generation and evaluation steps then use the same model-response and jailbreak-judge workflow as the main multi-seed artifacts.

```bash
python -m pipeline.prompt_rephrasings.setup_dataset \
  --model_path "$MODEL_ID" \
  --input-path data/prompt_rephrasings/${MODEL_ALIAS}_unprocessed_rephrasings.json

python -m pipeline.generation.generate_completions \
  --model-path "$MODEL_ID" \
  --rephrasings \
  --chunk-id "$GENERATION_CHUNK_ID" \
  --resume

python -m pipeline.evaluation.evaluate_completions \
  --model_path "$MODEL_ID" \
  --rephrasings \
  --chunk_id "$EVALUATION_CHUNK_ID" \
  --num_gpus "$NUM_JUDGE_GPUS"
```

**Output:** Prompt-rephrasing records are saved under `data/prompt_rephrasings/${MODEL_ALIAS}_prompt_rephrasings/`, with `generation_chunks/`, `evaluation_chunks/`, and final canonical `chunks/` following the same lifecycle as the main transfer artifacts.
