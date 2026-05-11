# Slurm Reproducibility Workflow

This is the recommended end-to-end guide for reproducing the paper results. It uses Slurm `sbatch` commands and Slurm job arrays to parallelize prompt and chunk jobs. It mirrors [workflow.md](workflow.md), which provides the local-command equivalent for reference but has not been tested end to end.

Slurm is an HPC workload manager/job scheduler. It is the recommended path for full paper-scale reproduction because the local commands process one prompt index or one chunk per invocation, while Slurm job arrays can run many prompt/chunk tasks in parallel.

## Table of Contents

- [Chunking Convention](#chunking-convention)
- [Slurm Job Arrays](#slurm-job-arrays)
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

## Slurm Job Arrays

Use Slurm job arrays for stages where each task handles one prompt index or one chunk. In the wrappers under `scripts/slurm/`, `SLURM_ARRAY_TASK_ID` becomes the prompt index or chunk ID. This is the recommended way to run generation and evaluation at paper scale.

Pass the actual scheduler partition to `sbatch` with `--partition "$SLURM_PARTITION"`. If your site uses accounts, constraints, or mail settings, pass them to `sbatch` directly, for example `--account`, `--constraint`, `--mail-type`, or `--mail-user`. Do not commit site-specific values to this repo.

## 0. Setup and Environment

Run commands from the repository root. Model IDs should match the models named in Paper Section 4 and Appendix B unless you are intentionally running an extension.

Create the conda environment once:

```bash
conda env create -f environment.yml
```

Define the shared model/cache and Slurm settings for each shell session:

```bash
export MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
export MODEL_ALIAS=llama-3.2-1b-instruct
export MODEL_CACHE_ROOT=/path/to/model/cache/root
export SLURM_PARTITION=general    # site-specific partition name

export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
```

Use `configs/models.example.yaml` for the model IDs used in the paper. Set `HF_TOKEN` as well if any gated model checkpoints require Hugging Face authentication.

Later steps define any extra variables immediately before use.

## 1. Generate GCG Suffixes

**Paper connection:** Section 4, "Generation of adversarial suffixes"; Section 5.1 Figures 1-2; Appendix B model/suffix counts.

`python -m pipeline suffixes run-gcg` runs the GCG attack for one JailbreakBench harmful prompt index and a seed range. Each seed produces a candidate adversarial suffix for the prompt. With Slurm, one array task runs one prompt index.

Set these variables for this step:

```bash
export NUM_PROMPTS=100
export GCG_ARRAY_START=0
export GCG_ARRAY_END=$((NUM_PROMPTS - 1))
export GCG_NUM_STEPS=500
export GCG_END_SEED=99
```

```bash
NUM_STEPS="$GCG_NUM_STEPS" END_SEED="$GCG_END_SEED" \
sbatch --partition "$SLURM_PARTITION" --array="${GCG_ARRAY_START}-${GCG_ARRAY_END}" \
  scripts/slurm/generate_suffixes.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" data/gcg_results/raw
```

**Output:** Raw GCG JSON files are saved under `data/gcg_results/raw/<model-or-alias>/index-*/seed-*/results.json`.

## 2. Create and Check Multi-Seed Datasets

**Paper connection:** Section 4, "Data" and "Generation of adversarial suffixes"; Section 5.1 intra-model transfer and multi-seed transfer matrices.

This CPU/data-prep step reads raw GCG results, extracts prompt/suffix records, creates no-transfer and transfer datasets, and prints validation results when `--check` is set. Run it locally, in an interactive Slurm session, or as a small batch job.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export EXPECTED_PROMPTS=100
export EXPECTED_SEEDS=100
```

```bash
python -m pipeline suffixes create-datasets \
  --model-path "$MODEL_ID" \
  --results-dir data/gcg_results/raw \
  --output-dir "$OUTPUT_ROOT" \
  --expected-prompts "$EXPECTED_PROMPTS" \
  --expected-seeds "$EXPECTED_SEEDS" \
  --check
```

Use the printed check output before continuing. If prompts or seeds are missing, rerun suffix generation for the missing prompt/seed combinations.

**Output:** Initial no-transfer and transfer JSON arrays are saved under `$OUTPUT_ROOT/$MODEL_ALIAS/{no_transfer,transfer}/`.

## 3. Canonical JSON Artifacts

**Paper connection:** Reproducibility support for Sections 4-5.

A manifest-backed artifact is a directory containing `manifest.json` plus canonical `chunks/`. The manifest records record counts, file sizes, and SHA256 checksums. `pipeline artifacts split-json` creates the chunked release format, and `pipeline artifacts verify-manifest` confirms that the committed chunks still match the manifest. See [artifacts.md](artifacts.md) for details.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export JSON_RECORDS_PER_CHUNK=5000
```

```bash
python -m pipeline artifacts split-json \
  --input "$ARTIFACT_DIR/combined.json" \
  --output "$ARTIFACT_DIR" \
  --records-per-chunk "$JSON_RECORDS_PER_CHUNK"

python -m pipeline artifacts verify-manifest \
  --manifest "$ARTIFACT_DIR/manifest.json"
```

**Output:** Canonical published chunks are saved to `$ARTIFACT_DIR/chunks/chunk_*.json`, and checksums/counts are saved to `$ARTIFACT_DIR/manifest.json`.

## 4. Prepare Generation Chunks

**Paper connection:** Operational setup for the Section 4 model-response generation step.

Generation jobs should not modify canonical `chunks/` directly. This step creates temporary `generation_chunks/`; it is usually small enough to run as a local command or in an interactive Slurm session before submitting the array.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NUM_GENERATION_CHUNKS=32
```

```bash
python -m pipeline artifacts prepare-generation \
  --artifact-dir "$ARTIFACT_DIR" \
  --num-output-chunks "$NUM_GENERATION_CHUNKS" \
  --input-column jailbreak
```

**Output:** Temporary generation shards are saved to `$ARTIFACT_DIR/generation_chunks/chunk_*.json`.

## 5. Generate Completions

**Paper connection:** Section 4, "Evaluating jailbreak success"; Definition 1 ASR inputs.

`python -m pipeline completions generate` sends each `jailbreak` string to the target model and writes a `response` field into `generation_chunks/`. These model responses are not yet ASR labels; they are the raw text that the jailbreak judge evaluates in the next step.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NUM_GENERATION_CHUNKS=32
export GENERATION_ARRAY_START=0
export GENERATION_ARRAY_END=$((NUM_GENERATION_CHUNKS - 1))
```

Submit one array task per `generation_chunks/chunk_*.json` file:

```bash
sbatch --partition "$SLURM_PARTITION" --array="${GENERATION_ARRAY_START}-${GENERATION_ARRAY_END}" \
  scripts/slurm/generate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" multi-seed
```

After the array finishes, check that every response was produced:

```bash
python -m pipeline artifacts check-completions \
  --artifact-dir "$ARTIFACT_DIR" \
  --subdir generation_chunks \
  --stage generation
```

**Output:** `$ARTIFACT_DIR/generation_chunks/chunk_*.json` is updated with `response` fields.

## 6. Combine Completions for Evaluation

**Paper connection:** Operational setup for the Section 4 jailbreak-judge evaluation step.

Evaluation uses the Llama 3 jailbreak judge and typically needs more GPUs per job than generation, so use fewer chunks for this step. `pipeline artifacts combine-completions` verifies that all `response` fields are populated before writing `evaluation_chunks/`.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NUM_EVALUATION_CHUNKS=8
```

```bash
python -m pipeline artifacts combine-completions \
  --input-dir "$ARTIFACT_DIR" \
  --input-subdir generation_chunks \
  --output-subdir evaluation_chunks \
  --num-output-chunks "$NUM_EVALUATION_CHUNKS" \
  --check-stage generation
```

**Output:** Judge-ready temporary shards are saved to `$ARTIFACT_DIR/evaluation_chunks/chunk_*.json`. The command fails if any `response` field is missing.

## 7. Evaluate Completions

**Paper connection:** Section 4, "Evaluating jailbreak success"; Definition 1 ASR; all Section 5 transfer labels.

`python -m pipeline completions evaluate` runs the jailbreak judge on each `(prompt, response)` pair and writes a `jailbroken` boolean field into `evaluation_chunks/`.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NUM_EVALUATION_CHUNKS=8
export EVALUATION_ARRAY_START=0
export EVALUATION_ARRAY_END=$((NUM_EVALUATION_CHUNKS - 1))
export NUM_JUDGE_GPUS=4
```

Submit one array task per evaluation chunk:

```bash
NUM_GPUS="$NUM_JUDGE_GPUS" \
sbatch --partition "$SLURM_PARTITION" --array="${EVALUATION_ARRAY_START}-${EVALUATION_ARRAY_END}" \
  scripts/slurm/evaluate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" multi-seed
```

After the array finishes, check that every record was evaluated:

```bash
python -m pipeline artifacts check-completions \
  --artifact-dir "$ARTIFACT_DIR" \
  --subdir evaluation_chunks \
  --stage evaluation
```

The field is called `jailbroken`: `true` means the judge classified the response as a successful jailbreak, and `false` means it did not.

**Output:** `$ARTIFACT_DIR/evaluation_chunks/chunk_*.json` is updated with `jailbroken` boolean fields.

## 8. Promote Evaluated Chunks Back to Canonical Artifacts

**Paper connection:** Reproducibility support for Section 5 analysis inputs.

After evaluation is complete, replace canonical `chunks/` with the evaluated records and update `manifest.json`. This is the step that makes the final evaluated artifact ready to commit and publish.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
export NUM_PUBLISHED_CHUNKS=50
```

```bash
python -m pipeline artifacts combine-completions \
  --input-dir "$ARTIFACT_DIR" \
  --input-subdir evaluation_chunks \
  --output-subdir chunks \
  --num-output-chunks "$NUM_PUBLISHED_CHUNKS" \
  --check-stage evaluation \
  --write-manifest \
  --manifest-source multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer

python -m pipeline artifacts verify-manifest \
  --manifest "$ARTIFACT_DIR/manifest.json"
```

Commit `chunks/` and `manifest.json`; do not commit `generation_chunks/`, `evaluation_chunks/`, or `combined.json`.

**Output:** Canonical evaluated chunks are saved to `$ARTIFACT_DIR/chunks/chunk_*.json`, and `$ARTIFACT_DIR/manifest.json` is updated.

## 9. Normalize Jailbreak Labels

**Paper connection:** Reproducibility support for Definition 1 ASR and Section 5 analyses.

Run this when imported artifacts or older outputs encode labels as `0`/`1`, strings, or nullable values. This CPU/data-prep step can run locally or as a small batch job.

Set these variables for this step:

```bash
export OUTPUT_ROOT=data/multiple_seed_results
export ARTIFACT_DIR="$OUTPUT_ROOT/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer"
```

```bash
python -m pipeline completions normalize-labels \
  --path "$ARTIFACT_DIR/combined.json" \
  --output "$ARTIFACT_DIR/combined.json"
```

After normalizing, split the file back into canonical chunks if it is part of the release artifacts.

**Output:** The normalized JSON is written to the command's `--output` path, commonly `$ARTIFACT_DIR/combined.json`; split it back to `$ARTIFACT_DIR/chunks/` before committing.

## 10. No-Suffix Baseline

**Paper connection:** Section 4 experimental setup, Section 5.1 model susceptibility, and Section 5.6 prompt filtering for altered-loss experiments.

The no-suffix baseline asks the model to answer each harmful prompt without an adversarial suffix. This gives the base refusal or unsafe-response rate for the model.

Set these variables for this step:

```bash
export NO_SUFFIX_ARTIFACT_DIR="data/no_suffix_generations/${MODEL_ALIAS}_no_suffix_generations"
export NUM_JUDGE_GPUS=4
```

```bash
sbatch --partition "$SLURM_PARTITION" \
  scripts/slurm/generate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" no-suffix-completions

NUM_GPUS="$NUM_JUDGE_GPUS" \
sbatch --partition "$SLURM_PARTITION" \
  scripts/slurm/evaluate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" no-suffix-completions
```

**Output:** No-suffix responses and `jailbroken` labels are saved under `data/no_suffix_generations/${MODEL_ALIAS}_no_suffix_generations/`, usually as `combined.json` plus any chunked release files you create from it.

## 11. Refusal Directions

**Paper connection:** Definition 2; Sections 3.2-3.3; Appendix A optimal-layer selection.

Stored refusal directions are used by activation-based analyses. The vectors are already committed under `data/refusal_directions/arditi_et_al_2024/`. `python -m pipeline refusal inspect` loads the stored direction, extracts a prompt activation at the selected layer, and prints the dot product and cosine similarity as a sanity check.

```bash
sbatch --partition "$SLURM_PARTITION" --gres=gpu:1 \
  --wrap "python -m pipeline refusal inspect --model-path \"$MODEL_ID\""
```

To regenerate directions from the external refusal-direction repo, see [refusal_directions.md](refusal_directions.md).

**Output:** Inspection prints the layer, dot product, and cosine similarity. Stored direction artifacts live at `data/refusal_directions/arditi_et_al_2024/$MODEL_ALIAS/direction.pt` and `direction_metadata.json`.

## 12. Activations

**Paper connection:** Definitions 2, 4, 5, and 6; Sections 5.2-5.5; Appendix C.

Activations are internal model vectors used for semantic-similarity, refusal-connectivity, suffix-push, and orthogonal-shift analyses. They are not stored in Git. The Slurm activation wrapper runs `python -m pipeline activations save` with the chosen input kind, number of chunks, and batch size.

Set these variables for this step:

```bash
export NUM_ACTIVATION_CHUNKS=32
export ACTIVATION_BATCH_SIZE=100
```

```bash
NUM_CHUNKS="$NUM_ACTIVATION_CHUNKS" BATCH_SIZE="$ACTIVATION_BATCH_SIZE" \
sbatch --partition "$SLURM_PARTITION" \
  scripts/slurm/save_activations.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" multi_seed_jailbreak
```

See [activations.md](activations.md) for activation formats and export commands.

**Output:** Activation tensor chunks are saved under `outputs/activations/$MODEL_ALIAS/<input-kind>/canonical_tensor_chunks/`. These files are intentionally ignored by Git.

## 13. Multi-Seed Analysis

**Paper connection:** Section 5.1; Sections 5.2-5.5; Figures 1-2 and 4-6; Tables 1-3; Appendix C.

`python -m pipeline analysis multi-seed` summarizes the multi-seed no-transfer and transfer datasets. It computes success matrices, aggregate attack success rates, most and least successful suffixes, and activation/refusal-direction plots when the required activation artifacts exist.

```bash
sbatch --partition "$SLURM_PARTITION" --gres=gpu:1 \
  --wrap "python -m pipeline analysis multi-seed --model-path \"$MODEL_ID\""
```

**Output:** Summary statistics are printed to stdout, and figures are saved under `figures/$MODEL_ALIAS/`.

## 14. Cross-Model Transfer

**Paper connection:** Section 3.1 inter-model transfer definition; Section 5.1 Figure 3; Section 5.5 inter-model results in Table 3; Appendix C.

Cross-model transfer asks whether suffixes generated on one source model jailbreak a different target model. The setup script builds the source-target records, generation runs the target model, evaluation labels each response with the judge, and the analysis script plots or summarizes the source-to-target success matrix.

Set these variables for this step:

```bash
export SOURCE_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
export TARGET_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
export SOURCE_ALIAS=llama-3.2-1b-instruct
export TARGET_ALIAS=qwen2.5-3b-instruct
export CROSS_MODEL_ARTIFACT_DIR="data/cross_model_transfer_generations/${SOURCE_ALIAS}_to_${TARGET_ALIAS}"
export NUM_CROSS_MODEL_GENERATION_CHUNKS=32
export CROSS_MODEL_GENERATION_ARRAY_START=0
export CROSS_MODEL_GENERATION_ARRAY_END=$((NUM_CROSS_MODEL_GENERATION_CHUNKS - 1))
export NUM_CROSS_MODEL_EVALUATION_CHUNKS=8
export CROSS_MODEL_EVALUATION_ARRAY_START=0
export CROSS_MODEL_EVALUATION_ARRAY_END=$((NUM_CROSS_MODEL_EVALUATION_CHUNKS - 1))
export NUM_JUDGE_GPUS=4
```

```bash
python -m pipeline cross-model setup \
  --source-model-path "$SOURCE_MODEL_ID" \
  --target-model-path "$TARGET_MODEL_ID" \
  --num-chunks "$NUM_CROSS_MODEL_GENERATION_CHUNKS"
```

Submit one array task per cross-model generation chunk:

```bash
sbatch --partition "$SLURM_PARTITION" --array="${CROSS_MODEL_GENERATION_ARRAY_START}-${CROSS_MODEL_GENERATION_ARRAY_END}" \
  scripts/slurm/cross_model_generate.sbatch \
  "$SOURCE_MODEL_ID" "$TARGET_MODEL_ID" "$MODEL_CACHE_ROOT"

python -m pipeline artifacts check-completions \
  --artifact-dir "$CROSS_MODEL_ARTIFACT_DIR" \
  --subdir generation_chunks \
  --stage generation

python -m pipeline artifacts combine-completions \
  --input-dir "$CROSS_MODEL_ARTIFACT_DIR" \
  --input-subdir generation_chunks \
  --output-subdir evaluation_chunks \
  --num-output-chunks "$NUM_CROSS_MODEL_EVALUATION_CHUNKS" \
  --check-stage generation

NUM_GPUS="$NUM_JUDGE_GPUS" \
sbatch --partition "$SLURM_PARTITION" --array="${CROSS_MODEL_EVALUATION_ARRAY_START}-${CROSS_MODEL_EVALUATION_ARRAY_END}" \
  scripts/slurm/cross_model_evaluate.sbatch \
  "$SOURCE_MODEL_ID" "$TARGET_MODEL_ID" "$MODEL_CACHE_ROOT"
```

Combine completed cross-model evaluation chunks and plot the success matrix:

```bash
python -m pipeline cross-model combine \
  --source-model-path "$SOURCE_MODEL_ID" \
  --target-model-path "$TARGET_MODEL_ID" \
  --num-chunks "$NUM_CROSS_MODEL_EVALUATION_CHUNKS"

sbatch --partition "$SLURM_PARTITION" --gres=gpu:1 \
  --wrap "python -m pipeline cross-model analyze --source-model-path \"$SOURCE_MODEL_ID\" --target-model-path \"$TARGET_MODEL_ID\""
```

**Output:** Cross-model records are saved under `$CROSS_MODEL_ARTIFACT_DIR/`, with temporary `generation_chunks/` and `evaluation_chunks/` during generation/evaluation. The combined evaluated dataset and figures are saved under the same cross-model artifact area and `figures/${SOURCE_ALIAS}_to_${TARGET_ALIAS}/`.

## 15. GCG Push

**Paper connection:** Section 5.6, "Altered GCG Loss"; Tables 4-5.

GCG-push experiments rerun GCG with an added loss term. `suffix_push` encourages suffixes to move away from the refusal direction, while `orth_shift` encourages movement in the component orthogonal to the refusal direction. The paper configuration is stored in `configs/gcg_push_paper.example.yaml`; copy it before editing cluster-specific paths.

```bash
cp configs/gcg_push_paper.example.yaml configs/gcg_push_paper.yaml
```

Edit `configs/gcg_push_paper.yaml` to set `slurm.partition`, `slurm.model_cache_root`, and, if needed, `slurm.max_parallel_tasks`. With 32 available GPUs, set `slurm.max_parallel_tasks: 32`; this caps each Slurm array at 32 simultaneous one-GPU tasks. If you submit many coefficient arrays at once, the scheduler still decides how many total jobs run, so submit one intervention or coefficient at a time when you need strict control.

Preview the raw altered-GCG jobs without submitting:

```bash
python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage raw
```

Submit raw altered-GCG jobs after reviewing the printed commands:

```bash
python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage raw \
  --submit
```

The raw stage writes one `results.json` per configured prompt index and seed under `outputs/gcg_push/raw/$MODEL_ALIAS/20_random_indices/{suffix_push,orth_shift}/coeff-$COEFF/index-*/seed-*/`. Those raw outputs are ignored by Git and are the reproducible source for the published chunked artifacts.

After the raw Slurm arrays finish, build the chunked no-transfer and transfer artifacts:

```bash
python -m pipeline gcg-push create-datasets \
  --config configs/gcg_push_paper.yaml \
  --check \
  --strict
```

This writes canonical artifacts to `data/gcg_push_results/$MODEL_ALIAS/{suffix_push,orth_shift}/coeff-$COEFF/{no_transfer,transfer}/chunks/` plus `manifest.json`. The published artifacts in this repo use this same layout.

Generate model responses for the transfer artifacts:

```bash
python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage generation

python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage generation \
  --submit
```

The generation stage first creates temporary `generation_chunks/`, then submits array jobs that fill the `response` field. After generation jobs finish, run evaluation:

```bash
python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage evaluation

python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage evaluation \
  --submit
```

The evaluation stage reshards `generation_chunks/` into fewer `evaluation_chunks/` for the multi-GPU jailbreak judge, then submits judge array jobs. After the evaluation arrays finish, promote evaluated records back to canonical published chunks:

```bash
python -m pipeline gcg-push launch \
  --config configs/gcg_push_paper.yaml \
  --backend slurm \
  --stage publish \
  --submit
```

Finally, compute the full GCG-push summary table across both interventions and all configured coefficients:

```bash
python -m pipeline gcg-push analyze \
  --config configs/gcg_push_paper.yaml
```

**Output:** Published GCG-push artifacts live under `data/gcg_push_results/$MODEL_ALIAS/{suffix_push,orth_shift}/coeff-$COEFF/{no_transfer,transfer}/`. Analysis writes summaries to `outputs/gcg_push_analysis/` and prints ASR comparisons against the zero-coefficient baseline.

## 16. Prompt Rephrasings

**Paper connection:** Section 5.6, "Prompt rephrasing"; Appendix C prompt-rephrasing instructions.

Prompt rephrasing tests whether changing the wording of a harmful prompt changes its alignment with the refusal direction and, in turn, changes transfer success. `python -m pipeline prompt-rephrasings setup` converts generated paraphrases into evaluation records. The generation and evaluation steps then use the same model-response and jailbreak-judge workflow as the main multi-seed artifacts.

Set these variables for this step:

```bash
export NUM_GENERATION_CHUNKS=32
export GENERATION_ARRAY_START=0
export GENERATION_ARRAY_END=$((NUM_GENERATION_CHUNKS - 1))
export NUM_EVALUATION_CHUNKS=8
export EVALUATION_ARRAY_START=0
export EVALUATION_ARRAY_END=$((NUM_EVALUATION_CHUNKS - 1))
export NUM_JUDGE_GPUS=4
```

```bash
python -m pipeline prompt-rephrasings setup \
  --model-path "$MODEL_ID" \
  --input-path data/prompt_rephrasings/${MODEL_ALIAS}_unprocessed_rephrasings.json

sbatch --partition "$SLURM_PARTITION" --array="${GENERATION_ARRAY_START}-${GENERATION_ARRAY_END}" \
  scripts/slurm/generate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" rephrasings

NUM_GPUS="$NUM_JUDGE_GPUS" \
sbatch --partition "$SLURM_PARTITION" --array="${EVALUATION_ARRAY_START}-${EVALUATION_ARRAY_END}" \
  scripts/slurm/evaluate_completions.sbatch \
  "$MODEL_ID" "$MODEL_CACHE_ROOT" rephrasings
```

**Output:** Prompt-rephrasing records are saved under `data/prompt_rephrasings/${MODEL_ALIAS}_prompt_rephrasings/`, with `generation_chunks/`, `evaluation_chunks/`, and final canonical `chunks/` following the same lifecycle as the main transfer artifacts.
