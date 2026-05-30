# Paper Results

This page is the paper-facing reproduction checklist. It complements the workflow docs, which are organized by operational stage. The legacy TMLR analysis snapshot was treated as read-only; its reusable analysis logic is exposed through `python -m pipeline paper ...` commands and the R scripts in `scripts/r/`.

## Result Registry

Print the machine-readable registry:

```bash
python -m pipeline paper registry --json
```

Or print the human-readable command map:

```bash
python -m pipeline paper registry
```

The registry maps paper figures/tables to the repo commands that build the needed artifacts.

## Feature Datasets

Build semantic-similarity datasets for Tables 1 and 10:

```bash
python -m pipeline paper semantic-dataset \
  --model-path "$MODEL_ID" \
  --embedding model \
  --dimensionality 100d
```

For independent sentence-transformer embeddings:

```bash
python -m pipeline paper prompt-embeddings \
  --model-path "$MODEL_ID" \
  --sentence-model all-mpnet-base-v2

python -m pipeline paper semantic-dataset \
  --model-path "$MODEL_ID" \
  --embedding independent \
  --embeddings-path outputs/paper/prompt_embeddings/all-mpnet-base-v2_prompt_embeddings.pt \
  --dimensionality 100d
```

Build refusal-connectivity, suffix-push, orthogonal-shift, and model-internal semantic-similarity rows for Tables 2, 3, 8, 9, 11, 12, and 13:

```bash
python -m pipeline paper feature-dataset \
  --model-path "$MODEL_ID" \
  --dimensionality 100d
```

For inter-model transfer columns, regenerate target-model activations on the source-to-target transfer artifact and then build the corresponding 1D feature dataset:

```bash
python -m pipeline activations save \
  --model-path "$TARGET_MODEL_ID" \
  --source-model-path "$SOURCE_MODEL_ID" \
  --input-kind cross_model_jailbreak

python -m pipeline paper feature-dataset \
  --model-path "$TARGET_MODEL_ID" \
  --source-model-path "$SOURCE_MODEL_ID" \
  --dimensionality 1d
```

These commands read committed judged JSON artifacts and regenerated activation chunks. They write CSV files under `outputs/paper/`.

## Mixed-Effects Tables

Fit the R mixed-effects models after building the CSV datasets:

```bash
Rscript scripts/r/semantics_model.R \
  --data-dir outputs/paper/semantic_datasets \
  --results-dir outputs/paper/r/semantics

Rscript scripts/r/full_random_effects_model.R \
  --data-dir outputs/paper/feature_datasets \
  --results-dir outputs/paper/r/features
```

The scripts emit CSV coefficient tables with fixed-effect estimates, sample sizes, and marginal/conditional R2 values.

The R scripts require `lme4`, `performance`, `broom.mixed`, `dplyr`, and `readr` in the active R environment.

## Interventions

Prompt rephrasing analysis:

```bash
python -m pipeline paper prompt-rephrasing-analysis \
  --model-path "$MODEL_ID"
```

Human validation of the jailbreak judge:

```bash
python -m pipeline paper human-eval summarize \
  --input data/human_eval/reviewed_sampled_items
```

Altered GCG loss summaries remain under the existing GCG-push command:

```bash
python -m pipeline gcg-push analyze \
  --config configs/gcg_push_paper.example.yaml
```

## Camera-Ready Note

The uploaded camera-ready PDF contains revision-era duplicate tables, especially main-text tables and Appendix E tables with different sample sizes or coefficients. The repo keeps the result registry location-specific rather than collapsing them into a single canonical table.
