# Artifacts

This repository stores JSON artifacts in a manifest-backed chunk format. A manifest-backed artifact is a directory where `manifest.json` is the index and checksum record for the actual chunk files. The manifest tells a reproducer how many records exist, which files belong to the artifact, and whether each chunk still matches its SHA256 checksum. These artifacts are the data products behind the paper's Section 4 experimental setup and Section 5 analyses.

## Table of Contents

- [Chunk Layout](#chunk-layout)
- [Canonical vs Working Chunks](#canonical-vs-working-chunks)
- [Raw GCG Results](#raw-gcg-results)
- [Binary Files](#binary-files)

## Chunk Layout

```text
data/<artifact-family>/<artifact-name>/
  manifest.json
  chunks/
    chunk_00000.json
    chunk_00001.json
```

A typical manifest looks like this:

```json
{
  "artifact_format": "chunked_json",
  "source": "multiple_seed_results/example.json",
  "total_records": 250000,
  "total_chunks": 50,
  "chunks": [
    {
      "path": "chunks/chunk_00000.json",
      "records": 5000,
      "bytes": 1234567,
      "sha256": "..."
    }
  ]
}
```

The canonical chunk files are the source of truth in Git. Some analysis scripts need a full temporary working JSON file named `combined.json`. Create it from a manifest like this:

```bash
python tools/combine_json_chunks.py \
  --manifest data/multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer/manifest.json \
  --output data/multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer/combined.json
```

After creating or modifying a working JSON file, split it back into chunked form before release. The example uses `JSON_RECORDS_PER_CHUNK` so you can change chunk size in one place:

```bash
export JSON_RECORDS_PER_CHUNK=5000

python tools/split_json_records.py \
  --input data/multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer/combined.json \
  --output data/multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer \
  --records-per-chunk "$JSON_RECORDS_PER_CHUNK"
```

Verify any artifact before publishing it:

```bash
python tools/verify_manifest.py \
  --manifest data/multiple_seed_results/$MODEL_ALIAS/transfer/${MODEL_ALIAS}_multiple_seed_results_transfer/manifest.json
```


## Canonical vs Working Chunks

`chunks/` is the canonical published artifact directory. It is the only JSON chunk directory that should be committed with `manifest.json`.

Completion and evaluation jobs use two ignored working directories:

```text
<artifact>/
  manifest.json              # committed
  chunks/                    # committed canonical artifact chunks
  generation_chunks/         # temporary, many chunks for single-GPU generation jobs
  evaluation_chunks/         # temporary, fewer chunks for multi-GPU judge jobs
  combined.json              # temporary full materialization when needed
```

The intended lifecycle is:

```text
chunks/ -> generation_chunks/ -> evaluation_chunks/ -> chunks/ + manifest.json
```

Use `tools/prepare_generation_chunks.py` to create `generation_chunks/` from a manifest. Use `tools/combine_completions.py` to re-shard `generation_chunks/` into `evaluation_chunks/`, and later to promote evaluated records back into canonical `chunks/`. `tools/check_completions.py` checks that each generation record has a `response` and that each evaluated record has a `jailbroken` label before the artifact is promoted.

## Raw GCG Results

Raw per-seed GCG result files are copied under:

```text
data/gcg_results/raw/<model>/index-NNNN/seed-NNNN/results.json
```

These files are copied as-is for the v1 release because the current GCG runner writes one file per prompt and seed. Future work is to make GCG write chunked artifacts directly and then migrate the existing raw result tree into that format.

## Binary Files

Refusal-direction `.pt` files are stored because they are below GitHub's normal file limit and are needed for the paper's Definition 2 and activation-derived features. Activation tensors are excluded from Git and regenerated with the commands in [activations.md](activations.md).
