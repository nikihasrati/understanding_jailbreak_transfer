# Future Work

- Refactor GCG suffix generation so `pipeline.suffix_generation.run_gcg` writes chunked artifacts directly instead of one `results.json` per prompt and seed. This would simplify the Section 4 suffix-generation provenance trail.
- Convert `data/gcg_results/raw/` into manifest-backed chunked artifacts after the GCG output format is stable.
- Add richer schema validation for every artifact family, including required columns for generation, evaluation, and analysis.
- Add optional external artifact hosting for activation tensors if the paper release needs precomputed activations.
- Normalize CLI option names so every command consistently uses hyphenated long flags while preserving aliases for older scripts.
- Update multi-seed and cross-model loaders to read manifest-backed `chunks/` directly instead of requiring temporary `combined.json` files, including for `no_transfer` artifacts.
- Decide whether future releases should keep GCG-push `no_transfer` artifacts in Git or regenerate them only as intermediates before publishing `transfer` artifacts.
