# Future Work

- Refactor GCG suffix generation so `python -m pipeline suffixes run-gcg` writes chunked artifacts directly instead of one `results.json` per prompt and seed. This would simplify the Section 4 suffix-generation provenance trail.
- Convert `data/gcg_results/raw/` into manifest-backed chunked artifacts after the GCG output format is stable.
- Add richer schema validation for every artifact family, including required columns for generation, evaluation, and analysis.
- Add optional external artifact hosting for activation tensors if the paper release needs precomputed activations.
- Finish migrating analysis loaders to read manifest-backed `chunks/` directly instead of requiring temporary `combined.json` files, including for multi-seed `no_transfer` artifacts.
- Decide whether future releases should keep GCG-push `no_transfer` artifacts in Git or regenerate them only as intermediates before publishing `transfer` artifacts.
