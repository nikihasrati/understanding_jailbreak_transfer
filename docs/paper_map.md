# Paper Map

This file is the paper-centric crosswalk for the repo. The README workflow table is intentionally operational; this document instead starts from the paper's definitions, sections, figures, and tables, then points to the code and artifacts that support each one.

## Table of Contents

- [Section and Result Crosswalk](#section-and-result-crosswalk)
- [Reproducibility Helpers](#reproducibility-helpers)
- [Paper Concepts Used by the Workflow](#paper-concepts-used-by-the-workflow)

## Section and Result Crosswalk

| Paper section | Relevant files | Description |
| --- | --- | --- |
| Section 3.1, transfer definitions | `pipeline.analysis.multi_seed_data_analysis`, `pipeline.cross_model.*` | Intra-model transfer is analyzed within one model's multi-seed artifact; inter-model transfer is built and analyzed with the cross-model pipeline. |
| Definition 1, ASR | `pipeline.evaluation.evaluate_completions`, `tools/check_completions.py`, `pipeline.evaluation.normalize_jailbreak_labels` | The jailbreak judge writes the `jailbroken` boolean label. ASR is computed from that label, and helper tools verify or normalize the labels before analysis. |
| Definition 2 and Appendix A, refusal direction | `data/refusal_directions/arditi_et_al_2024/*`, `pipeline.refusal_directions.inspect_model`, `docs/refusal_directions.md` | Stored refusal-direction tensors and metadata provide the vectors used by activation-based features. The inspect command sanity-checks the selected direction against a model activation. |
| Definitions 3-6, semantic similarity, refusal connectivity, suffix push, orthogonal shift | `pipeline.activations.save_activations`, `pipeline.activations.export_activations`, `pipeline.analysis.multi_seed_data_analysis`, `docs/activations.md` | Activation regeneration provides the tensors needed to compute prompt similarity and refusal-direction geometry. The analysis scripts consume those tensors for feature plots and regressions. |
| Section 4, models and data | `configs/models.example.yaml`, `data/processed/`, `pipeline/model_utils/*` | Model identifiers, JailbreakBench-derived prompt artifacts, and model wrappers define the experimental subjects and input data. |
| Section 4, adversarial suffix generation | `pipeline.suffix_generation.run_gcg`, `scripts/slurm/generate_suffixes.sbatch`, `data/gcg_results/raw/` | Runs GCG for prompt/seed combinations and stores the raw per-seed suffix outputs used to build transfer datasets. |
| Section 4, dataset construction | `pipeline.suffix_generation.create_datasets --check` | Converts raw GCG files into no-transfer and transfer datasets and checks expected prompt/seed coverage before generation. |
| Section 4, jailbreak-success evaluation | `pipeline.generation.generate_completions`, `pipeline.evaluation.evaluate_completions`, `tools/check_completions.py` | Generates target-model responses, judges those responses with the jailbreak judge, and checks that all response and judge fields are populated. |
| Section 5.1 and Figures 1-2, intra-model and multi-seed transfer | `pipeline.analysis.multi_seed_data_analysis`, `data/multiple_seed_results/` | Produces intra-model transfer matrices, ASR summaries, and multi-seed comparisons from the judged datasets. |
| Section 5.1 Figure 3 and Section 5.5 cross-model columns | `pipeline.cross_model.set_up_dataset`, `pipeline.cross_model.generate_completions`, `pipeline.cross_model.evaluate_completions`, `pipeline.cross_model.combine_dataset`, `pipeline.cross_model.data_analysis` | Builds source-to-target transfer records, generates target responses, judges them, combines evaluated chunks, and analyzes cross-model transfer. |
| Sections 5.2-5.5 and Tables 1-3 | `pipeline.analysis.multi_seed_data_analysis`, `pipeline.activations.save_activations` | Computes semantic-similarity, refusal-connectivity, suffix-push, orthogonal-shift, and joint-regression analyses from judged outputs and regenerated activations. |
| Section 5.6, altered GCG loss, Tables 4-5 | `pipeline.generation.generate_completions --gcg-push`, `pipeline.evaluation.evaluate_completions --gcg_push`, `pipeline.gcg_push.data_analysis` | Evaluates suffixes from coefficient-modified GCG runs and compares intervention ASR against the zero-coefficient baseline. |
| Section 5.6 and Appendix C, prompt rephrasing | `pipeline.prompt_rephrasings.setup_dataset`, `pipeline.generation.generate_completions --rephrasings`, `pipeline.evaluation.evaluate_completions --rephrasings` | Converts prompt rephrases into evaluation records, generates and judges responses, and supports the rephrasing intervention analysis. |
| Appendix B, model details | `configs/models.example.yaml`, `pipeline/model_utils/*` | Records the model IDs and wrapper support needed to reproduce the model set. |
| Appendix C, additional quantitative results | `pipeline.analysis.multi_seed_data_analysis`, `pipeline.cross_model.data_analysis`, `pipeline.gcg_push.data_analysis` | Produces extended feature plots, interaction analyses, and supporting intervention results. |

## Reproducibility Helpers

These utilities are not paper claims by themselves, but they make the release reproducible and keep large artifacts Git-friendly.

| Helper | Purpose |
| --- | --- |
| `tools/split_json_records.py` | Splits large JSON arrays into canonical `chunks/` plus `manifest.json`. |
| `tools/combine_json_chunks.py` | Rebuilds temporary full JSON files such as `combined.json` from committed chunks. |
| `tools/verify_manifest.py` and `tools/make_manifest.py` | Verifies or creates checksum manifests for committed artifacts. |
| `tools/prepare_generation_chunks.py` | Creates temporary `generation_chunks/` from canonical chunks before model-response generation. |
| `tools/combine_completions.py` | Re-shards generated records for evaluation jobs and promotes evaluated records back to canonical chunks. |
| `scripts/slurm/*` and `scripts/local/*` | Provide Slurm and local wrappers for the same Python entry points. The recommended workflow is documented in `docs/slurm_workflow.md`; `docs/workflow.md` is the local-command equivalent. |

## Paper Concepts Used by the Workflow

- **Attack success rate (ASR)** is Definition 1. In the repo, ASR is computed from the `jailbroken` labels produced by the judge.
- **Intra-model transfer** is defined in Section 3.1 and measured by applying suffixes generated for one prompt to other prompts on the same model.
- **Inter-model transfer** is defined in Section 3.1 and measured by applying suffixes generated on a source model to a target model.
- **Refusal direction** is Definition 2. Stored `.pt` vectors provide the direction used by activation-based features.
- **Semantic similarity** is Definition 3. The workflow supports both external sentence embeddings and model-internal activation similarity where available.
- **Refusal connectivity** is Definition 4. It measures how aligned a harmful prompt activation is with the refusal direction.
- **Suffix push** is Definition 5. It measures how much adding a suffix moves the activation away from the refusal direction.
- **Orthogonal shift** is Definition 6. It measures how much adding a suffix changes the activation in directions perpendicular to refusal.
