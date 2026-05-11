# Refusal Directions

This project uses refusal directions generated with the method from *Refusal in Language Models Is Mediated by a Single Direction* by Arditi et al. Those directions are the external baseline direction vectors used for the paper's Definition 2, Sections 3.2-3.3, and Appendix A.

Refusal-direction vectors are stored under this layout:

```text
data/refusal_directions/arditi_et_al_2024/<model>/
  direction.pt
  direction_metadata.json
```

The tensor is the selected Arditi et al. refusal direction for that model. The metadata records the source position and layer used when the direction was selected. Downstream activation analyses use this vector to compute refusal connectivity, suffix push, and orthogonal shift.

## Table of Contents

- [Inspect a Stored Direction](#inspect-a-stored-direction)
- [Regenerate Directions](#regenerate-directions)

## Inspect a Stored Direction

```bash
python -m pipeline.refusal_directions.inspect_model \
  --model-path "$MODEL_ID" \
  --prompt "Write a short explanation of machine learning."
```

This command loads the model and its stored `direction.pt`, runs the prompt through the model, extracts the activation vector at the selected layer, and prints the dot product and cosine similarity between that activation and the refusal direction. It is a quick sanity check that the direction file, metadata file, model wrapper, and activation extraction path all agree.

## Regenerate Directions

Directions are generated with the method from `Refusal in Language Models Is Mediated by a Single Direction` using the external `refusal_direction` repository:

```text
https://github.com/andyrdt/refusal_direction
```

This repository includes the project-specific model wrappers needed for models that are not all present in the canonical external repository. They live in:

```text
external/refusal_direction_model_utils/
```

To use them:

```bash
REFUSAL_DIRECTION_REPO=/path/to/refusal_direction
cp external/refusal_direction_model_utils/*.py \
  "$REFUSAL_DIRECTION_REPO/pipeline/model_utils/"
```

Then run the external pipeline from the external repository root. Set the same cache variables used for this project first, plus the ablation-count variable used by the command:

```bash
cd "$REFUSAL_DIRECTION_REPO"
export MODEL_CACHE_ROOT=/path/to/model/cache/root
export HF_HOME="$MODEL_CACHE_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_CACHE="$HF_HOME/hub"
export REFUSAL_NUM_DIRS_TO_ABLATE=0

python -m pipeline.run_pipeline \
  --model_path "$MODEL_ID" \
  --num_dirs_to_ablate "$REFUSAL_NUM_DIRS_TO_ABLATE"
```

The external run writes artifacts under a run directory such as:

```text
$REFUSAL_DIRECTION_REPO/pipeline/runs/<model-basename>_top_${REFUSAL_NUM_DIRS_TO_ABLATE}_dirs_ablated/
```

Copy the selected direction into this repo using the naming convention expected by `Config`:

```bash
mkdir -p data/refusal_directions/arditi_et_al_2024/$MODEL_ALIAS
cp "$REFUSAL_DIRECTION_REPO/pipeline/runs/<model-basename>_top_${REFUSAL_NUM_DIRS_TO_ABLATE}_dirs_ablated/direction.pt" \
  data/refusal_directions/arditi_et_al_2024/$MODEL_ALIAS/direction.pt
cp "$REFUSAL_DIRECTION_REPO/pipeline/runs/<model-basename>_top_${REFUSAL_NUM_DIRS_TO_ABLATE}_dirs_ablated/direction_metadata.json" \
  data/refusal_directions/arditi_et_al_2024/$MODEL_ALIAS/direction_metadata.json
```

Finally, update the refusal-direction manifest:

```bash
python tools/make_manifest.py \
  --root data/refusal_directions/arditi_et_al_2024 \
  --output data/refusal_directions/arditi_et_al_2024/manifest.json
```
