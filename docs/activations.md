# Activation Artifacts

Activations are the internal vectors produced by a model layer while processing a prompt or jailbreak string. They are the tensors used for the paper's semantic-similarity, refusal-connectivity, suffix-push, and orthogonal-shift analyses in Sections 5.2-5.5. They are large, model-specific tensors, so this repo documents how to regenerate them instead of committing them to Git.

## Table of Contents

- [Formats](#formats)
- [Regenerate Prompt Activations](#regenerate-prompt-activations)
- [Regenerate Multi-Seed Jailbreak Activations](#regenerate-multi-seed-jailbreak-activations)
- [Export Nested Layout](#export-nested-layout)
- [Paper Mapping](#paper-mapping)

## Formats

`canonical_tensor_chunks` is the default project format. It stores a sequence of tensor chunks plus a manifest. This is the format to use when regenerating tensors for the paper's Definitions 4-6. Each tensor has shape:

```text
[num_inputs_in_chunk, num_layers, hidden_dimension]
```

Here, an input is one string sent through the model, such as one harmful prompt, one suffix-augmented jailbreak prompt, or one multi-seed jailbreak string. Example manifest entry:

```json
{
  "artifact_format": "canonical_tensor_chunks",
  "input_kind": "multi_seed_jailbreak",
  "num_examples": 250000,
  "num_chunks": 32,
  "chunks": [
    {
      "path": "activations_chunk_00000.pt",
      "shape": [7813, 28, 2048]
    }
  ]
}
```

`nested_by_suffix_layer` is an alternate export layout for code that wants to index by suffix first and layer second, which can be convenient when plotting suffix-level effects like Paper Figure 6 and Appendix C Figures 7-9. The exported object is a nested mapping:

```text
{suffix_id -> {layer_id -> [prompt_0_vector, prompt_1_vector, ...]}}
```

Example entry conceptually looks like:

```python
activations_by_suffix_layer[17][12][3]
# activation vector for suffix 17, layer 12, prompt 3
```

## Regenerate Prompt Activations

```bash
python -m pipeline.activations.save_activations \
  --model-path "$MODEL_ID" \
  --input-kind prompts \
  --output-format canonical_tensor_chunks
```

## Regenerate Multi-Seed Jailbreak Activations

Before running this command, rebuild the relevant multi-seed `combined.json` file from its manifest as described in [artifacts.md](artifacts.md). The examples use `NUM_ACTIVATION_CHUNKS` and `NUM_PROMPTS` so you can change chunking and prompt count in one place.

```bash
export NUM_ACTIVATION_CHUNKS=32
export NUM_PROMPTS=100
```

```bash
python -m pipeline.activations.save_activations \
  --model-path "$MODEL_ID" \
  --input-kind multi_seed_jailbreak \
  --num-chunks "$NUM_ACTIVATION_CHUNKS" \
  --output-format canonical_tensor_chunks
```

Outputs are written under `outputs/activations/<model>/...`, which is intentionally ignored by Git.

## Export Nested Layout

```bash
python -m pipeline.activations.export_activations \
  --input-dir outputs/activations/$MODEL_ALIAS/multi_seed_jailbreak_activations_transfer/canonical_tensor_chunks \
  --output-dir outputs/activations/$MODEL_ALIAS/multi_seed_jailbreak_activations_transfer/nested_by_suffix_layer \
  --format nested_by_suffix_layer \
  --num-prompts "$NUM_PROMPTS"
```

## Paper Mapping

- Prompt-only activations support refusal connectivity in Definition 4 and the model-susceptibility plots in Paper Section 5.3.
- Prompt-plus-suffix activations support suffix push in Definition 5 and orthogonal shift in Definition 6.
- Multi-seed jailbreak activations support the joint statistical analyses in Paper Sections 5.4-5.5 and Appendix C.
- Activations should be regenerated at the same selected refusal-direction layer described in Appendix A when reproducing the reported features.
