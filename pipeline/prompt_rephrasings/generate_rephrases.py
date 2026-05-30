from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pipeline.config import Config
from pipeline.paper.common import read_json_records


DEFAULT_REPHRASE_MODEL = "lmsys/vicuna-13b-v1.5"
DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"

REPHRASE_PROMPT = """Write {num_rephrases} semantically equivalent rephrasings of the request below.
Keep the original intent unchanged while varying the wording and structure.
Return only a numbered list.

Request:
{prompt}"""

ITEM_RE = re.compile(r"^\s*(?:\d{1,3}[\).\:-]|\*|-)\s*(.+?)\s*$")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt rephrases and refusal-direction scores for the prompt-rephrasing intervention."
    )
    parser.add_argument("--model-path", "--model_path", dest="model_path", required=True)
    parser.add_argument("--rephrase-model-path", default=DEFAULT_REPHRASE_MODEL)
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--prompts-path", default=None)
    parser.add_argument("--refusal-direction-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--num-rephrases", type=int, default=10)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated prompt IDs or ranges, for example 0,2,10-19.")
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--rephrase-batch-size", type=int, default=1)
    parser.add_argument("--activation-batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--allow-partial", action="store_true", help="Write rows even if fewer than --num-rephrases parse.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in --output-path.")
    return parser.parse_args(argv)


parse_arguments = parse_args


def parse_prompt_ids(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    prompt_ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid prompt ID range {part!r}")
            prompt_ids.update(range(start, end + 1))
        else:
            prompt_ids.add(int(part))
    return prompt_ids


def clean_rephrase(text: str) -> str:
    text = text.strip()
    text = text.strip("\"'")
    return text.strip()


def extract_rephrases(text: str, expected_count: int) -> list[str]:
    rephrases: list[str] = []
    for line in text.splitlines():
        match = ITEM_RE.match(line)
        if not match:
            continue
        candidate = clean_rephrase(match.group(1))
        if candidate:
            rephrases.append(candidate)
        if len(rephrases) >= expected_count:
            return rephrases[:expected_count]

    if rephrases:
        return rephrases[:expected_count]

    chunks = [clean_rephrase(chunk) for chunk in re.split(r"\n\s*\n", text) if clean_rephrase(chunk)]
    return chunks[:expected_count]


def _prompt_id(record: dict[str, Any], fallback: int) -> int:
    for key in ("prompt_id", "id", "index"):
        if key in record:
            return int(record[key])
    return fallback


def _select_prompts(records: list[dict[str, Any]], prompt_ids: set[int] | None, prompt_limit: int | None) -> list[dict[str, Any]]:
    selected = []
    for fallback, record in enumerate(records):
        pid = _prompt_id(record, fallback)
        if prompt_ids is not None and pid not in prompt_ids:
            continue
        selected.append({"prompt_id": pid, "prompt": record["prompt"]})
        if prompt_limit is not None and len(selected) >= prompt_limit:
            break
    return selected


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(record["prompt_id"]): record for record in data}


def _has_rephrases(record: dict[str, Any], count: int) -> bool:
    return all(record.get(f"paraphrase_{idx}") for idx in range(count))


def _has_scores(record: dict[str, Any], count: int) -> bool:
    if record.get("ori_dot_product") is None:
        return False
    return all(record.get(f"dot_product_{idx}") is not None and record.get(f"similarity_{idx}") is not None for idx in range(count))


def _write_records(records_by_id: dict[int, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [records_by_id[pid] for pid in sorted(records_by_id)]
    output_path.write_text(json.dumps(records, indent=2) + "\n")


def _construct_model(model_path: str):
    from pipeline.model_utils.model_factory import construct_model_base

    return construct_model_base(model_path)


def _release_model(model: Any) -> None:
    try:
        model.del_model()
    except AttributeError:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_rephrases(
    args: argparse.Namespace,
    records_by_id: dict[int, dict[str, Any]],
    prompts: list[dict[str, Any]],
    output_path: Path,
) -> None:
    pending = [
        prompt
        for prompt in prompts
        if not (args.resume and _has_rephrases(records_by_id.get(prompt["prompt_id"], {}), args.num_rephrases))
    ]
    if not pending:
        return

    rephrase_model = _construct_model(args.rephrase_model_path)
    try:
        for batch in _chunks(pending, args.rephrase_batch_size):
            instructions = [
                REPHRASE_PROMPT.format(num_rephrases=args.num_rephrases, prompt=item["prompt"])
                for item in batch
            ]
            outputs = rephrase_model.generate_completions(
                instructions,
                batch_size=args.rephrase_batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            for item, output in zip(batch, outputs):
                rephrases = extract_rephrases(output, args.num_rephrases)
                if len(rephrases) < args.num_rephrases and not args.allow_partial:
                    raise RuntimeError(
                        f"Parsed {len(rephrases)} rephrases for prompt_id={item['prompt_id']}; "
                        f"expected {args.num_rephrases}."
                    )
                record = {
                    "prompt_id": int(item["prompt_id"]),
                    "prompt": item["prompt"],
                }
                for idx, rephrase in enumerate(rephrases):
                    record[f"paraphrase_{idx}"] = rephrase
                records_by_id[int(item["prompt_id"])] = record
            _write_records(records_by_id, output_path)
    finally:
        _release_model(rephrase_model)


def _dot_products(model: Any, texts: list[str], layer: int, direction_path: str, batch_size: int) -> list[float]:
    import torch

    from pipeline.submodules.generate_activations import get_activations

    direction = torch.load(direction_path, map_location="cpu", weights_only=True).float()
    activations = get_activations(
        model,
        texts,
        model.model_block_modules,
        batch_size=batch_size,
    )
    vectors = activations[:, layer, :].cpu().float()
    return torch.matmul(vectors, direction).tolist()


def _similarities(pairs: list[tuple[str, str]], sentence_model: str, batch_size: int) -> list[float]:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(sentence_model)
    originals = [pair[0] for pair in pairs]
    rephrases = [pair[1] for pair in pairs]
    original_embeddings = embedder.encode(
        originals,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    rephrase_embeddings = embedder.encode(
        rephrases,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return (original_embeddings * rephrase_embeddings).sum(axis=1).tolist()


def _score_rephrases(
    args: argparse.Namespace,
    cfg: Config,
    records_by_id: dict[int, dict[str, Any]],
    prompts: list[dict[str, Any]],
    output_path: Path,
) -> None:
    selected_ids = [int(prompt["prompt_id"]) for prompt in prompts]
    records = [
        records_by_id[pid]
        for pid in selected_ids
        if pid in records_by_id and not (args.resume and _has_scores(records_by_id[pid], args.num_rephrases))
    ]
    if not records:
        return

    texts: list[str] = []
    positions: list[tuple[dict[str, Any], str]] = []
    for record in records:
        texts.append(record["prompt"])
        positions.append((record, "ori_dot_product"))
        for idx in range(args.num_rephrases):
            key = f"paraphrase_{idx}"
            if key not in record:
                continue
            texts.append(record[key])
            positions.append((record, f"dot_product_{idx}"))

    target_model = _construct_model(args.model_path)
    try:
        dots = _dot_products(
            target_model,
            texts,
            cfg.arditi_et_al_refusal_direction_layer(),
            args.refusal_direction_path or cfg.arditi_et_al_refusal_direction_path(),
            args.activation_batch_size,
        )
    finally:
        _release_model(target_model)

    for (record, key), value in zip(positions, dots):
        record[key] = float(value)
    _write_records(records_by_id, output_path)

    pairs: list[tuple[str, str]] = []
    pair_positions: list[tuple[dict[str, Any], str]] = []
    for record in records:
        for idx in range(args.num_rephrases):
            key = f"paraphrase_{idx}"
            if key not in record:
                continue
            pairs.append((record["prompt"], record[key]))
            pair_positions.append((record, f"similarity_{idx}"))

    if not pairs:
        return
    similarities = _similarities(pairs, args.sentence_model, args.embedding_batch_size)
    for (record, key), value in zip(pair_positions, similarities):
        record[key] = float(value)
    _write_records(records_by_id, output_path)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(args.model_path)
    prompts_path = args.prompts_path or cfg.prompts_path()
    output_path = Path(args.output_path or cfg.unprocessed_prompt_rephrasings_path())
    prompts = _select_prompts(read_json_records(prompts_path), parse_prompt_ids(args.prompt_ids), args.prompt_limit)
    if not prompts:
        raise ValueError("No prompts selected.")

    records_by_id = _load_existing(output_path) if args.resume else {}
    _generate_rephrases(args, records_by_id, prompts, output_path)
    _write_records(records_by_id, output_path)
    _score_rephrases(args, cfg, records_by_id, prompts, output_path)
    _write_records(records_by_id, output_path)
    print(f"Wrote {len(records_by_id)} prompt rephrasing rows to {output_path}")


if __name__ == "__main__":
    main()
