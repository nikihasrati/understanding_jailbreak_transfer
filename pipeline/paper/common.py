from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline.artifacts import load_artifact_dataframe, load_manifest_records


PAPER_MODEL_ALIASES = [
    "qwen2.5-3b-instruct",
    "llama-3.2-1b-instruct",
    "vicuna-13b-v1.5",
    "llama-2-7b-chat-hf",
]

DISPLAY_NAMES = {
    "qwen2.5-3b-instruct": "Qwen",
    "llama-3.2-1b-instruct": "Llama 3.2",
    "vicuna-13b-v1.5": "Vicuna",
    "llama-2-7b-chat-hf": "Llama 2",
}


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.name == "manifest.json":
        return load_manifest_records(path)[1]
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.exists():
            return load_manifest_records(manifest)[1]
        raise FileNotFoundError(f"No manifest.json found in {path}")
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.name == "manifest.json" or path.is_dir():
        return pd.DataFrame(read_json_records(path))
    return load_artifact_dataframe(path)


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = ensure_parent(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def write_json(data: dict[str, Any] | list[Any], output_path: str | Path) -> Path:
    output_path = ensure_parent(output_path)
    output_path.write_text(json.dumps(data, indent=2) + "\n")
    return output_path


def source_prompt_id_from_suffix(
    suffix_id: int,
    *,
    suffixes_per_prompt: int,
    one_dimensional: bool = False,
) -> int:
    if one_dimensional:
        return int(suffix_id)
    return int(suffix_id) // suffixes_per_prompt


def add_source_prompt_ids(
    df: pd.DataFrame,
    *,
    suffixes_per_prompt: int,
    one_dimensional: bool = False,
    suffix_column: str = "suffix_id",
    output_column: str = "source_prompt_id",
) -> pd.DataFrame:
    result = df.copy()
    if one_dimensional:
        result[output_column] = result[suffix_column].astype(int)
    else:
        result[output_column] = result[suffix_column].astype(int) // suffixes_per_prompt
    return result


def filter_transfer_rows(
    df: pd.DataFrame,
    *,
    source_column: str = "source_prompt_id",
    target_column: str = "prompt_id",
    exclude_source: bool = True,
) -> pd.DataFrame:
    result = df.copy()
    if "transfer" in result.columns:
        if exclude_source:
            result = result[bool_series(result["transfer"])]
    elif exclude_source:
        result = result[result[target_column].astype(int) != result[source_column].astype(int)]
    return result.reset_index(drop=True)


def iter_manifest_paths(manifest_path: str | Path) -> Iterable[Path]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["chunks"]:
        yield manifest_path.parent / entry["path"]


def cross_model_artifact_path(repo_root: str | Path, source_alias: str, target_alias: str) -> Path:
    return (
        Path(repo_root)
        / "data"
        / "cross_model_transfer_generations"
        / f"{source_alias}_to_{target_alias}"
        / "cross_model_transfer_generations"
        / "combined.json"
    )


def cross_model_activation_dir(repo_root: str | Path, source_alias: str, target_alias: str) -> Path:
    return (
        Path(repo_root)
        / "outputs"
        / "activations"
        / f"{source_alias}_to_{target_alias}"
        / "cross_model_jailbreak_activations"
        / "canonical_tensor_chunks"
    )
