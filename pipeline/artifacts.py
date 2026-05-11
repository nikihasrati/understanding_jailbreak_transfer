from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a project dependency.
    pd = None


CHUNK_NAME_RE = re.compile(r'^chunk_(\d+)\.json$')
LEGACY_PREFIX_RE = re.compile(r'^(\d+)_.*\.json$')


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def populated(value: Any) -> bool:
    if value is None:
        return False
    if pd is not None:
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
    try:
        if value != value:
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return False
    return not (isinstance(value, str) and value.strip() == '')


def assert_column_populated(table: Any, column: str, path: str | Path) -> None:
    if column not in table.columns:
        raise ValueError(f'{path} is missing required column {column!r}')
    missing = sum(not populated(value) for value in table[column].tolist())
    print(f'{column}: {len(table) - missing}/{len(table)} populated in {path}')
    if missing:
        raise RuntimeError(f'{missing} records in {path} are missing {column!r}')


def workflow_chunk_path(path: str | Path, chunk_id: int, stage: str) -> str:
    path = Path(path)
    directory = path.parent
    filename = path.name
    if stage not in {'generation', 'evaluation'}:
        raise ValueError(stage)

    preferred_subdir = f'{stage}_chunks'
    preferred = directory / preferred_subdir / f'chunk_{chunk_id:05d}.json'
    if preferred.exists():
        return str(preferred)

    legacy = directory / f'{chunk_id}_{filename}'
    if legacy.exists():
        return str(legacy)

    if stage == 'generation':
        canonical = directory / 'chunks' / f'chunk_{chunk_id:05d}.json'
        if canonical.exists():
            raise FileNotFoundError(
                f'Found canonical chunk {canonical}, but generation writes to temporary generation_chunks/. '
                'Run tools/prepare_generation_chunks.py for this artifact before generation.'
            )
    else:
        generation = directory / 'generation_chunks' / f'chunk_{chunk_id:05d}.json'
        if generation.exists():
            raise FileNotFoundError(
                f'Found generation chunk {generation}, but evaluation reads temporary evaluation_chunks/. '
                'Run tools/combine_completions.py to re-shard generation chunks for evaluation.'
            )
        canonical = directory / 'chunks' / f'chunk_{chunk_id:05d}.json'
        if canonical.exists():
            raise FileNotFoundError(
                f'Found canonical chunk {canonical}, but evaluation reads temporary evaluation_chunks/. '
                'Create evaluation_chunks/ before running the jailbreak judge.'
            )

    raise FileNotFoundError(f'No {stage} chunk found for chunk_id={chunk_id} under {directory}')


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else [data]


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def manifest_entries(manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if 'chunks' in manifest:
        return 'chunks', manifest['chunks']
    if 'files' in manifest:
        return 'files', manifest['files']
    raise KeyError("Manifest must contain either 'chunks' or 'files'")


def verify_manifest(manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    _, entries = manifest_entries(load_manifest(manifest_path))
    for item in entries:
        path = root / item['path']
        if not path.exists():
            raise FileNotFoundError(path)
        if 'bytes' in item and path.stat().st_size != item['bytes']:
            raise ValueError(f'Byte-size mismatch: {path}')
        actual = sha256_file(path)
        if actual != item['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')


def load_manifest_records(manifest_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    if 'chunks' not in manifest:
        raise KeyError(f'{manifest_path} is not a chunked JSON manifest')
    records: list[dict[str, Any]] = []
    for chunk in manifest['chunks']:
        records.extend(load_json_records(manifest_path.parent / chunk['path']))
    return manifest, records


def chunk_sort_key(path: Path) -> tuple[int, int | str]:
    match = CHUNK_NAME_RE.match(path.name)
    if match:
        return (0, int(match.group(1)))
    match = LEGACY_PREFIX_RE.match(path.name)
    if match:
        return (1, int(match.group(1)))
    return (2, path.name)


def json_chunk_paths(chunks_dir: str | Path) -> list[Path]:
    chunks_dir = Path(chunks_dir)
    paths = sorted(chunks_dir.glob('chunk_*.json'), key=chunk_sort_key)
    if paths:
        return paths
    return sorted(chunks_dir.glob('*.json'), key=chunk_sort_key)


def load_chunk_records(chunks_dir: str | Path) -> list[dict[str, Any]]:
    paths = json_chunk_paths(chunks_dir)
    if not paths:
        raise FileNotFoundError(f'No JSON chunks found in {chunks_dir}')
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_json_records(path))
    return records


def iter_chunk_records(chunks_dir: str | Path) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    paths = json_chunk_paths(chunks_dir)
    if not paths:
        raise FileNotFoundError(f'No JSON chunks found in {chunks_dir}')
    for path in paths:
        for index, row in enumerate(load_json_records(path)):
            yield path, index, row


def split_records(records: list[dict[str, Any]], num_chunks: int) -> list[list[dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError('num_chunks must be positive')
    chunk_size = len(records) // num_chunks
    chunks = []
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = (chunk_id + 1) * chunk_size if chunk_id != num_chunks - 1 else len(records)
        chunks.append(records[start:end])
    return chunks


def write_json_chunks(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    num_chunks: int,
    *,
    artifact_dir: str | Path | None = None,
    indent: int | None = None,
    clear_existing: bool = True,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for stale in output_dir.glob('chunk_*.json'):
            stale.unlink()

    chunk_info = []
    for chunk_id, chunk in enumerate(split_records(records, num_chunks)):
        path = output_dir / f'chunk_{chunk_id:05d}.json'
        if indent is None:
            payload = json.dumps(chunk, separators=(',', ':')) + '\n'
        else:
            payload = json.dumps(chunk, indent=indent) + '\n'
        path.write_text(payload)
        manifest_path = path.relative_to(artifact_dir) if artifact_dir else path
        chunk_info.append(
            {
                'path': str(manifest_path),
                'records': len(chunk),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }
        )
    return chunk_info


def required_fields(stage: str) -> list[str]:
    if stage == 'generation':
        return ['response']
    if stage == 'evaluation':
        return ['response', 'jailbroken']
    if stage == 'none':
        return []
    raise ValueError(stage)


def count_missing_fields(records: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    return {
        field: sum(1 for record in records if not populated(record.get(field)))
        for field in fields
    }
