from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch


def _tensor_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    manifest = path / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        return [manifest.parent / chunk["path"] for chunk in data["chunks"]]
    paths = sorted(candidate for candidate in path.iterdir() if candidate.suffix in {".pt", ".pth"})
    if not paths:
        raise FileNotFoundError(f"No tensor files found in {path}")
    return paths


def load_tensor_artifact(path: str | Path, *, map_location: str = "cpu") -> torch.Tensor:
    paths = _tensor_paths(Path(path))
    tensors = [torch.load(candidate, map_location=map_location, weights_only=True) for candidate in paths]
    if len(tensors) == 1:
        tensor = tensors[0]
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)
        return tensor
    return torch.cat(tensors, dim=0)


def iter_tensor_chunks(path: str | Path, *, map_location: str = "cpu") -> Iterator[torch.Tensor]:
    for candidate in _tensor_paths(Path(path)):
        tensor = torch.load(candidate, map_location=map_location, weights_only=True)
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)
        yield tensor


def select_layer(activations: torch.Tensor, layer: int) -> torch.Tensor:
    if activations.ndim == 2:
        return activations
    if activations.ndim < 3:
        raise ValueError(f"Expected activation tensor with 2 or 3+ dims, got shape {tuple(activations.shape)}")
    return activations[:, layer, :]


def cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    vectors = vectors.float()
    denom = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vectors / denom
    return normed @ normed.T


def dot_with_direction(vectors: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return torch.matmul(vectors.float(), direction.float())

