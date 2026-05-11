from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'gcg_push_paper.example.yaml'


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open('r') as f:
        config = yaml.safe_load(f) or {}
    config['_config_path'] = str(config_path)
    return config


def repo_path(value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def model_id(config: dict[str, Any]) -> str:
    return config['experiment']['model_id']


def model_alias(config: dict[str, Any]) -> str:
    return config['experiment'].get('model_alias') or os.path.basename(model_id(config)).lower()


def experiment_name(config: dict[str, Any]) -> str:
    return config.get('experiment', {}).get('name', 'gcg_push')


def indices(config: dict[str, Any]) -> list[int]:
    experiment = config.get('experiment', {})
    if 'indices' in experiment and experiment['indices'] is not None:
        return [int(index) for index in experiment['indices']]
    indices_file = experiment.get('indices_file')
    if not indices_file:
        return []
    path = repo_path(indices_file, must_exist=True)
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def interventions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return config.get('interventions', {})


def selected_interventions(config: dict[str, Any], selected: str | None = None) -> list[str]:
    available = interventions(config)
    if selected:
        if selected not in available:
            raise ValueError(f'Unknown intervention {selected!r}. Available: {sorted(available)}')
        return [selected]
    return list(available)


def coefficients(config: dict[str, Any], intervention: str) -> list[str]:
    values = interventions(config)[intervention].get('coefficients', [])
    return [str(value) for value in values]


def selected_coefficients(config: dict[str, Any], intervention: str, selected: str | None = None) -> list[str]:
    available = coefficients(config, intervention)
    if selected:
        selected = str(selected)
        if selected not in available:
            raise ValueError(f'Unknown coefficient {selected!r} for {intervention}. Available: {available}')
        return [selected]
    return available


def seed_range(config: dict[str, Any], intervention: str) -> tuple[int, int]:
    item = interventions(config)[intervention]
    seeds = item.get('seeds', {})
    start = int(seeds.get('start', 0))
    if 'end' in seeds:
        end = int(seeds['end'])
    else:
        end = start + int(seeds.get('count', 1)) - 1
    if end < start:
        raise ValueError(f'Invalid seed range for {intervention}: {start}..{end}')
    return start, end


def num_seeds(config: dict[str, Any], intervention: str) -> int:
    start, end = seed_range(config, intervention)
    return end - start + 1


def coefficient_dir(coeff: str | float) -> str:
    return f'coeff-{coeff}'


def raw_results_root(config: dict[str, Any]) -> Path:
    return repo_path(config.get('paths', {}).get('raw_results_root', 'outputs/gcg_push/raw'))


def raw_results_dir(config: dict[str, Any], intervention: str, coeff: str | float) -> Path:
    return raw_results_root(config) / model_alias(config) / experiment_name(config) / intervention / coefficient_dir(coeff)


def raw_result_path(config: dict[str, Any], intervention: str, coeff: str | float, index: int, seed: int) -> Path:
    return raw_results_dir(config, intervention, coeff) / f'index-{index:04d}' / f'seed-{seed:04d}' / 'results.json'


def artifact_root(config: dict[str, Any]) -> Path:
    return repo_path(config.get('paths', {}).get('artifact_root', 'data/gcg_push_results'))


def artifact_dir(config: dict[str, Any], intervention: str, coeff: str | float, split: str = 'transfer') -> Path:
    return artifact_root(config) / model_alias(config) / intervention / coefficient_dir(coeff) / split


def refusal_direction_path(config: dict[str, Any]) -> Path:
    explicit = config.get('paths', {}).get('refusal_direction_path')
    if explicit:
        return repo_path(explicit, must_exist=True)
    return repo_path(f'data/refusal_directions/arditi_et_al_2024/{model_alias(config)}/direction.pt', must_exist=True)


def loss_type(config: dict[str, Any], intervention: str) -> str:
    return str(interventions(config)[intervention].get('loss_type', 'mse'))


def num_steps(config: dict[str, Any]) -> int:
    return int(config.get('experiment', {}).get('num_steps', 500))


def artifact_chunks(config: dict[str, Any], intervention: str, split: str) -> int:
    item = interventions(config)[intervention]
    artifacts = item.get('artifacts', {})
    if split == 'transfer':
        return int(artifacts.get('transfer_chunks', config.get('artifacts', {}).get('transfer_chunks', 1)))
    return int(artifacts.get('no_transfer_chunks', config.get('artifacts', {}).get('no_transfer_chunks', 1)))


def workflow_chunks(config: dict[str, Any], intervention: str, stage: str) -> int:
    item = interventions(config)[intervention]
    workflow = item.get('workflow', {})
    return int(workflow.get(stage, config.get('workflow', {}).get(stage, 1)))


def slurm_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get('slurm', {}).get(key, default)
