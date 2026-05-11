import json
from pathlib import Path

import pandas as pd


class Config:
    def __init__(self, model_path: str):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.model_path = model_path
        self.model_alias = self._get_model_alias(model_path)
        self.DATASET_DIR = self.repo_root / 'data'
        self.REFUSAL_DIR_PATH = self.DATASET_DIR / 'refusal_directions'
        self.PROCESSED_DATASET_DIR = self.DATASET_DIR / 'processed'
        self.FIGURES = str(self.repo_root / 'figures')
        self.num_suffixes_per_prompt = 100
        self._num_prompts = None

    def _path_obj(self, *parts) -> Path:
        return Path(*parts)

    def _path(self, *parts) -> str:
        return str(self._path_obj(*parts))

    @property
    def num_prompts(self) -> int:
        if self._num_prompts is None:
            prompts_path = self._path_obj(self.prompts_path())
            self._num_prompts = len(pd.read_json(prompts_path)) if prompts_path.exists() else 100
        return self._num_prompts

    def _model_name(self, model_alias: str) -> str:
        if 'llama-2' in model_alias:
            return 'Llama 2'
        if 'llama-3.2' in model_alias:
            return 'Llama 3.2'
        if 'vicuna' in model_alias:
            return 'Vicuna'
        if 'qwen2.5' in model_alias:
            return 'Qwen 2.5'
        if 'phi-3' in model_alias:
            return 'Phi 3'
        return model_alias

    def model_name(self) -> str:
        return self._model_name(self.model_alias)

    def prompts_path(self) -> str:
        return self._path(self.PROCESSED_DATASET_DIR, 'jailbreakbench_prompts', 'prompts', 'chunks', 'chunk_00000.json')

    def suffixes_path(self) -> str:
        return self._path(self.PROCESSED_DATASET_DIR, 'jailbreakbench_suffixes', f'{self.model_alias}_suffixes', 'chunks', 'chunk_00000.json')

    def cross_prompt_transfer_generations_path(self) -> str:
        return self._path(self.DATASET_DIR, 'cross_prompt_transfer_generations', f'{self.model_alias}_cross_prompt_transfer_generations', 'combined.json')

    def cross_model_transfer_generations_dir(self) -> str:
        return self._path(self.DATASET_DIR, 'cross_model_transfer_generations')

    def no_suffix_generations_path(self) -> str:
        return self._path(self.DATASET_DIR, 'no_suffix_generations', f'{self.model_alias}_no_suffix_generations', 'combined.json')

    def multi_seed_generations_no_transfer_path(self) -> str:
        return self._path(self.DATASET_DIR, 'multiple_seed_results', self.model_alias, 'no_transfer', f'{self.model_alias}_multiple_seed_results_no_transfer', 'combined.json')

    def multi_seed_generations_transfer_path(self) -> str:
        return self._path(self.DATASET_DIR, 'multiple_seed_results', self.model_alias, 'transfer', f'{self.model_alias}_multiple_seed_results_transfer', 'combined.json')

    def multi_seed_generations_transfer_dir(self) -> str:
        return str(self._path_obj(self.multi_seed_generations_transfer_path()).parent)

    def gcg_push_root_dir(self) -> str:
        return self._path(self.DATASET_DIR, 'gcg_push_results', self.model_alias)

    def gcg_push_artifact_dir(self, intervention: str, coeff: str, split: str = 'transfer') -> str:
        return self._path(self.gcg_push_root_dir(), intervention, f'coeff-{coeff}', split)

    def gcg_push_transfer_dir(self) -> str:
        return self._path(self.gcg_push_root_dir(), 'transfer')

    def gcg_push_orth_shift_transfer_path(self, coeff: str) -> str:
        return self._path(self.gcg_push_artifact_dir('orth_shift', coeff, 'transfer'), 'combined.json')

    def gcg_push_suffix_push_transfer_path(self, coeff: str) -> str:
        return self._path(self.gcg_push_artifact_dir('suffix_push', coeff, 'transfer'), 'combined.json')

    def activations_dir(self) -> str:
        return self._path(self.repo_root, 'outputs', 'activations')

    def prompt_activations_path(self) -> str:
        return self._path(self.activations_dir(), self.model_alias, 'prompt_activations', 'canonical_tensor_chunks')

    def suffix_activations_path(self) -> str:
        return self._path(self.activations_dir(), self.model_alias, 'suffix_activations', 'canonical_tensor_chunks')

    def jailbreak_activations_dir(self) -> str:
        return self._path(self.activations_dir(), self.model_alias, 'jailbreak_activations', 'canonical_tensor_chunks')

    def multi_seed_jailbreak_activations_transfer_dir(self) -> str:
        return self._path(self.activations_dir(), self.model_alias, 'multi_seed_jailbreak_activations_transfer', 'canonical_tensor_chunks')

    def prompt_rephrasings_dir(self) -> str:
        return self._path(self.DATASET_DIR, 'prompt_rephrasings')

    def prompt_rephrasings_path(self) -> str:
        return self._path(self.prompt_rephrasings_dir(), f'{self.model_alias}_prompt_rephrasings', 'combined.json')

    def unprocessed_prompt_rephrasings_path(self) -> str:
        return self._path(self.prompt_rephrasings_dir(), f'{self.model_alias}_unprocessed_rephrasings.json')

    def cosine_similarity_with_refusal_path(self) -> str:
        return self._path(self.activations_dir(), self.model_alias, 'cosine_sim_with_refusal', 'cosine_similarity_with_refusal.pt')

    def arditi_et_al_refusal_direction_dir(self) -> str:
        return self._path(self.REFUSAL_DIR_PATH, 'arditi_et_al_2024', self.model_alias)

    def arditi_et_al_refusal_direction_path(self) -> str:
        return self._path(self.arditi_et_al_refusal_direction_dir(), 'direction.pt')

    def arditi_et_al_refusal_direction_metadata_path(self) -> str:
        return self._path(self.arditi_et_al_refusal_direction_dir(), 'direction_metadata.json')

    def arditi_et_al_refusal_direction_layer(self) -> int:
        with open(self.arditi_et_al_refusal_direction_metadata_path(), 'r') as f:
            data = json.load(f)
        return data['layer']

    def refusal_direction_dir(self) -> str:
        return self._path(self.REFUSAL_DIR_PATH, 'ours', self.model_alias)

    def _get_model_alias(self, model_path: str) -> str:
        return Path(model_path).name.lower()
