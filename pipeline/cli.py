from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Command:
    module: str
    aliases: dict[str, str] = field(default_factory=dict)


COMMANDS: dict[tuple[str, str], Command] = {
    ('artifacts', 'split-json'): Command('tools.split_json_records'),
    ('artifacts', 'split-json-records'): Command('tools.split_json_records'),
    ('artifacts', 'combine-json'): Command('tools.combine_json_chunks'),
    ('artifacts', 'combine-json-chunks'): Command('tools.combine_json_chunks'),
    ('artifacts', 'verify-manifest'): Command('tools.verify_manifest'),
    ('artifacts', 'make-manifest'): Command('tools.make_manifest'),
    ('artifacts', 'prepare-generation'): Command('tools.prepare_generation_chunks'),
    ('artifacts', 'combine-completions'): Command('tools.combine_completions'),
    ('artifacts', 'check-completions'): Command('tools.check_completions'),
    ('artifacts', 'split-tensor'): Command('tools.split_torch_tensor'),
    ('artifacts', 'split-torch-tensor'): Command('tools.split_torch_tensor'),
    ('artifacts', 'combine-tensor'): Command('tools.combine_torch_chunks'),
    ('artifacts', 'combine-torch-chunks'): Command('tools.combine_torch_chunks'),
    ('suffixes', 'run-gcg'): Command('pipeline.suffix_generation.run_gcg'),
    ('suffixes', 'create-datasets'): Command('pipeline.suffix_generation.create_datasets'),
    ('completions', 'generate'): Command('pipeline.generation.generate_completions'),
    ('completions', 'evaluate'): Command(
        'pipeline.evaluation.evaluate_completions',
        aliases={
            '--model-path': '--model_path',
            '--num-gpus': '--num_gpus',
            '--multi-seed': '--multi_seed',
            '--no-multi-seed': '--no-multi_seed',
            '--chunk-id': '--chunk_id',
            '--no-suffix-completions': '--no_suffix_completions',
            '--no-no-suffix-completions': '--no-no_suffix_completions',
            '--gcg-push': '--gcg_push',
            '--no-gcg-push': '--no-gcg_push',
            '--suffix-push': '--suffix_push',
            '--no-suffix-push': '--no-suffix_push',
            '--orth-shift': '--orth_shift',
            '--no-orth-shift': '--no-orth_shift',
        },
    ),
    ('completions', 'normalize-labels'): Command('pipeline.evaluation.normalize_jailbreak_labels'),
    ('activations', 'save'): Command('pipeline.activations.save_activations'),
    ('activations', 'export'): Command('pipeline.activations.export_activations'),
    ('cross-model', 'setup'): Command(
        'pipeline.cross_model.set_up_dataset',
        aliases={'--source-model-path': '--source_model_path', '--target-model-path': '--target_model_path'},
    ),
    ('cross-model', 'generate'): Command(
        'pipeline.cross_model.generate_completions',
        aliases={
            '--source-model-path': '--source_model_path',
            '--target-model-path': '--target_model_path',
            '--chunk-id': '--chunk_id',
        },
    ),
    ('cross-model', 'evaluate'): Command(
        'pipeline.cross_model.evaluate_completions',
        aliases={
            '--source-model-path': '--source_model_path',
            '--target-model-path': '--target_model_path',
            '--chunk-id': '--chunk_id',
            '--num-gpus': '--num_gpus',
        },
    ),
    ('cross-model', 'combine'): Command('pipeline.cross_model.combine_dataset'),
    ('cross-model', 'analyze'): Command(
        'pipeline.cross_model.data_analysis',
        aliases={'--source-model-path': '--source_model_path', '--target-model-path': '--target_model_path'},
    ),
    ('cross-model', 'check'): Command(
        'pipeline.cross_model.check_all_exist',
        aliases={'--source-model-path': '--source_model_path', '--target-model-path': '--target_model_path'},
    ),
    ('analysis', 'multi-seed'): Command(
        'pipeline.analysis.multi_seed_data_analysis',
        aliases={'--model-path': '--model_path'},
    ),
    ('analysis', 'paper'): Command('pipeline.analysis.data_analysis', aliases={'--model-path': '--model_path'}),
    ('gcg-push', 'launch'): Command('pipeline.gcg_push.launch_experiment'),
    ('gcg-push', 'run'): Command('pipeline.gcg_push.run_gcg'),
    ('gcg-push', 'create-datasets'): Command('pipeline.gcg_push.create_datasets'),
    ('gcg-push', 'analyze'): Command(
        'pipeline.gcg_push.data_analysis',
        aliases={
            '--model-path': '--model_path',
            '--suffix-push': '--suffix_push',
            '--no-suffix-push': '--no-suffix_push',
            '--orth-shift': '--orth_shift',
            '--no-orth-shift': '--no-orth_shift',
        },
    ),
    ('refusal', 'inspect'): Command('pipeline.refusal_directions.inspect_model'),
    ('prompt-rephrasings', 'setup'): Command(
        'pipeline.prompt_rephrasings.setup_dataset',
        aliases={'--model-path': '--model_path'},
    ),
}


def command_names(group: str) -> list[str]:
    return sorted(command for candidate_group, command in COMMANDS if candidate_group == group)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m pipeline',
        description='Canonical CLI for the jailbreak-transfer reproducibility pipeline.',
    )
    parser.add_argument('group', nargs='?', choices=sorted({group for group, _ in COMMANDS}))
    parser.add_argument('command', nargs='?')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    return parser


def translate_args(args: Sequence[str], aliases: dict[str, str]) -> list[str]:
    return [aliases.get(arg, arg) for arg in args]


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if parsed.group is None:
        parser.print_help()
        return
    if parsed.command is None:
        print(f'Available {parsed.group} commands:')
        for command in command_names(parsed.group):
            print(f'  {command}')
        return

    key = (parsed.group, parsed.command)
    if key not in COMMANDS:
        parser.error(
            f"unknown command {parsed.group} {parsed.command!r}; "
            f"available: {', '.join(command_names(parsed.group))}"
        )
    command = COMMANDS[key]
    translated = translate_args(parsed.args, command.aliases)
    module = importlib.import_module(command.module)
    module.main(translated)
