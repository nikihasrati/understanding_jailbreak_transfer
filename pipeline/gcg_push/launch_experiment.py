from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from pipeline.gcg_push import config as gcg_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print or submit config-driven GCG-push workflow commands.')
    parser.add_argument('--config', default=str(gcg_config.DEFAULT_CONFIG))
    parser.add_argument('--backend', choices=['local', 'slurm'], default='slurm')
    parser.add_argument('--stage', choices=['raw', 'artifacts', 'generation', 'evaluation', 'publish', 'analysis', 'all'], default='raw')
    parser.add_argument('--intervention', choices=['suffix_push', 'orth_shift'], default=None)
    parser.add_argument('--coeff', default=None)
    parser.add_argument('--submit', action=argparse.BooleanOptionalAction, default=False, help='Actually run/submit commands. Default prints them.')
    parser.add_argument('--max-parallel-tasks', type=int, default=None)
    parser.add_argument('--partition', default=None)
    parser.add_argument('--gpus-per-task', type=int, default=None)
    parser.add_argument('--judge-gpus-per-task', type=int, default=None)
    parser.add_argument('--cpus-per-task', type=int, default=None)
    parser.add_argument('--mem', default=None)
    parser.add_argument('--time', default=None)
    parser.add_argument('--model-cache-root', default=None)
    parser.add_argument('--logs-dir', default=None)
    return parser.parse_args(argv)


def q(value: object) -> str:
    return shlex.quote(str(value))


def selected_pairs(config: dict, args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    for intervention in gcg_config.selected_interventions(config, args.intervention):
        for coeff in gcg_config.selected_coefficients(config, intervention, args.coeff):
            yield intervention, coeff


def slurm_option(config: dict, args: argparse.Namespace, key: str, arg_value, default):
    return arg_value if arg_value is not None else gcg_config.slurm_value(config, key, default)


def run_or_print(command: str, submit: bool) -> None:
    print(command)
    if submit:
        subprocess.run(command, shell=True, check=True)


def array_spec(config: dict, args: argparse.Namespace) -> str:
    indices = ','.join(str(index) for index in gcg_config.indices(config))
    max_tasks = slurm_option(config, args, 'max_parallel_tasks', args.max_parallel_tasks, None)
    return f'{indices}%{max_tasks}' if max_tasks else indices


def slurm_wrap(config: dict, args: argparse.Namespace, name: str, array: str | None, inner: str, *, judge: bool = False) -> str:
    partition = slurm_option(config, args, 'partition', args.partition, 'general')
    gpus = slurm_option(config, args, 'judge_gpus_per_task' if judge else 'gpus_per_task', args.judge_gpus_per_task if judge else args.gpus_per_task, 4 if judge else 1)
    cpus = slurm_option(config, args, 'cpus_per_task', args.cpus_per_task, 1)
    mem = slurm_option(config, args, 'mem', args.mem, '50G')
    time = slurm_option(config, args, 'time', args.time, '48:00:00')
    logs_dir = gcg_config.repo_path(args.logs_dir or gcg_config.slurm_value(config, 'logs_dir', 'logs/slurm/gcg_push'))
    if args.submit:
        logs_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'mkdir -p {q(logs_dir)}'
    cache_root = args.model_cache_root or gcg_config.slurm_value(config, 'model_cache_root')
    if cache_root:
        prefix += f' && export HF_HOME={q(str(Path(cache_root) / ".cache" / "huggingface"))} TRANSFORMERS_CACHE=$HF_HOME/hub'
    wrapped = f'{prefix} && {inner}'
    command = [
        'sbatch', '--parsable', f'--partition={partition}', f'--job-name={name}', f'--gres=gpu:{gpus}',
        f'--cpus-per-task={cpus}', f'--mem={mem}', f'--time={time}',
        f'--output={logs_dir}/%x-%A_%a.out' if array else f'--output={logs_dir}/%x-%j.out',
    ]
    if array:
        command.append(f'--array={array}')
    command.extend(['--wrap', wrapped])
    return ' '.join(q(part) for part in command)


def raw_commands(config: dict, args: argparse.Namespace) -> list[str]:
    commands = []
    for intervention, coeff in selected_pairs(config, args):
        inner = ' '.join([
            'python -m pipeline gcg-push run',
            '--config', q(args.config),
            '--intervention', q(intervention),
            '--coeff', q(coeff),
            '--index', '"$SLURM_ARRAY_TASK_ID"' if args.backend == 'slurm' else '<index>',
        ])
        if args.backend == 'slurm':
            commands.append(slurm_wrap(config, args, f'gcg-{intervention}-{coeff}', array_spec(config, args), inner))
        else:
            for index in gcg_config.indices(config):
                commands.append(inner.replace('<index>', str(index)))
    return commands


def artifact_commands(config: dict, args: argparse.Namespace) -> list[str]:
    commands = []
    for intervention, coeff in selected_pairs(config, args):
        commands.append(' '.join([
            'python -m pipeline gcg-push create-datasets',
            '--config', q(args.config), '--intervention', q(intervention), '--coeff', q(coeff), '--check',
        ]))
    return commands


def generation_commands(config: dict, args: argparse.Namespace) -> list[str]:
    commands = []
    for intervention, coeff in selected_pairs(config, args):
        artifact = gcg_config.artifact_dir(config, intervention, coeff, 'transfer')
        num_chunks = gcg_config.workflow_chunks(config, intervention, 'generation_chunks')
        flag = '--suffix-push' if intervention == 'suffix_push' else '--orth-shift'
        commands.append(f'python -m pipeline artifacts prepare-generation --artifact-dir {q(artifact)} --num-output-chunks {num_chunks}')
        inner = ' '.join([
            'python -m pipeline completions generate', '--model-path', q(gcg_config.model_id(config)),
            '--gcg-push', '--coeff', q(coeff), flag, '--chunk-id', '"$SLURM_ARRAY_TASK_ID"' if args.backend == 'slurm' else '<chunk>', '--resume',
        ])
        if args.backend == 'slurm':
            commands.append(slurm_wrap(config, args, f'gen-{intervention}-{coeff}', f'0-{num_chunks - 1}%{slurm_option(config, args, "max_parallel_tasks", args.max_parallel_tasks, num_chunks)}', inner))
        else:
            for chunk in range(num_chunks):
                commands.append(inner.replace('<chunk>', str(chunk)))
    return commands


def evaluation_commands(config: dict, args: argparse.Namespace) -> list[str]:
    commands = []
    for intervention, coeff in selected_pairs(config, args):
        artifact = gcg_config.artifact_dir(config, intervention, coeff, 'transfer')
        eval_chunks = gcg_config.workflow_chunks(config, intervention, 'evaluation_chunks')
        flag = '--suffix-push' if intervention == 'suffix_push' else '--orth-shift'
        commands.append(f'python -m pipeline artifacts combine-completions --input-dir {q(artifact)} --input-subdir generation_chunks --output-subdir evaluation_chunks --num-output-chunks {eval_chunks} --check-stage generation')
        inner = ' '.join([
            'python -m pipeline completions evaluate', '--model-path', q(gcg_config.model_id(config)),
            '--gcg-push', '--coeff', q(coeff), flag, '--chunk-id', '"$SLURM_ARRAY_TASK_ID"' if args.backend == 'slurm' else '<chunk>',
            '--num-gpus', str(slurm_option(config, args, 'judge_gpus_per_task', args.judge_gpus_per_task, 4)),
        ])
        if args.backend == 'slurm':
            commands.append(slurm_wrap(config, args, f'eval-{intervention}-{coeff}', f'0-{eval_chunks - 1}%{slurm_option(config, args, "max_parallel_tasks", args.max_parallel_tasks, eval_chunks)}', inner, judge=True))
        else:
            for chunk in range(eval_chunks):
                commands.append(inner.replace('<chunk>', str(chunk)))
    return commands


def publish_commands(config: dict, args: argparse.Namespace) -> list[str]:
    commands = []
    for intervention, coeff in selected_pairs(config, args):
        artifact = gcg_config.artifact_dir(config, intervention, coeff, 'transfer')
        publish_chunks = gcg_config.artifact_chunks(config, intervention, 'transfer')
        source = f'gcg_push_results/{gcg_config.model_alias(config)}/{intervention}/coeff-{coeff}/transfer'
        commands.append(f'python -m pipeline artifacts combine-completions --input-dir {q(artifact)} --input-subdir evaluation_chunks --output-subdir chunks --num-output-chunks {publish_chunks} --check-stage evaluation --write-manifest --manifest-source {q(source)}')
    return commands


def analysis_commands(config: dict, args: argparse.Namespace) -> list[str]:
    base = ['python -m pipeline gcg-push analyze', '--config', q(args.config)]
    if args.intervention == 'suffix_push':
        base.append('--suffix-push')
    if args.intervention == 'orth_shift':
        base.append('--orth-shift')
    if args.coeff:
        base.extend(['--coeff', q(args.coeff)])
    return [' '.join(base)]


def main(argv=None) -> None:
    args = parse_args(argv)
    config = gcg_config.load_config(args.config)
    if args.submit and args.stage == 'all':
        raise SystemExit('Do not submit --stage all: run each stage after the previous Slurm jobs finish.')
    stages = ['raw', 'artifacts', 'generation', 'evaluation', 'publish', 'analysis'] if args.stage == 'all' else [args.stage]
    commands = []
    for stage in stages:
        if stage == 'raw':
            commands.extend(raw_commands(config, args))
        elif stage == 'artifacts':
            commands.extend(artifact_commands(config, args))
        elif stage == 'generation':
            commands.extend(generation_commands(config, args))
        elif stage == 'evaluation':
            commands.extend(evaluation_commands(config, args))
        elif stage == 'publish':
            commands.extend(publish_commands(config, args))
        elif stage == 'analysis':
            commands.extend(analysis_commands(config, args))
    for command in commands:
        run_or_print(command, args.submit)


if __name__ == '__main__':
    main()
