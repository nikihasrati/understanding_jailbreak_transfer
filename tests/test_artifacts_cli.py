import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from pipeline import artifacts
from pipeline import cli
from pipeline.cli import translate_args
from pipeline.config import Config
from pipeline.evaluation.evaluate_completions import evaluate_generations, parse_arguments as parse_evaluation_legacy
from pipeline.evaluation.evaluate_completions import parse_args as parse_evaluation_args
from pipeline.generation.generate_completions import generate_for_path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ArtifactHelperTests(unittest.TestCase):
    def test_chunk_paths_sort_canonical_and_legacy_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['chunk_00010.json', 'chunk_00002.json', 'chunk_00001.json']:
                (root / name).write_text('[]\n')
            self.assertEqual(
                [path.name for path in artifacts.json_chunk_paths(root)],
                ['chunk_00001.json', 'chunk_00002.json', 'chunk_00010.json'],
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['10_records.json', '2_records.json', '1_records.json']:
                (root / name).write_text('[]\n')
            self.assertEqual(
                [path.name for path in artifacts.json_chunk_paths(root)],
                ['1_records.json', '2_records.json', '10_records.json'],
            )

    def test_manifest_verification_supports_chunks_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{'id': 1}, {'id': 2}, {'id': 3}]
            chunks = artifacts.write_json_chunks(records, root / 'chunks', 2, artifact_dir=root)
            manifest = root / 'manifest.json'
            manifest.write_text(
                json.dumps(
                    {
                        'artifact_format': 'chunked_json',
                        'total_records': 3,
                        'total_chunks': 2,
                        'chunks': chunks,
                    },
                    indent=2,
                )
                + '\n'
            )
            artifacts.verify_manifest(manifest)

            raw = root / 'raw.json'
            raw.write_text('{"ok": true}\n')
            file_manifest = root / 'files_manifest.json'
            file_manifest.write_text(
                json.dumps(
                    {
                        'artifact_format': 'file_manifest',
                        'files': [
                            {
                                'path': raw.name,
                                'bytes': raw.stat().st_size,
                                'sha256': artifacts.sha256_file(raw),
                            }
                        ],
                    },
                    indent=2,
                )
                + '\n'
            )
            artifacts.verify_manifest(file_manifest)

    def test_workflow_chunk_path_uses_stage_specific_temporary_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation_chunk = root / 'generation_chunks' / 'chunk_00002.json'
            evaluation_chunk = root / 'evaluation_chunks' / 'chunk_00003.json'
            generation_chunk.parent.mkdir()
            evaluation_chunk.parent.mkdir()
            generation_chunk.write_text('[]\n')
            evaluation_chunk.write_text('[]\n')

            self.assertEqual(
                artifacts.workflow_chunk_path(root / 'combined.json', 2, 'generation'),
                str(generation_chunk),
            )
            self.assertEqual(
                artifacts.workflow_chunk_path(root / 'combined.json', 3, 'evaluation'),
                str(evaluation_chunk),
            )

    def test_workflow_chunk_path_rejects_wrong_stage_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / 'chunks' / 'chunk_00000.json'
            canonical.parent.mkdir()
            canonical.write_text('[]\n')
            with self.assertRaisesRegex(FileNotFoundError, 'generation writes to temporary generation_chunks'):
                artifacts.workflow_chunk_path(root / 'combined.json', 0, 'generation')

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation_chunk = root / 'generation_chunks' / 'chunk_00000.json'
            generation_chunk.parent.mkdir()
            generation_chunk.write_text('[]\n')
            with self.assertRaisesRegex(FileNotFoundError, 'evaluation reads temporary evaluation_chunks'):
                artifacts.workflow_chunk_path(root / 'combined.json', 0, 'evaluation')

    def test_generation_resume_preserves_completed_rows(self):
        class FakeModel:
            def __init__(self):
                self.inputs = []

            def generate_completions(self, texts, max_new_tokens):
                self.inputs.append((texts, max_new_tokens))
                return [f'generated:{text}' for text in texts]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.json'
            path.write_text(json.dumps([
                {'prompt': 'p0', 'jailbreak': 'j0', 'response': 'existing'},
                {'prompt': 'p1', 'jailbreak': 'j1'},
            ]))
            model = FakeModel()

            generate_for_path(model, str(path), resume=True, batch_size=1)

            self.assertEqual(model.inputs, [(['j1'], 200)])
            self.assertEqual(
                json.loads(path.read_text()),
                [
                    {'prompt': 'p0', 'jailbreak': 'j0', 'response': 'existing'},
                    {'prompt': 'p1', 'jailbreak': 'j1', 'response': 'generated:j1'},
                ],
            )

    def test_evaluation_resume_preserves_completed_rows(self):
        class FakeJudge:
            def __init__(self):
                self.calls = []

            def classify_responses(self, prompts, responses):
                self.calls.append((prompts, responses))
                return [False for _ in prompts]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.json'
            path.write_text(json.dumps([
                {'prompt': 'p0', 'response': 'r0', 'jailbroken': True},
                {'prompt': 'p1', 'response': 'r1'},
            ]))
            judge = FakeJudge()

            evaluate_generations(judge, str(path), batch_size=1)

            self.assertEqual(judge.calls, [(['p1'], ['r1'])])
            records = json.loads(path.read_text())
            self.assertIs(records[0]['jailbroken'], True)
            self.assertIs(records[1]['jailbroken'], False)
            self.assertEqual(
                records,
                [
                    {'prompt': 'p0', 'response': 'r0', 'jailbroken': True},
                    {'prompt': 'p1', 'response': 'r1', 'jailbroken': False},
                ],
            )

    def test_config_alias_paths_and_lazy_num_prompts(self):
        with mock.patch('pipeline.config.pd.read_json') as read_json:
            cfg = Config('/models/meta-llama/Llama-3.2-1B-Instruct')
            self.assertEqual(cfg.model_alias, 'llama-3.2-1b-instruct')
            self.assertTrue(cfg.prompts_path().endswith('data/processed/jailbreakbench/prompts/chunks/chunk_00000.json'))
            self.assertTrue(
                cfg.suffixes_path().endswith(
                    'data/processed/jailbreakbench/suffixes/llama-3.2-1b-instruct/chunks/chunk_00000.json'
                )
            )
            self.assertTrue(
                cfg.single_seed_transfer_path().endswith(
                    'data/intra_model_transfer/single_seed/llama-3.2-1b-instruct/combined.json'
                )
            )
            self.assertTrue(
                cfg.multi_seed_no_transfer_path().endswith(
                    'data/intra_model_transfer/multi_seed/llama-3.2-1b-instruct/no_transfer/combined.json'
                )
            )
            self.assertTrue(
                cfg.multi_seed_transfer_path().endswith(
                    'data/intra_model_transfer/multi_seed/llama-3.2-1b-instruct/transfer/all/combined.json'
                )
            )
            self.assertEqual(cfg.single_seed_cross_prompt_transfer_generations_path(), cfg.single_seed_transfer_path())
            self.assertEqual(cfg.multi_seed_generations_no_transfer_path(), cfg.multi_seed_no_transfer_path())
            self.assertEqual(cfg.multi_seed_generations_transfer_path(), cfg.multi_seed_transfer_path())
            self.assertTrue(cfg.no_suffix_generations_path().endswith('data/no_suffix_generations/llama-3.2-1b-instruct/combined.json'))
            self.assertTrue(cfg.prompt_rephrasings_path().endswith('data/prompt_rephrasings/llama-3.2-1b-instruct/combined.json'))
            read_json.assert_not_called()

            with tempfile.TemporaryDirectory() as tmp:
                cfg.PROCESSED_DATASET_DIR = Path(tmp)
                self.assertEqual(cfg.num_prompts, 100)
            read_json.assert_not_called()

    def test_grouped_cli_verifies_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = artifacts.write_json_chunks([{'id': 1}], root / 'chunks', 1, artifact_dir=root)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'chunks': chunks}, indent=2) + '\n')
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pipeline',
                    'artifacts',
                    'verify-manifest',
                    '--manifest',
                    str(manifest),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('OK:', result.stdout)

    def test_cli_alias_translation_keeps_canonical_and_legacy_flags(self):
        aliases = {'--model-path': '--model_path', '--chunk-id': '--chunk_id'}
        self.assertEqual(
            translate_args(['--model-path', 'm', '--chunk-id', '3', '--legacy'], aliases),
            ['--model_path', 'm', '--chunk_id', '3', '--legacy'],
        )

    def test_cli_dispatch_calls_module_main_with_translated_args(self):
        module = types.ModuleType('fake_pipeline_command')
        calls = []
        module.main = calls.append
        command = cli.Command('fake_pipeline_command', aliases={'--canonical': '--legacy'})

        with mock.patch.dict(sys.modules, {'fake_pipeline_command': module}):
            with mock.patch.dict(cli.COMMANDS, {('fake', 'run'): command}):
                cli.main(['fake', 'run', '--canonical', 'value'])

        self.assertEqual(calls, [['--legacy', 'value']])

    def test_canonical_and_legacy_evaluation_args_parse_equivalently(self):
        canonical = parse_evaluation_args([
            '--model-path', 'model',
            '--num-gpus', '2',
            '--multi-seed',
            '--chunk-id', '4',
            '--gcg-push',
            '--coeff', '0.1',
            '--suffix-push',
        ])
        legacy = parse_evaluation_legacy([
            '--model_path', 'model',
            '--num_gpus', '2',
            '--multi_seed',
            '--chunk_id', '4',
            '--gcg_push',
            '--coeff', '0.1',
            '--suffix_push',
        ])
        self.assertEqual(vars(canonical), vars(legacy))

    def test_normalize_labels_supports_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'input.json'
            dst = Path(tmp) / 'output.json'
            src.write_text(
                json.dumps(
                    [
                        {'jailbroken': 'true'},
                        {'jailbroken': '0'},
                        {'jailbroken': None},
                    ]
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pipeline',
                    'completions',
                    'normalize-labels',
                    '--path',
                    str(src),
                    '--output',
                    str(dst),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(dst.read_text()),
                [{'jailbroken': True}, {'jailbroken': False}, {'jailbroken': None}],
            )


if __name__ == '__main__':
    unittest.main()
