"""Strategy-path and adapter transport checks without loading model tensors."""
from contextlib import nullcontext, redirect_stdout
import importlib
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent import run
from agent.models import InferenceError, managed_model, prepare_model
from development_worlds import runner

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(path):
    path.mkdir()
    (path/'config.json').write_text('{}', encoding='utf-8')
    (path/'tokenizer.json').write_text('{}', encoding='utf-8')
    (path/'model.safetensors').write_bytes(b'checkpoint fixture, not model tensors')


class StrategyAgentLayoutTests(unittest.TestCase):
    def test_legacy_imports_alias_the_editable_strategy_modules(self):
        for name, actual in (('agent', 'run'), ('models', 'models'), ('tools', 'tools')):
            self.assertIs(importlib.import_module('shore_agent.'+name),
                          importlib.import_module('agent.'+actual))

    def test_cli_selects_primary_or_named_model_with_its_own_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint(root/'planner')
            checkpoint(root/'critic')
            adapter = root/'critic-lora'
            adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{"peft_type":"LORA","r":32}', encoding='utf-8')
            (adapter/'adapter_model.safetensors').write_bytes(b'adapter fixture')
            registry = root/'models.json'
            registry.write_text(json.dumps({'schema_version':'1.0.0', 'primary_model_id':'planner',
                'models':[
                    {'id':'planner','base_model':'Qwen3.8-27B','path':'planner','adaptation':'sft'},
                    {'id':'critic','base_model':'Qwen3.6-35B-A3B','path':'critic',
                     'adapter_path':'critic-lora','adaptation':'rl'},
                ]}), encoding='utf-8')
            for extra, base, path, selected_adapter in (
                ([], 'Qwen3.8-27B', root/'planner', None),
                (['--model-ref','critic'], 'Qwen3.6-35B-A3B', root/'critic', adapter),
            ):
                with self.subTest(extra=extra), \
                        patch('agent.models.managed_model', return_value=nullcontext(object())) as load, \
                        patch.object(run, 'Console'), patch.object(run, 'DiscoveryAgent') as discovery, \
                        redirect_stdout(StringIO()):
                    discovery.return_value.run.return_value = {'has_saved_pair':True}
                    self.assertEqual(run.main(['--models-file', str(registry), '--output-dir', str(root/'out'), *extra]), 0)
                self.assertEqual(load.call_args.args[0], base)
                self.assertEqual(Path(load.call_args.kwargs['model_path']), path.resolve())
                if selected_adapter:
                    self.assertEqual(Path(load.call_args.kwargs['adapter_path']), selected_adapter.resolve())
                else:
                    self.assertIsNone(load.call_args.kwargs['adapter_path'])

    def test_lora_plan_requests_adapter_instead_of_unadapted_base(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'adapter_config.json').write_text(json.dumps({
                'peft_type':'LORA', 'r':16, 'rank_pattern':{'layer':96}}), encoding='utf-8')
            (root/'adapter_model.safetensors').write_bytes(b'not model tensors')
            plan = prepare_model('27b', inspect=False, adapter_path=root)
        command = plan['command']
        self.assertEqual(command[command.index('--served-model-name')+1], 'Qwen3.8-27B-base')
        self.assertIn('--enable-lora', command)
        self.assertEqual(command[command.index('--max-lora-rank')+1], '128')
        module = json.loads(command[command.index('--lora-modules')+1])
        self.assertEqual(module['name'], plan['model']['served_model_name'])
        self.assertEqual(module['path'], str(root.resolve()))
        self.assertEqual(plan['adapter']['rank'], 96)

    def test_adapter_path_cannot_be_silently_ignored_by_existing_service(self):
        with self.assertRaisesRegex(InferenceError, 'adapter path'):
            with managed_model('27b', base_url='http://127.0.0.1:8000/v1', adapter_path='/adapter'):
                self.fail('An unverified adapter service must not be used.')

    def test_invalid_adapter_rank_fails_before_service_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'adapter_model.safetensors').write_bytes(b'weights')
            for rank in (True, 0, 513):
                with self.subTest(rank=rank):
                    (root/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':rank}), encoding='utf-8')
                    with self.assertRaises(InferenceError):
                        prepare_model('27b', inspect=False, adapter_path=root)

    def test_isolated_worker_uses_official_sdk_with_strategy_outside_project(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = Path(folder)/'agent/tool_worker.py'
            worker.parent.mkdir()
            shutil.copyfile(ROOT/'agent/tool_worker.py', worker)
            environment = dict(os.environ, SHORE_SDK_ROOT=str(ROOT), PYTHONIOENCODING='utf-8')
            done = subprocess.run([sys.executable, '-I', str(worker)],
                input=json.dumps({'source':'', 'records':[]}), capture_output=True, text=True,
                encoding='utf-8', env=environment, cwd=folder, timeout=30)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn('permitted probe trajectory', json.loads(done.stdout)['error'])

    def test_development_strategy_options_must_be_a_complete_pair(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            for arguments in (['--agent-dir',folder], ['--models-file',folder],
                              ['--agent-dir',folder,'--models-file',folder,'--agent-command','python','x.py']):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                    runner.main(['--output',str(Path(folder)/'result'), *arguments])
                self.assertEqual(error.exception.code, 2)

    def test_development_passes_complete_strategy_and_registry_to_fixed_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            strategy = root/'agent'
            strategy.mkdir()
            (strategy/'__init__.py').write_text('', encoding='utf-8')
            (strategy/'run.py').write_text('def main(): pass\n', encoding='utf-8')
            (strategy/'strategy.json').write_text(json.dumps({'schema_version':'1.0.0',
                'team_id':'team', 'submission_id':'v1', 'version':'1'}), encoding='utf-8')
            checkpoint(root/'weights')
            registry = root/'models.json'
            registry.write_text(json.dumps({'schema_version':'1.0.0','primary_model_id':'planner',
                'models':[{'id':'planner','base_model':'Qwen3.8-27B','path':'weights'}]}), encoding='utf-8')
            result = {'world':'1','has_valid_submission':True,'score':{'total':0.}}
            with patch.object(runner, 'run_episode', return_value=result) as episode, redirect_stdout(StringIO()):
                status = runner.main(['--worlds','1','--output',str(root/'out'),
                    '--agent-dir',str(strategy),'--models-file',str(registry),'--max-turns','2'])
            self.assertEqual(status, 0)
            command = episode.call_args.args[2]
            self.assertEqual(command[:3], [sys.executable,'-I',str(ROOT/'scripts/run_strategy.py')])
            self.assertEqual(Path(command[command.index('--agent-dir')+1]), strategy.resolve())
            self.assertEqual(Path(command[command.index('--models-file')+1]), registry.resolve())
            self.assertEqual(command[command.index('--max-turns')+1], '2')


if __name__ == '__main__':
    unittest.main()
