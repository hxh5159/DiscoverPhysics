"""Offline name, metadata, transport and launch checks; no real model calls."""
import copy
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rift.model_registry import catalog, model_spec, canonical_base_model
from rift.submission import validate_manifest
from shore_agent.models import (ModelClient, InferenceError, prepare_model, inspect_checkpoint,
                                resolve_model_path, _runtime_environment, verify_capacity, stop_server, managed_model,
                                _model_python)

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(root, precision='BF16'):
    root.mkdir(parents=True,exist_ok=True)
    config = {'torch_dtype':'bfloat16'}
    if precision == 'FP8': config['quantization_config'] = {'quant_method':'fp8'}
    (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
    (root/'tokenizer.json').write_text('{}',encoding='utf-8')
    header = json.dumps({'weight':{'dtype':'F8_E4M3' if precision=='FP8' else 'BF16',
        'shape':[8], 'data_offsets':[0,8 if precision=='FP8' else 16]}}).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+bytes(8 if precision=='FP8' else 16))


class Response(BytesIO):
    def __init__(self,value):super().__init__(json.dumps(value).encode())


class Opener:
    def __init__(self,values): self.values=list(values);self.requests=[]
    def open(self, request, timeout):
        self.requests.append((request,timeout))
        return Response(self.values.pop(0))


class ModelEntryTests(unittest.TestCase):
    def test_five_variants_have_distinct_names_and_manifest_identity(self):
        self.assertEqual(len(catalog()['models']),5)
        original = json.loads((ROOT/'examples/submission.json').read_text(encoding='utf-8'))
        for item in catalog()['models']:
            for alias in (item['id'],item['directory'],*item['directory_aliases']):
                self.assertEqual(model_spec(alias)['id'],item['id'])
                value=copy.deepcopy(original)
                value['models'][0]['base_model']=alias
                value['models'][0]['path']='/models/'+item['directory']
                self.assertEqual(validate_manifest(value)['models'][0]['base_model'],item['base_model'])
        self.assertNotEqual(canonical_base_model('122b'),canonical_base_model('122b-fp8'))
        with self.assertRaises(ValueError):canonical_base_model('unlisted-closed-model')

    def test_header_precision_alias_and_new_cu128_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/'Qwen3.6-35B-A3B-FP8'
            checkpoint(path,'FP8')
            plan=prepare_model('35b-fp8',model_root=root,python_executable=sys.executable)
            self.assertEqual(Path(plan['model_path']),path.resolve())
            self.assertEqual(plan['checkpoint']['weight_payload_bytes'],8)
            self.assertEqual(plan['model']['precision'],'FP8')
            self.assertIn('--language-model-only',plan['command'])
            self.assertNotIn('--quantization',plan['command'])
            self.assertFalse(plan['loaded_or_generated'])
            with self.assertRaises(InferenceError):inspect_checkpoint(path,'BF16')
            checkpoint(root/'bad')
            with self.assertRaises(InferenceError):inspect_checkpoint(root/'bad','FP8')

    def test_header_must_be_complete_and_contained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint(root)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../elsewhere.safetensors'}}))
            with self.assertRaises(InferenceError):inspect_checkpoint(root,'BF16')
            (root/'model.safetensors.index.json').unlink()
            (root/'model.safetensors').write_bytes(b'bad')
            with self.assertRaises(InferenceError):inspect_checkpoint(root,'BF16')

    def test_loopback_only_no_proxy_or_external_service_fallback(self):
        for url in ('https://external.example/v1','http://example.org:8000',
                    'http://u:p@127.0.0.1:8000/v1','http://127.0.0.1:8000/v1?x=1',
                    'http://127.0.0.1:8000/path'):
            with self.subTest(url=url),self.assertRaises(InferenceError):ModelClient(url)
        ModelClient('http://[::1]:8000/v1')

    def test_discovery_chat_and_thinking_only(self):
        client=ModelClient('http://127.0.0.1:8000')
        client._opener=Opener([{'data':[{'id':'Qwen-3.6-35B-A3B-FP8'}]},
            {'choices':[{'message':{'content':'<think>private reasoning</think>{"action":"observe"}'}}]},
            {'choices':[{'message':{'content':None,'reasoning_content':'still thinking'}}]}])
        ready=client.ready()
        self.assertEqual(ready['served_model_name'],'Qwen-3.6-35B-A3B-FP8')
        self.assertEqual(client.complete([{'role':'user','content':'hello'}],timeout=2),'{"action":"observe"}')
        self.assertEqual(client._opener.requests[-1][1],2)
        self.assertEqual(client.complete([{'role':'user','content':'hello'}]),'')
        self.assertIsNotNone(client.last_response)

    def test_late_response_not_accepted_and_ambiguous_model_rejected(self):
        client=ModelClient('http://127.0.0.1:8000','27b')
        client.model='Qwen3.8-27B'  # This isolated timing fixture bypasses discovery.
        client._opener=Opener([{'choices':[{'message':{'content':'late'}}]}])
        with patch('shore_agent.models.time.monotonic',side_effect=[0.,3.]):
            with self.assertRaises(InferenceError):client.complete([{'role':'user','content':'x'}],timeout=2.)
        client=ModelClient('http://127.0.0.1:8000')
        client._opener=Opener([{'data':[{'id':'Qwen3.8-27B'},{'id':'Qwen3.6-35B-A3B'}]}])
        with self.assertRaises(InferenceError):client.ready()

    def test_runtime_caches_writable_manifest_restores_toolchain(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            executable=root/'.venv-vllm-cu128/bin/python'
            executable.parent.mkdir(parents=True)
            cuda=root/'toolchain'
            (cuda/'lib64').mkdir(parents=True)
            (root/'cuda128-build-manifest.json').write_text(json.dumps({'target':'h200',
                'cuda_home':str(cuda),'cc':'/cc','cxx':'/cxx'}))
            with patch.dict(os.environ,{'LD_LIBRARY_PATH':os.pathsep.join(('/old/stubs','/good'))}):
                env=_runtime_environment(str(executable),root/'output-cache')
            self.assertEqual(env['CUDA_HOME'],str(cuda))
            self.assertEqual(env['CC'],'/cc')
            self.assertEqual(env['TRITON_PTXAS_PATH'],str(cuda/'bin/ptxas'))
            self.assertEqual(env['FLASHINFER_NO_DOWNLOAD'],'1')
            self.assertNotIn('/old/stubs',env['LD_LIBRARY_PATH'])
            for key in ('HF_HOME','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','FLASHINFER_WORKSPACE_BASE',
                        'CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR'):
                self.assertTrue(Path(env[key]).is_relative_to(root/'output-cache'))
                self.assertTrue(Path(env[key]).is_dir())

    def test_venv_interpreter_path_never_dereferenced_and_manifest_stays_at_environment_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            executable=root/'.venv-vllm-cu128/bin/python'
            executable.parent.mkdir(parents=True)
            # The assertion against resolve also works on Windows accounts
            # without the symlink privilege, and detects the original defect.
            try: executable.symlink_to(Path(sys.executable))
            except OSError: executable.write_text('interpreter placeholder',encoding='utf-8')
            cuda=root/'toolchain'
            (root/'cuda128-build-manifest.json').write_text(json.dumps({
                'target':'h200','cuda_home':str(cuda),'cc':'/cc','cxx':'/cxx'}))
            with patch('pathlib.Path.resolve',side_effect=AssertionError('Never dereference an interpreter symlink')):
                selected=_model_python(str(executable))
            self.assertEqual(selected,str(executable.absolute()))
            environment=_runtime_environment(selected,root/'cache')
            self.assertEqual(environment['CUDA_HOME'],str(cuda))
            self.assertEqual(environment['CXX'],'/cxx')

    def test_capacity_rejects_impossible_bf16_without_implicit_quantization(self):
        plan=prepare_model('122b',inspect=False)
        plan['checkpoint']={'weight_payload_bytes':244_000_000_000}
        completed=subprocess.CompletedProcess([],0,json.dumps({'vllm':'0.24.0+cu128','cuda':'12.8',
            'devices':[{'free_bytes':48_000_000_000,'total_bytes':48_000_000_000}]*4}), '')
        with patch('shore_agent.models.subprocess.run',return_value=completed):
            with self.assertRaisesRegex(InferenceError,'exceed'):verify_capacity(plan)
            plan['checkpoint']['weight_payload_bytes']=120_000_000_000
            verify_capacity(plan)
        self.assertTrue(plan['gpu_capacity_checked'])
        self.assertNotIn('--quantization',plan['command'])

    def test_plan_only_needs_no_model_gpu_or_inference_package(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/serve_model.py'),'--model','122b-fp8',
            '--plan-only','--model-root','/nonexistent/models'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIsNone(json.loads(result.stdout)['checkpoint'])
        self.assertFalse(json.loads(result.stdout)['loaded_or_generated'])

    def test_exited_server_parent_does_not_leave_managed_worker_group(self):
        class Process:
            pid=12345
            def poll(self):return 1
            def wait(self,timeout):return 1
        with patch('shore_agent.models.os.name','posix'),patch('shore_agent.models.os.killpg',create=True) as kill, \
                patch('shore_agent.models.signal.SIGKILL',9,create=True):
            stop_server(Process())
        self.assertEqual(kill.call_count,2)

    def test_existing_model_service_is_checked_without_starting_another_process(self):
        with patch('shore_agent.models.ModelClient.ready',return_value={'served_model_name':'Qwen3.8-27B'}) as ready, \
                patch('shore_agent.models.subprocess.Popen') as spawn:
            with managed_model('27b',base_url='http://127.0.0.1:8000/v1') as client:
                self.assertIsInstance(client,ModelClient)
            ready.assert_called_once()
            spawn.assert_not_called()

    def test_managed_model_cleans_up_when_the_agent_body_fails(self):
        class Process:
            pid=12345
            def poll(self):return None
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint(root/'weights')
            with patch('shore_agent.models.verify_capacity'), \
                    patch('shore_agent.models.subprocess.Popen',return_value=Process()) as spawn, \
                    patch('shore_agent.models.ModelClient.ready',return_value={}), \
                    patch('shore_agent.models.stop_server') as stop:
                with self.assertRaisesRegex(RuntimeError,'agent failed'):
                    with managed_model('27b',model_path=root/'weights',log_dir=root/'output',
                                       python_executable=sys.executable) as client:
                        self.assertIsInstance(client,ModelClient)
                        raise RuntimeError('agent failed')
                stop.assert_called_once()
                self.assertEqual(spawn.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
                self.assertTrue(Path(spawn.call_args.kwargs['env']['HF_HOME']).is_relative_to(root/'output'))


if __name__=='__main__':unittest.main()
