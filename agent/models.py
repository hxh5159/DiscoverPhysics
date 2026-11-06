"""Local, offline model serving and a small HTTP chat client.

No API/provider fallback, model download or package installation is performed.
The managed server starts before mission(), independently of exploration time.
Official vLLM reference: https://docs.vllm.ai/en/v0.24.0/cli/serve/
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler

from rift.model_registry import catalog, model_spec, canonical_base_model

DEFAULT_MODEL_PYTHON = '/opt/shore-h200-cu128/.venv-vllm-cu128/bin/python'
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def action_output_schema():
    """Constrain transport syntax only; no scientific answers or defaults.

    Console and candidate validators still check action semantics. This schema
    deliberately does not require a new law/claims when a saved pair is reused.
    """
    names = ['observe', 'launch', 'archive', 'view_record', 'check', 'fit', 'draft', 'submit']
    properties = {
        'action': {'type': 'string', 'enum': names},
        'law': {'type': 'string'},
        'law_ref': {'type': 'string', 'enum': ['saved']},
        'claims': {'type': 'object', 'properties': {
            'dynamics_order': {'type': 'integer', 'enum': [1, 2]},
            'hidden_matter': {'type': 'boolean'},
            'time_dependence': {'type': 'boolean'}},
            'required': ['dynamics_order', 'hidden_matter', 'time_dependence'],
            'additionalProperties': False},
        'times': {'type': 'array', 'items': {'type': 'number'}},
        't_launch': {'type': 'number'},
        'readout': {'type': 'array', 'items': {'type': 'number'}},
        'source_properties': {'type': 'object', 'additionalProperties': {'type': 'number'}},
        'probes': {'type': 'array', 'items': {'type': 'object'}},
        'index': {'type': 'integer'}, 'start': {'type': 'integer'},
        'count': {'type': 'integer'}, 'max_nfev': {'type': 'integer'},
    }
    single = {'type': 'object', 'properties': properties,
              'required': ['action'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        **properties, 'action': {'type': 'string', 'enum': names + ['batch']},
        'actions': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': single}},
        'required': ['action'], 'additionalProperties': False}


class InferenceError(RuntimeError):
    """Configuration, inference transport, response or startup failure."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise InferenceError(name+' must be finite and positive.')
    return float(value)


def _base_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1', 'localhost')
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise InferenceError('Use an explicit local HTTP inference URL without credentials or query parameters.')
    path = parsed.path.rstrip('/')
    if path not in ('', '/v1'):
        raise InferenceError('Inference URL must end in /v1 or have no path.')
    return urlunsplit((parsed.scheme, parsed.netloc, '/v1', '', ''))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ModelClient:
    """Use the specified local server only; served identity is self-reported."""
    capability_label = 'self_hosted_local_model_transport_unverified_weight_provenance'
    def __init__(self, base_url, model=None, timeout=300., *, structured_output=None):
        self.base_url = _base_url(base_url)
        if model is not None:
            try: model_spec(model)
            except ValueError:
                try: canonical_base_model(model)
                except ValueError: raise InferenceError('Requested model is not a registered whitelist model.') from None
        self._requested_model = model
        self.model = None
        self.timeout = _positive(timeout, 'timeout')
        mode = os.environ.get('SHORE_STRUCTURED_OUTPUT', 'off') if structured_output is None else structured_output
        if mode not in ('off', 'json_schema'):
            raise InferenceError('SHORE_STRUCTURED_OUTPUT must be off or json_schema.')
        self.structured_output = mode
        self.last_response = None
        self.last_request_report = None
        self.last_generation_report = None
        self.generation_reports = []
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, suffix, body=None, timeout=None):
        seconds = self.timeout if timeout is None else min(self.timeout, _positive(timeout, 'timeout'))
        data = None if body is None else json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
        request = Request(self.base_url+suffix, data=data,
            headers={'Content-Type':'application/json', 'Accept':'application/json'},
            method='GET' if data is None else 'POST')
        started = time.monotonic()
        self.last_request_report = {'allowed_seconds': seconds}
        try:
            with self._opener.open(request, timeout=seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES+1)
        except HTTPError as exc:
            raise InferenceError(f'Local inference returned HTTP {exc.code}. No fallback was attempted.') from None
        except (URLError, TimeoutError, OSError):
            raise InferenceError('Local inference connection failed or timed out. No fallback was attempted.') from None
        finally:
            self.last_request_report['elapsed_seconds'] = time.monotonic()-started
        # urllib socket timeouts bound an idle operation, not the whole request.
        # A surrounding episode watchdog stops the agent; never accept a late action.
        if self.last_request_report['elapsed_seconds'] >= seconds:
            raise InferenceError('Local inference response arrived after the allowed request window.')
        if len(raw) > MAX_RESPONSE_BYTES:
            raise InferenceError('Local inference response exceeds the byte limit.')
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise InferenceError('Local inference response is not JSON.') from None
        if not isinstance(result, dict):
            raise InferenceError('Local inference response must be a JSON object.')
        return result

    def ready(self, timeout=None):
        result = self._request('/models', timeout=timeout)
        data = result.get('data')
        if not isinstance(data, list):
            raise InferenceError('Local inference did not return a model list.')
        candidates = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str):
                continue
            supplied = item['id']
            try:
                try: canonical = model_spec(supplied)['base_model']
                except ValueError: canonical = canonical_base_model(supplied)
            except ValueError:
                continue
            candidates.append((supplied, canonical))
        if self._requested_model:
            try: wanted = model_spec(self._requested_model)['base_model']
            except ValueError:
                try: wanted = canonical_base_model(self._requested_model)
                except ValueError: raise InferenceError('Requested model is not a registered whitelist model.') from None
            candidates = [item for item in candidates if item[0] == self._requested_model or item[1] == wanted]
        if len(candidates) != 1:
            raise InferenceError('Expected exactly one matching registered model in the local service.')
        self.model = candidates[0][0]
        return {'served_model_name':self.model, 'identity':'server-reported, not proof of weight provenance'}

    def complete(self, messages, *, max_tokens=4096, timeout=None):
        if not self.model:
            raise InferenceError('Call ready() before generation.')
        if type(max_tokens) is not int or max_tokens <= 0:
            raise InferenceError('max_tokens must be a positive integer.')
        if not isinstance(messages, list) or not messages:
            raise InferenceError('messages must be a nonempty chat message list.')
        body = {'model':self.model, 'messages':messages,
            'max_tokens':max_tokens, 'temperature':.2, 'stream':False}
        if self.structured_output == 'json_schema':
            # vLLM's native field, opt-in for services supporting it. An HTTP
            # rejection is surfaced, never retried as an uncounted model call.
            body['structured_outputs'] = {'json': action_output_schema()}
        report = {'status': 'started', 'max_tokens': max_tokens,
                  'structured_output': self.structured_output, 'elapsed_seconds': None}
        self.last_response = self.last_request_report = None
        self.last_generation_report = report
        self.generation_reports.append(report)
        try:
            result = self._request('/chat/completions', body, timeout=timeout)
            self.last_response = result
            try:
                choice = result['choices'][0]
                message = choice['message']
                content = message.get('content')
            except (KeyError, IndexError, TypeError, AttributeError):
                raise InferenceError('Local inference response has no chat message.') from None
            reason = choice.get('finish_reason')
            report['finish_reason'] = reason[:64] if isinstance(reason, str) else None
            usage = result.get('usage')
            report['usage'] = {key: usage[key] for key in
                ('prompt_tokens', 'completion_tokens', 'total_tokens')
                if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
            report['output_characters'] = len(content) if isinstance(content, str) else 0
            if content is None and isinstance(message.get('reasoning_content', message.get('reasoning')), str):
                report['status'] = 'thinking_only'
                return ''  # Thinking-only generation has no final action to execute.
            if not isinstance(content, str):
                raise InferenceError('Local inference final content must be text.')
            report['status'] = 'received'
            return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        except Exception as exc:
            report['status'] = 'error'
            report['error_type'] = type(exc).__name__
            raise
        finally:
            if self.last_request_report is not None:
                report['elapsed_seconds'] = self.last_request_report.get('elapsed_seconds')
                report['allowed_seconds'] = self.last_request_report.get('allowed_seconds')


def resolve_model_path(key, *, model_root=None, model_path=None):
    spec = model_spec(key)
    if model_path is not None:
        return Path(model_path).expanduser().resolve()
    default_root = '/models' if Path('/models').is_dir() else catalog()['model_root']
    root = Path(model_root or os.environ.get('SHORE_MODEL_ROOT') or default_root).expanduser()
    paths = [root/name for name in (spec['directory'], *spec['directory_aliases'])]
    present = [path for path in paths if path.is_dir()]
    return (present[0] if present else paths[0]).resolve()


def inspect_checkpoint(path, expected_precision):
    """Read config and safetensors headers only, not tensor payloads."""
    root = Path(path).resolve()
    try:
        with (root/'config.json').open(encoding='utf-8') as stream:
            cfg = json.load(stream)
    except (OSError, ValueError):
        raise InferenceError('Complete local model directory with readable config.json is required.') from None
    if not isinstance(cfg, dict):
        raise InferenceError('Model config.json must be an object.')
    if not any((root/name).is_file() for name in ('tokenizer.json','tokenizer.model','vocab.json')):
        raise InferenceError('Local tokenizer assets are missing; no download is attempted.')
    index = root/'model.safetensors.index.json'
    if index.is_file():
        try:
            names = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
        except (OSError, ValueError, TypeError, KeyError):
            raise InferenceError('Invalid safetensors index.') from None
    else:
        names = {p.name for p in root.glob('*.safetensors')}
    if not names:
        raise InferenceError('No local safetensors checkpoint shards were found.')
    dtype_bytes, total_bytes = {}, 0
    for name in sorted(names):
        if not isinstance(name, str) or not name.endswith('.safetensors'):
            raise InferenceError('Checkpoint index must refer to safetensors files.')
        shard = (root/name).resolve()
        if not shard.is_relative_to(root) or not shard.is_file():
            raise InferenceError('Checkpoint shard is missing or resolves outside the model directory.')
        try:
            with shard.open('rb') as stream:
                prefix = stream.read(8)
                if len(prefix) != 8: raise ValueError()
                length = struct.unpack('<Q', prefix)[0]
                if not 2 <= length <= 64*1024*1024: raise ValueError()
                header = json.loads(stream.read(length))
            if not isinstance(header, dict): raise ValueError()
            end = 0
            for tensor, detail in header.items():
                if tensor == '__metadata__': continue
                offsets = detail['data_offsets']
                start, stop = offsets
                if type(start) is not int or type(stop) is not int or not 0 <= start <= stop: raise ValueError()
                dtype = detail['dtype']
                dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + stop-start
                end = max(end, stop)
            if shard.stat().st_size < 8+length+end: raise ValueError()
            total_bytes += end
        except (OSError, ValueError, TypeError, KeyError, struct.error):
            raise InferenceError('Invalid or truncated local safetensors header/shard: '+name) from None
    fp8_bytes = sum(size for dtype,size in dtype_bytes.items() if dtype.startswith('F8'))
    fp8 = fp8_bytes > total_bytes*.5
    if total_bytes <= 0: raise InferenceError('Checkpoint has no tensor payload.')
    text_config = cfg.get('text_config') if isinstance(cfg.get('text_config'),dict) else {}
    quant = cfg.get('quantization_config') or text_config.get('quantization_config')
    if expected_precision == 'FP8' and (not fp8 or not isinstance(quant,dict) or not quant):
        raise InferenceError('Selected FP8 variant needs FP8-stored tensors and its quantization configuration; a name alone is insufficient.')
    if expected_precision == 'BF16' and (fp8 or (isinstance(quant,dict) and quant)):
        raise InferenceError('Selected unquantized variant is quantized; select the corresponding FP8 preset instead.')
    return {'shards':len(names), 'weight_payload_bytes':total_bytes, 'dtype_payload_bytes':dtype_bytes,
            'precision_checked_from_headers':True, 'weight_provenance_verified':False}


def _model_python(value=None):
    value = value or os.environ.get('SHORE_MODEL_PYTHON')
    # uv/venv bin/python is commonly a symlink into a managed interpreter.
    # Dereferencing it discards pyvenv.cfg discovery and its installed packages.
    if value: return str(Path(value).expanduser().absolute())
    candidates = [Path(DEFAULT_MODEL_PYTHON),Path(sys.prefix).parent/'.venv-vllm-cu128/bin/python']
    return str(next((p for p in candidates if p.is_file()),Path(sys.executable)))


def _runtime_environment(python_executable, cache_root):
    env = os.environ.copy()
    executable = Path(python_executable)
    root = executable.parent.parent.parent
    manifest = root/'cuda128-build-manifest.json'
    if manifest.is_file():
        try: build = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError,ValueError): raise InferenceError('Invalid CPU image build manifest.') from None
        for key,field in (('CUDA_HOME','cuda_home'),('CC','cc'),('CXX','cxx')):
            if isinstance(build.get(field),str): env[key] = build[field]
        if build.get('target') == 'h200':
            env.update(TORCH_CUDA_ARCH_LIST='9.0',FLASHINFER_CUDA_ARCH_LIST='9.0a',TVM_FFI_CUDA_ARCH_LIST='9.0a')
        # Bootstrap records lowercase compiler fields. Installed cubin/JIT-cache
        # packages remain read-only inputs; only writable JIT caches move.
    if env.get('CUDA_HOME'):
        cuda = Path(env['CUDA_HOME'])
        env.update(CUDACXX=str(cuda/'bin/nvcc'), TRITON_PTXAS_PATH=str(cuda/'bin/ptxas'))
        env['PATH'] = str(cuda/'bin')+os.pathsep+env.get('PATH','')
        libraries = [str(p) for p in (cuda/'lib64',cuda/'lib',cuda/'targets/x86_64-linux/lib') if p.is_dir()]
        libraries += [part for part in env.get('LD_LIBRARY_PATH','').split(os.pathsep) if part and 'stubs' not in Path(part).parts]
        env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(libraries))
    if env.get('CXX'): env.update(CUDAHOSTCXX=env['CXX'],NVCC_CCBIN=env['CXX'])
    env['PATH'] = str(executable.parent)+os.pathsep+env.get('PATH','')
    for key,leaf in (('HF_HOME','huggingface'),('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),
                     ('FLASHINFER_WORKSPACE_BASE','flashinfer'),('DG_JIT_CACHE_DIR','deepgemm'),('TVM_FFI_CACHE_DIR','tvm-ffi'),
                     ('XDG_CACHE_HOME','xdg'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','torch-extensions')):
        path = Path(cache_root)/leaf
        path.mkdir(parents=True,exist_ok=True)
        env[key] = str(path)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
               FLASHINFER_NO_DOWNLOAD='1',HF_HUB_DISABLE_TELEMETRY='1',VLLM_NO_USAGE_STATS='1',
               VLLM_USE_RUST_FRONTEND='0',VLLM_TARGET_DEVICE='cuda')
    return env


def _adapter_metadata(path):
    root = Path(path).expanduser().resolve()
    try:
        config = json.loads((root/'adapter_config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise InferenceError('LoRA adapter needs a readable adapter_config.json.') from None
    if not isinstance(config, dict) or str(config.get('peft_type', '')).upper() != 'LORA':
        raise InferenceError('The basic loader accepts PEFT LoRA adapters; use a custom loader for other adapter formats.')
    rank = config.get('r')
    rank_pattern = config.get('rank_pattern') or {}
    if not isinstance(rank_pattern, dict):
        raise InferenceError('LoRA rank_pattern must be an object.')
    ranks = [rank, *rank_pattern.values()]
    if any(type(value) is not int or value <= 0 for value in ranks):
        raise InferenceError('LoRA ranks must be positive integers.')
    rank = max(ranks)
    supported = (1, 8, 16, 32, 64, 128, 256, 320, 512)
    maximum = next((value for value in supported if value >= rank), None)
    if maximum is None:
        raise InferenceError('LoRA rank exceeds the basic vLLM loader limit of 512.')
    if not any((root/name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        raise InferenceError('LoRA adapter weights are missing.')
    return {'path':str(root), 'rank':rank, 'max_lora_rank':maximum, 'type':'LORA'}


def prepare_model(model_key='27b', *, model_root=None, model_path=None, adapter_path=None, python_executable=None,
                  tensor_parallel_size=4, host='127.0.0.1', port=8000, max_model_len=32768,
                  gpu_memory_utilization=.9, inspect=True):
    spec = model_spec(model_key)
    if host not in ('127.0.0.1','::1','localhost'): raise InferenceError('Model service must bind a local loopback address.')
    if type(port) is not int or not 0 < port < 65536: raise InferenceError('Invalid inference port.')
    if type(tensor_parallel_size) is not int or tensor_parallel_size <= 0: raise InferenceError('TP size must be a positive integer.')
    if type(max_model_len) is not int or max_model_len < 1024: raise InferenceError('Model context length must be at least 1024.')
    if not 0 < _positive(gpu_memory_utilization,'gpu_memory_utilization') <= 1: raise InferenceError('GPU utilization must be in (0,1].')
    path = resolve_model_path(model_key,model_root=model_root,model_path=model_path)
    metadata = inspect_checkpoint(path,spec['precision']) if inspect else None
    python = _model_python(python_executable)
    adapter = _adapter_metadata(adapter_path) if adapter_path is not None else None
    base_name = spec['served_model_name']+'-base' if adapter else spec['served_model_name']
    command = [python,'-m','vllm.entrypoints.cli.main','serve',str(path),'--served-model-name',base_name,
        '--host',host,'--port',str(port),'--tensor-parallel-size',str(tensor_parallel_size),
        '--dtype','auto','--max-model-len',str(max_model_len),'--max-num-seqs','1',
        '--gpu-memory-utilization',str(gpu_memory_utilization),'--language-model-only',
        '--reasoning-parser','qwen3','--generation-config','vllm','--enforce-eager']
    if adapter:
        # Advertise the adapter under the requested model name. Requests never
        # accidentally target the unadapted base model on this managed server.
        command += ['--enable-lora', '--max-lora-rank', str(adapter['max_lora_rank']),
            '--lora-modules', json.dumps({'name':spec['served_model_name'], 'path':adapter['path'],
                                         'base_model_name':base_name})]
    return {'model':spec, 'model_path':str(path), 'container_path':'/models/'+spec['directory'],
            'adapter':adapter,
            'python_executable':python,'command':command,'checkpoint':metadata,
            'gpu_capacity_checked':False,'loaded_or_generated':False,'gpu_memory_utilization':gpu_memory_utilization,
            'tensor_parallel_size':tensor_parallel_size,'max_model_len':max_model_len,
            'base_url':'http://'+('['+host+']' if ':' in host else host)+':'+str(port)+'/v1'}


_GPU_PROBE = '''import json, importlib.metadata as md
import torch
print(json.dumps({'vllm':md.version('vllm'),'cuda':torch.version.cuda,
 'devices':[{'free_bytes':torch.cuda.mem_get_info(i)[0],'total_bytes':torch.cuda.mem_get_info(i)[1]}
  for i in range(torch.cuda.device_count())]}))'''


def verify_capacity(plan, *, environment=None, timeout=90.):
    """Reject impossible full-GPU placement; successful check is not model loading."""
    try:
        result = subprocess.run([plan['python_executable'],'-c',_GPU_PROBE],env=environment,
            capture_output=True,text=True,timeout=_positive(timeout,'timeout'),check=False)
        if result.returncode: raise ValueError()
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
        raise InferenceError('Cannot inspect the model Python/CUDA environment; run the offline H200 environment verifier first.') from None
    if not re.fullmatch(r'0\.24\.0(?:[+._-][A-Za-z0-9._+-]+)?',metadata.get('vllm','')) or metadata.get('cuda') != '12.8':
        raise InferenceError('This launch profile requires vLLM0.24.0 and Torch CUDA12.8; use the dedicated model environment.')
    devices = metadata.get('devices',[])
    if len(devices) != plan['tensor_parallel_size']:
        raise InferenceError('Visible GPU count must match tensor_parallel_size; retain the scheduler allocation.')
    payload = plan['checkpoint']['weight_payload_bytes']
    budget = sum(item['free_bytes'] for item in devices)*plan['gpu_memory_utilization']
    if payload >= budget or payload/len(devices) >= min(item['free_bytes'] for item in devices)*plan['gpu_memory_utilization']:
        raise InferenceError('Checkpoint weights exceed the visible GPU memory budget before KV/runtime overhead. '
            'Use an appropriate existing FP8 preset or more memory; this launcher never silently quantizes BF16 weights.')
    plan['gpu_capacity_checked'] = True
    plan['gpu_memory_inventory'] = devices
    return metadata


def stop_server(process):
    """Stop this managed process group, including TP workers, not other services."""
    if os.name != 'posix' and process.poll() is not None: return
    try:
        if os.name == 'posix': os.killpg(process.pid,signal.SIGTERM)
        else: process.terminate()
        process.wait(timeout=15)
    except (OSError,subprocess.TimeoutExpired):
        try:
            if os.name == 'posix': os.killpg(process.pid,signal.SIGKILL)
            else: process.kill()
            process.wait(timeout=10)
        except (OSError,subprocess.TimeoutExpired): pass
    finally:
        # The vLLM parent can exit before its GPU/TP subprocesses. Even when
        # poll() reports an exit, finish cleaning the group created by Popen.
        if os.name == 'posix':
            try: os.killpg(process.pid,signal.SIGKILL)
            except OSError: pass


@contextmanager
def managed_model(model_key=None, *, base_url=None, model_id=None, model_root=None, model_path=None, adapter_path=None,
                  python_executable=None, tensor_parallel_size=4, startup_timeout=900.,
                  max_model_len=32768, log_dir=None):
    """Load or validate a local model before entering a timed universe."""
    startup_timeout = _positive(startup_timeout,'startup_timeout')
    deadline = time.monotonic()+startup_timeout
    if base_url:
        if adapter_path is not None:
            raise InferenceError('An existing service cannot verify this adapter path; let the registry start its managed adapter service or use a custom verified loader.')
        requested = model_id or (model_spec(model_key)['served_model_name'] if model_key else None)
        client = ModelClient(base_url,requested)
        client.ready(timeout=min(30.,startup_timeout))
        yield client
        return
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        port = probe.getsockname()[1]
    plan = prepare_model(model_key or catalog()['default_model'],model_root=model_root,model_path=model_path,
        adapter_path=adapter_path,
        python_executable=python_executable,tensor_parallel_size=tensor_parallel_size,port=port,max_model_len=max_model_len)
    destination = Path(log_dir or os.environ.get('RIFT_OUTPUT_DIR') or tempfile.gettempdir())/'model-service'
    destination.mkdir(parents=True,exist_ok=True)
    environment = _runtime_environment(plan['python_executable'],destination/'.cache')
    remaining = deadline-time.monotonic()
    if remaining <= 0: raise InferenceError('Model setup exhausted the startup window.')
    verify_capacity(plan,environment=environment,timeout=min(90.,remaining))
    if time.monotonic() >= deadline: raise InferenceError('Model setup exhausted the startup window.')
    client = ModelClient(plan['base_url'],plan['model']['served_model_name'])
    logfile = destination/'vllm.log'
    (destination/'launch.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with logfile.open('ab') as stream:
        try:
            process = subprocess.Popen(plan['command'],env=environment,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=(os.name=='posix'))
        except OSError:
            raise InferenceError('Cannot start local vLLM; verify the model Python path and offline environment.') from None
        try:
            while True:
                if process.poll() is not None:
                    detail = (' Check adapter target modules and LoRA support for the selected model architecture.'
                              if plan['adapter'] else '')
                    raise InferenceError('Local vLLM exited during startup.'+detail+' See '+str(logfile))
                remaining = deadline-time.monotonic()
                if remaining <= 0: raise InferenceError('Model startup timed out; see '+str(logfile))
                try:
                    client.ready(timeout=min(3.,remaining))
                    break
                except InferenceError:
                    time.sleep(min(.5,max(0.,deadline-time.monotonic())))
            yield client
        finally:
            stop_server(process)


__all__ = ['InferenceError','ModelClient','prepare_model','managed_model','verify_capacity',
           'resolve_model_path','stop_server']
