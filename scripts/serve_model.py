"""Inspect or start one supplied local model using the CUDA12.8 profile."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shore_agent.models import InferenceError, prepare_model, verify_capacity, _runtime_environment, stop_server
from rift.model_registry import catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='27b',help='Preset id or supplied directory/model name.')
    parser.add_argument('--model-root')
    parser.add_argument('--model-path')
    parser.add_argument('--model-python',help='Model venv Python; separate from the experiment Python.')
    parser.add_argument('--tp-size',type=int,default=4)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--max-model-len',type=int,default=32768)
    parser.add_argument('--cache-root',type=Path,default=Path(os.environ.get('RIFT_OUTPUT_DIR','/tmp/shore-model'))/'.cache')
    parser.add_argument('--dry-run',action='store_true',help='Inspect local headers and print argv; do not query GPUs or load models.')
    parser.add_argument('--plan-only',action='store_true',help='Print configuration without requiring model files, GPUs or model packages.')
    parser.add_argument('--list',action='store_true',help='List the seven presets without touching a checkpoint.')
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(catalog(),ensure_ascii=False,indent=2)); return 0
    try:
        plan = prepare_model(args.model,model_root=args.model_root,model_path=args.model_path,
            python_executable=args.model_python,tensor_parallel_size=args.tp_size,port=args.port,
            max_model_len=args.max_model_len,inspect=not args.plan_only)
        if args.dry_run or args.plan_only:
            print(json.dumps(plan,ensure_ascii=False,indent=2)); return 0
        environment = _runtime_environment(plan['python_executable'],args.cache_root.resolve())
        verify_capacity(plan,environment=environment)
        process = subprocess.Popen(plan['command'],env=environment,start_new_session=(os.name=='posix'))
        try: return process.wait()
        except KeyboardInterrupt: return 130
        finally: stop_server(process)
    except (InferenceError,ValueError) as exc:
        parser.exit(2,'Model setup error: '+str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
