"""Launch one self-contained agent directory with the official public SDK.

Use an isolated Python interpreter to keep the organizer's import paths out
of the submitted process. Operating-system permissions are set by its parent.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

SDK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_ROOT))

from rift.strategy import check_strategy_files, load_model_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--models-file', type=Path, required=True)
    parser.add_argument('agent_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    strategy = args.agent_dir.expanduser().absolute()
    models = args.models_file.expanduser().resolve()
    check_strategy_files(strategy)
    strategy = strategy.resolve()
    load_model_registry(models, check_files=True)
    os.environ['SHORE_STRATEGY_DIR'] = str(strategy)
    os.environ['SHORE_MODEL_REGISTRY'] = str(models)
    os.environ['SHORE_SDK_ROOT'] = str(SDK_ROOT)
    os.chdir(strategy)

    package = importlib.util.spec_from_file_location('agent', strategy/'__init__.py',
        submodule_search_locations=[str(strategy)])
    if package is None or package.loader is None:
        raise RuntimeError('Unable to load the submitted agent package.')
    module = importlib.util.module_from_spec(package)
    sys.modules['agent'] = module
    package.loader.exec_module(module)
    forwarded = args.agent_args
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    sys.argv = ['agent.run', *forwarded]
    runpy.run_module('agent.run', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
