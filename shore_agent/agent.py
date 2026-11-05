"""Compatibility entry for the strategy now maintained in agent/run.py."""
from importlib import import_module
import sys

implementation = import_module('agent.run')
if __name__ == '__main__':
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
