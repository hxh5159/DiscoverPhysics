"""Compatibility entry for the candidate worker maintained inside agent/."""
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
implementation = import_module('agent.tool_worker')
if __name__ == '__main__':
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
