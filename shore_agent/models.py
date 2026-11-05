"""Compatibility import for the editable strategy maintained inside agent/."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('agent.models')
