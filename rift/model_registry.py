"""Public model names and precision variants, without loading model packages.

Directory aliases do not change precision. The registry does not establish
weight provenance or imply that every variant fits the allocated hardware.
"""
from __future__ import annotations
import json
from pathlib import Path

_CATALOG_FILE = Path(__file__).with_name('model_catalog.json')


def catalog():
    """Return detached public configuration; never import Torch or vLLM."""
    return json.loads(_CATALOG_FILE.read_text(encoding='utf-8'))


def _basename(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty public model identifier.')
    return value.replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]


def model_spec(value):
    """Resolve a preset, exact served name or registered directory alias."""
    name = _basename(value)
    for item in catalog()['models']:
        if name in (item['id'], item['base_model'], item['served_model_name'], item['directory'],
                    *item['directory_aliases']):
            return item
    raise ValueError('This identifier has no supplied model preset: '+name)


def resolve_model_id(value):
    return model_spec(value)['id']


def canonical_base_model(value):
    """Normalize known aliases while preserving each weight precision variant."""
    # Submission IDs are names, not host paths. Runtime server metadata may
    # contain a model path; callers handling that metadata use model_spec first.
    if not isinstance(value, str) or '/' in value or '\\' in value:
        raise ValueError('Expected a model name, not a filesystem path.')
    try:
        return model_spec(value)['base_model']
    except ValueError:
        raise ValueError('Model is outside the published whitelist and supplied variants.') from None


# The current catalog is also the published whitelist. Keep no fallback to
# retired models: declarations, local model services and strategy registries
# must all enforce the same current selection.
WHITELIST_MODELS = tuple(item['base_model'] for item in catalog()['models'])

__all__ = ['catalog', 'model_spec', 'resolve_model_id', 'canonical_base_model', 'WHITELIST_MODELS']
