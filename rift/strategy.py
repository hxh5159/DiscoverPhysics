"""Public strategy-directory and multi-model path contracts.

These checks read local metadata and file names. They do not execute policy
code, load model tensors, or establish weight provenance or OS isolation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath

from .submission import (SubmissionError, _keys, _model_path, _pairs,
                         _invalid_constant, artifact_digest, validate_manifest)

STRATEGY_SCHEMA_VERSION = "1.0.0"
MODEL_REGISTRY_SCHEMA_VERSION = "1.0.0"
_STRATEGY_FIELDS = ("schema_version", "team_id", "submission_id", "version")
_REGISTRY_FIELDS = ("schema_version", "primary_model_id", "models")


def _read_json(path, field):
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            return json.load(stream, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(field, "A readable UTF-8 JSON file is required.") from exc


def _registry(value):
    _keys(value, _REGISTRY_FIELDS, _REGISTRY_FIELDS, "$.model_registry")
    if value["schema_version"] != MODEL_REGISTRY_SCHEMA_VERSION:
        raise SubmissionError("$.model_registry.schema_version", "Expected 1.0.0.")
    normalized = validate_manifest({
        "schema_version": "1.0.0", "team_id": "registry", "submission_id": "registry",
        "version": "1", "entrypoint": ["python", "-m", "agent.run"], "models": value["models"],
    })["models"]
    if any("id" not in item for item in normalized):
        raise SubmissionError("$.models", "Every model must declare a unique binding id.")
    primary = value["primary_model_id"]
    if not isinstance(primary, str) or primary not in {item["id"] for item in normalized}:
        raise SubmissionError("$.primary_model_id", "Select an existing model binding id.")
    return {"schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
            "primary_model_id": primary, "models": normalized}


def _weight_file(root, relative, field):
    if (not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
            or Path(relative).is_absolute()):
        raise SubmissionError(field, "Weight shard names must be relative paths.")
    _model_path(relative, field)
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise SubmissionError(field, "Every declared weight shard must exist inside its model directory and be nonempty.")
    return path


def _check_checkpoint(path, field):
    root = Path(path).resolve()
    if not root.is_dir():
        raise SubmissionError(field, "A complete local checkpoint directory is required.")
    if not isinstance(_read_json(root/"config.json", field), dict):
        raise SubmissionError(field, "Model config.json must be a JSON object.")
    if not any((root/name).is_file() and (root/name).stat().st_size > 0
               for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise SubmissionError(field, "Local tokenizer assets are required.")
    index = next((root/name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                  if (root/name).is_file()), None)
    if index is not None:
        content = _read_json(index, field)
        mapping = content.get("weight_map") if isinstance(content, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise SubmissionError(field, "Checkpoint index requires a nonempty weight_map.")
        names = list(mapping.values())
    else:
        names = [item.name for item in root.iterdir()
                 if item.is_file() and item.suffix in (".safetensors", ".bin")]
    if not names:
        raise SubmissionError(field, "Checkpoint weight files are missing.")
    checked = {_weight_file(root, name, field) for name in names}
    return {"path": str(root), "weight_files": len(checked), "tensors_loaded": False}


def _check_adapter(path, field):
    root = Path(path).resolve()
    if not root.is_dir() or not isinstance(_read_json(root/"adapter_config.json", field), dict):
        raise SubmissionError(field, "Adapter directory with adapter_config.json is required.")
    if not any((root/name).is_file() and (root/name).stat().st_size > 0
               for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise SubmissionError(field, "Adapter weights are missing or empty.")
    return {"path": str(root), "adapter_config_checked": True, "tensors_loaded": False}


def check_model_registry_files(registry):
    """Check every full checkpoint and adapter, including secondary agents."""
    registry = _registry(registry)
    checked = []
    for index, model in enumerate(registry["models"]):
        row = {"id": model["id"], "checkpoint": _check_checkpoint(model["path"], f"$.models[{index}].path")}
        if "sha256" in model:
            if artifact_digest(model["path"]) != model["sha256"]:
                raise SubmissionError(f"$.models[{index}].sha256", "Checkpoint SHA-256 does not match its declaration.")
            row["sha256_verified"] = True
        if "adapter_path" in model:
            row["adapter"] = _check_adapter(model["adapter_path"], f"$.models[{index}].adapter_path")
        checked.append(row)
    return checked


def load_model_registry(path, *, check_files=False):
    """Resolve all weight paths relative to this registry, regardless of cwd."""
    source = Path(path).expanduser().resolve()
    registry = _registry(_read_json(source, "$.model_registry"))
    for model in registry["models"]:
        for key in ("path", "adapter_path"):
            if key in model:
                declared = Path(model[key])
                resolved = declared if declared.is_absolute() else source.parent/declared
                model[key] = resolved.resolve().as_posix()
    if check_files:
        check_model_registry_files(registry)
    return registry


def model_binding(registry, model_id=None):
    """Return one named binding or the declared primary, preserving variants."""
    registry = _registry(registry)
    selected = registry["primary_model_id"] if model_id is None else model_id
    for model in registry["models"]:
        if model["id"] == selected:
            return copy.deepcopy(model)
    raise SubmissionError("$.model_id", "Unknown model binding id.")


def check_strategy_files(agent_dir):
    """Require policy files under a single real folder named agent.

    This is a submission layout check. The organizer still controls the process
    identity and filesystem permissions used for executing submitted code.
    """
    lexical = Path(agent_dir).expanduser().absolute()
    if lexical.name != "agent" or lexical.is_symlink() or not lexical.is_dir():
        raise SubmissionError("$.agent_dir", "Use an existing directory named agent, without a directory symlink.")
    root = lexical.resolve()
    for name in ("run.py", "__init__.py", "strategy.json"):
        if not (root/name).is_file():
            raise SubmissionError("$.agent_dir", f"Required strategy file is missing: {name}.")
    files = []
    for entry in root.rglob("*"):
        if entry.is_symlink() or not entry.resolve().is_relative_to(root):
            raise SubmissionError("$.agent_dir", "Strategy files must stay inside agent and cannot be symlinks.")
        if entry.is_file():
            files.append(entry.relative_to(root).as_posix())
        elif not entry.is_dir():
            raise SubmissionError("$.agent_dir", "Strategy supports only regular files and directories.")
    descriptor = _read_json(root/"strategy.json", "$.strategy")
    _keys(descriptor, _STRATEGY_FIELDS, _STRATEGY_FIELDS, "$.strategy")
    if descriptor["schema_version"] != STRATEGY_SCHEMA_VERSION:
        raise SubmissionError("$.strategy.schema_version", "Expected 1.0.0.")
    identity = validate_manifest(dict(descriptor, entrypoint=["python", "-m", "agent.run"],
                                      models=[{"base_model": "27b", "path": "weights"}]))
    return {"agent_dir": str(root), "files": sorted(files),
            "strategy": {key: identity[key] for key in _STRATEGY_FIELDS}}


def load_strategy_submission(agent_dir, models_file, *, check_files=True):
    """Produce the one internal submission from the two supplied path inputs."""
    strategy = check_strategy_files(agent_dir)["strategy"]
    registry = load_model_registry(models_file, check_files=check_files)
    return validate_manifest(dict(strategy, entrypoint=["python", "-m", "agent.run"],
                                  models=registry["models"]))


__all__ = ["STRATEGY_SCHEMA_VERSION", "MODEL_REGISTRY_SCHEMA_VERSION", "load_model_registry",
           "model_binding", "check_model_registry_files", "check_strategy_files", "load_strategy_submission"]
