"""Validate an outer agent submission, without executing code or loading weights.

The manifest declares a command and model provenance.  It never grants host
mounts, network access, or permission to inspect the organizer environment.
Container isolation and weight provenance review belong to the trusted runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


SUBMISSION_SCHEMA_VERSION = "1.0.0"
WHITELIST_MODELS = (
    "Qwen3.8-27B", "GLM-4.7-Flash", "Qwen3.6-35B-A3B",
    "Kimi-Linear-48B-A3B-Instruct", "GLM-4.5-Air-FP8",
    "Qwen3.5-122B-A10B-FP8", "DeepSeek-V4-Flash", "MiniMax-M2.5",
    "Qwen3.5-397B-A17B-FP8",
)
_ROOT_FIELDS = frozenset(("schema_version", "submission_id", "team_id", "version", "entrypoint", "models"))
_MODEL_FIELDS = frozenset(("base_model", "path", "revision", "sha256", "adaptation"))
_ADAPTATIONS = frozenset(("base", "quantized", "sft", "rl", "other"))
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_SHELLS = frozenset(("sh", "bash", "dash", "zsh", "fish", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"))


class SubmissionError(ValueError):
    def __init__(self, path, message, code="invalid_submission"):
        self.path, self.message, self.code = path, message, code
        super().__init__(f"{path}: {message}")

    def as_dict(self):
        return {"code": self.code, "path": self.path, "message": self.message}


def _text(value, path, maximum=1024):
    if not isinstance(value, str) or not value or len(value) > maximum or not value.isprintable():
        raise SubmissionError(path, f"Expected a nonempty printable string of at most {maximum} characters.")
    return value


def _keys(value, allowed, required, path):
    if not isinstance(value, dict):
        raise SubmissionError(path, "Expected an object.")
    missing = sorted(set(required) - value.keys())
    unknown = sorted(value.keys() - set(allowed), key=str)
    if missing:
        raise SubmissionError(path, f"Missing fields: {', '.join(missing)}.")
    if unknown:
        raise SubmissionError(path, "Unknown fields are not accepted.")


def _model_path(value, path):
    value = _text(value, path)
    if "\\" in value or ":" in value:
        raise SubmissionError(path, "Use a relative POSIX path or /models/... inside the container.")
    parts = value.split("/")
    if ".." in parts or "." in parts or "" in parts[1:]:
        raise SubmissionError(path, "Path traversal, repeated separators, and dot components are not accepted.")
    p = PurePosixPath(value)
    if p.is_absolute() and (len(p.parts) < 3 or p.parts[1] != "models"):
        raise SubmissionError(path, "Absolute model paths must be under the official /models mount.")
    return value


def validate_manifest(value):
    """Return a detached normalized manifest; no code, image or model is read."""
    _keys(value, _ROOT_FIELDS, _ROOT_FIELDS, "$")
    if value["schema_version"] != SUBMISSION_SCHEMA_VERSION:
        raise SubmissionError("$.schema_version", f"Expected {SUBMISSION_SCHEMA_VERSION}.")
    result = {"schema_version": SUBMISSION_SCHEMA_VERSION}
    for name in ("submission_id", "team_id"):
        text = _text(value[name], f"$.{name}", 64)
        if not _IDENTIFIER.fullmatch(text):
            raise SubmissionError(f"$.{name}", "Use an identifier made of ASCII letters, digits, dot, dash or underscore, starting with a letter or digit.")
        result[name] = text
    result["version"] = _text(value["version"], "$.version", 128)
    command = value["entrypoint"]
    if not isinstance(command, list) or not command:
        raise SubmissionError("$.entrypoint", "Expected a nonempty argv array; shell command strings are not accepted.")
    result["entrypoint"] = [_text(arg, f"$.entrypoint[{i}]", 4096) for i, arg in enumerate(command)]
    executable = PurePosixPath(command[0]).name.lower()
    if executable in _SHELLS:
        raise SubmissionError("$.entrypoint[0]", "Launch the agent executable directly, rather than a shell wrapper.")
    if "\\" in command[0] or ":" in command[0] or ".." in command[0].split("/"):
        raise SubmissionError("$.entrypoint[0]", "Expected a container executable or a non-traversing POSIX path.")
    models = value["models"]
    if not isinstance(models, list) or not models:
        raise SubmissionError("$.models", "Declare at least one model from the published whitelist.")
    normalized = []
    for i, item in enumerate(models):
        path = f"$.models[{i}]"
        _keys(item, _MODEL_FIELDS, ("base_model", "path"), path)
        if not isinstance(item["base_model"], str) or item["base_model"] not in WHITELIST_MODELS:
            raise SubmissionError(path + ".base_model", "Expected a model identifier from the published whitelist.")
        model = {"base_model": item["base_model"], "path": _model_path(item["path"], path + ".path")}
        if "revision" in item:
            model["revision"] = _text(item["revision"], path + ".revision", 256)
        if "sha256" in item:
            digest = _text(item["sha256"], path + ".sha256", 64)
            if not _SHA256.fullmatch(digest):
                raise SubmissionError(path + ".sha256", "Expected a 64-character hexadecimal SHA-256 digest.")
            model["sha256"] = digest.lower()
        if "adaptation" in item:
            adaptation = item["adaptation"]
            if not isinstance(adaptation, str) or adaptation not in _ADAPTATIONS:
                raise SubmissionError(path + ".adaptation", "Expected base, quantized, sft, rl or other.")
            model["adaptation"] = adaptation
        normalized.append(model)
    result["models"] = normalized
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SubmissionError("$", f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise SubmissionError("$", "JSON NaN and Infinity are not accepted.")


def parse_manifest(text):
    try:
        if not isinstance(text, str):
            raise SubmissionError("$", "Expected UTF-8 JSON text.")
        value = json.loads(text.removeprefix("\ufeff"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SubmissionError("$", "Manifest must be valid UTF-8 JSON.") from exc
    return validate_manifest(value)


def load_manifest(path):
    try:
        return parse_manifest(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise SubmissionError("$", "Unable to read the manifest as UTF-8 JSON.", "manifest_read_error") from exc


def manifest_digest(value):
    """Canonical manifest identity; not evidence that files or weights match."""
    encoded = json.dumps(validate_manifest(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact_digest(path):
    """Hash a file, or a directory's sorted [{path,sha256}] inventory.

    Deliberately explicit: hashing a multi-gigabyte model can be expensive.
    Symlinks are not followed; absolute official mounts are checked by deploy.
    """
    path = Path(path)
    if path.is_symlink():
        raise SubmissionError("$.models", "A packaged model artifact must not be a symlink.")
    if path.is_file():
        return _file_hash(path)
    if not path.is_dir():
        raise SubmissionError("$.models", "A declared local model artifact does not exist.")
    inventory = []
    for child in sorted(path.rglob("*"), key=lambda p: p.relative_to(path).as_posix()):
        if child.is_symlink():
            raise SubmissionError("$.models", "A packaged model directory must not contain symlinks.")
        if child.is_file():
            inventory.append({"path": child.relative_to(path).as_posix(), "sha256": _file_hash(child)})
        elif not child.is_dir():
            raise SubmissionError("$.models", "Only regular files and directories are accepted in a model artifact.")
    return hashlib.sha256(json.dumps(inventory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_submission_files(value, root):
    """Check relative model artifacts under root, never arbitrary host paths.

    /models references are returned as deferred mount checks.  This operation
    does not load weights or prove model provenance, and does not inspect code.
    """
    value = validate_manifest(value)
    root = Path(root).resolve()
    if not root.is_dir():
        raise SubmissionError("$.root", "Submission root must be an existing directory.")
    checked = []
    for i, model in enumerate(value["models"]):
        item = {"path": model["path"], "sha256_verified": False}
        if PurePosixPath(model["path"]).is_absolute():
            item["status"] = "official_mount_check_deferred"
        else:
            lexical = root.joinpath(*PurePosixPath(model["path"]).parts)
            resolved = lexical.resolve()
            if not resolved.is_relative_to(root):
                raise SubmissionError(f"$.models[{i}].path", "Model artifact resolves outside the submission root.")
            # Reject intermediate symlinks as well as a symlink at the leaf.
            current = root
            for part in PurePosixPath(model["path"]).parts:
                current = current / part
                if current.is_symlink():
                    raise SubmissionError(f"$.models[{i}].path", "Packaged model paths must not use symlinks.")
            if not resolved.exists():
                raise SubmissionError(f"$.models[{i}].path", "Declared local model artifact does not exist.")
            item["status"] = "local_artifact_exists"
            if "sha256" in model:
                actual = artifact_digest(resolved)
                if actual != model["sha256"]:
                    raise SubmissionError(f"$.models[{i}].sha256", "Model artifact SHA-256 does not match the declaration.")
                item["sha256_verified"] = True
        checked.append(item)
    return checked


__all__ = ["SUBMISSION_SCHEMA_VERSION", "WHITELIST_MODELS", "SubmissionError",
           "validate_manifest", "parse_manifest", "load_manifest", "manifest_digest",
           "artifact_digest", "check_submission_files"]
