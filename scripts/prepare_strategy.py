"""Prepare the two submission descriptors without copying any model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.model_registry import catalog, model_spec
from rift.strategy import load_model_registry, load_strategy_submission
from rift.submission import SubmissionError, validate_manifest


def prepare(agent_dir, models_file, *, team_id, submission_id, version="1.0.0",
            model="27b", model_path=None, adaptation="base", adapter_path=None,
            models_from=None, overwrite=False):
    declared_agent = Path(agent_dir).expanduser().absolute()
    if declared_agent.is_symlink():
        raise SubmissionError("$.agent_dir", "The strategy directory must not be a symlink.")
    agent = declared_agent.resolve()
    destination = Path(models_file).expanduser().resolve()
    if agent.name != "agent" or not (agent/"run.py").is_file() or not (agent/"__init__.py").is_file():
        raise SubmissionError("$.agent_dir", "Select the supplied agent directory containing run.py and __init__.py.")
    strategy_path = agent/"strategy.json"
    if strategy_path == destination:
        raise SubmissionError("$.models_file", "The strategy descriptor and weight registry need different paths.")
    for path in (strategy_path, destination):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Existing descriptor is preserved: {path}. Use --overwrite to replace it.")
        if path.is_symlink():
            raise SubmissionError("$.output", "Descriptor output paths must not be symlinks.")
    descriptor = {"schema_version": "1.0.0", "team_id": team_id,
                  "submission_id": submission_id, "version": version}
    if models_from is not None:
        registry = load_model_registry(models_from)
    else:
        spec = model_spec(model)
        path = model_path or (Path(catalog()["model_root"])/spec["directory"]).as_posix()
        entry = {"id": "primary", "role": "discovery", "base_model": spec["base_model"],
                 "path": path, "adaptation": adaptation}
        if adapter_path is not None:
            entry["adapter_path"] = adapter_path
        # Explicit relative paths are defined relative to the output registry.
        normalized = validate_manifest(dict(descriptor, entrypoint=["python", "-m", "agent.run"], models=[entry]))
        registry = {"schema_version": "1.0.0", "primary_model_id": "primary", "models": normalized["models"]}
    validate_manifest(dict(descriptor, entrypoint=["python", "-m", "agent.run"], models=registry["models"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    documents = [(destination, registry), (strategy_path, descriptor)]
    previous = {path: path.read_bytes() if path.exists() else None for path, _ in documents}
    written = []
    try:
        for path, data in documents:
            with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
                written.append(path)
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        manifest = load_strategy_submission(agent, destination, check_files=False)
    except BaseException:
        for path in reversed(written):
            if previous[path] is None:
                path.unlink()
            else:
                path.write_bytes(previous[path])
        raise
    return {"agent_dir": str(agent), "models_file": str(destination),
            "team_id": manifest["team_id"], "submission_id": manifest["submission_id"],
            "model_count": len(manifest["models"]), "weights_copied": False, "model_loaded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, default=Path("agent"))
    parser.add_argument("--models-file", type=Path, default=Path("models.json"))
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--model", default="27b")
    parser.add_argument("--model-path")
    parser.add_argument("--adaptation", choices=("base", "quantized", "sft", "rl", "other"), default="base")
    parser.add_argument("--adapter-path")
    parser.add_argument("--models-from", type=Path, help="Existing multi-model registry to normalize and copy as metadata")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.agent_dir, args.models_file, team_id=args.team_id,
            submission_id=args.submission_id, version=args.version, model=args.model,
            model_path=args.model_path, adaptation=args.adaptation, adapter_path=args.adapter_path,
            models_from=args.models_from, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
