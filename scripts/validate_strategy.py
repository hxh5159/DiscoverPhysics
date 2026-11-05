"""Check the strategy directory and all declared model/adapter paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.strategy import check_strategy_files, load_model_registry, load_strategy_submission
from rift.submission import SubmissionError, manifest_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--models-file", required=True, type=Path)
    parser.add_argument("--check-files", action="store_true",
                        help="Check every checkpoint, tokenizer, weight shard and optional adapter")
    args = parser.parse_args(argv)
    try:
        manifest = load_strategy_submission(args.agent_dir, args.models_file, check_files=args.check_files)
        strategy = check_strategy_files(args.agent_dir)
        registry = load_model_registry(args.models_file)
        result = {"ok": True, "team_id": manifest["team_id"], "submission_id": manifest["submission_id"],
                  "manifest_sha256": manifest_digest(manifest), "strategy_file_count": len(strategy["files"]),
                  "model_count": len(registry["models"]), "primary_model_id": registry["primary_model_id"],
                  "files_checked": args.check_files, "code_executed": False, "model_loaded": False,
                  "model_provenance_verified": False}
    except SubmissionError as exc:
        print(json.dumps({"ok": False, "errors": [exc.as_dict()]}, ensure_ascii=False))
        return 2
    except OSError:
        print(json.dumps({"ok": False, "error": "Unable to inspect the submitted paths."}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
