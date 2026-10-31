"""Check an agent manifest; never execute submitted code or load model weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rift.submission import (SubmissionError, load_manifest, manifest_digest,
                             check_submission_files, SUBMISSION_SCHEMA_VERSION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, help="Submission directory; default: manifest parent")
    parser.add_argument("--check-files", action="store_true", help="Explicitly check relative model files; declared SHA-256 hashing may be expensive")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        result = {"ok": True, "schema_version": SUBMISSION_SCHEMA_VERSION,
                  "manifest_sha256": manifest_digest(manifest), "submission_id": manifest["submission_id"],
                  "team_id": manifest["team_id"], "model_count": len(manifest["models"]),
                  "code_execution_checked": False, "model_provenance_checked": False,
                  "scientific_correctness_checked": False, "files_checked": args.check_files}
        if args.check_files:
            result["models"] = check_submission_files(manifest, args.root or args.manifest.parent)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except SubmissionError as exc:
        print(json.dumps({"ok": False, "errors": [exc.as_dict()]}, ensure_ascii=False))
        return 2
    except OSError:
        print(json.dumps({"ok": False, "errors": [{"code": "artifact_read_error", "path": "$.models", "message": "Unable to inspect the declared model artifact."}]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
