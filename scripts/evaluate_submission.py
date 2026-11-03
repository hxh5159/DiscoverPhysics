"""Run the externally supplied organizer evaluator from the team's project.

This public entry contains no hidden simulator, reference data or scoring code.
The organizer supplies that separate runtime at evaluation time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organizer-root", required=True, type=Path,
                        help="directory containing the external direct_project_eval.py")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("submission.json"))
    parser.add_argument("--output", required=True, type=Path,
                        help="new organizer-retained run directory")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--agent-python", help="actual Python executable for the team's agent")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    project = args.project_root.expanduser().resolve()
    runtime = args.organizer_root.expanduser().resolve()
    evaluator = runtime / "direct_project_eval.py"
    manifest = args.manifest.expanduser()
    if not manifest.is_absolute():
        manifest = project / manifest
    output = args.output.expanduser().resolve()
    if not project.is_dir():
        print(json.dumps({"ok": False, "error": "project_root_missing"}), file=sys.stderr)
        return 2
    if not evaluator.is_file():
        print(json.dumps({"ok": False, "error": "organizer_runtime_missing",
            "message": "The organizer must connect the nine-universe runtime before evaluation."}), file=sys.stderr)
        return 2
    command = [sys.executable, str(evaluator), "--project-root", str(project),
               "--manifest", str(manifest.resolve()), "--output", str(output)]
    if args.profile:
        command += ["--profile", str(args.profile.expanduser().resolve())]
    if args.agent_python:
        command += ["--agent-python", args.agent_python]
    if args.check_only:
        command.append("--check-only")
    try:
        # Preserve the installed environment. The external runner provides the
        # session-specific Console settings; no shell is used here.
        completed = subprocess.run(command, cwd=project, env=os.environ.copy(), check=False)
        return completed.returncode
    except OSError:
        print(json.dumps({"ok": False, "error": "organizer_evaluator_start_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
