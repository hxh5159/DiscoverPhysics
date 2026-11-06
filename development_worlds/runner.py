"""Run the same submitted LLM agent on any of the six public universes.

All observations, source and detailed scores are public in this development
driver. The organizer-only driver runs the submitted agent against unknown
environments instead; it does not give their implementation to contestants.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import catalog
from .console import DevelopmentConsole
from .evaluation import evaluate_law
from .service import DevelopmentSession, serve_session

ROOT = Path(__file__).resolve().parents[1]


def _stop(process, *, graceful=False):
    if graceful:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if os.name != "nt":
        # Managed model servers are children of the agent. A dedicated process
        # group prevents orphaned GPU allocations on timeout or interruption.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=5)


def run_episode(world, directory, command, *, seed=0, startup_seconds=900.,
                evaluate=True, cwd=None):
    """Start one agent process; judge energy from the session, never self-report.

    ``command`` is a list of executable arguments, never a shell string. The
    caller can supply a different editable agent with the same Console API.
    A new directory is required to preserve earlier experiments and results.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    console = DevelopmentConsole(world, catalog.noise_sigma(world), seed, catalog.config(world))
    session = DevelopmentSession(console, directory/"session")
    environment = os.environ.copy()
    environment["RIFT_OUTPUT_DIR"] = str(directory/"agent-output")
    environment["PYTHONPATH"] = str(ROOT)+os.pathsep+environment.get("PYTHONPATH", "")
    launched = time.monotonic()
    reason, exit_code = "agent_exited", None
    with serve_session(session) as credentials, (directory/"agent.log").open("w", encoding="utf-8") as log:
        environment.update(RIFT_CONSOLE_URL=credentials["endpoint"], RIFT_SESSION_TOKEN=credentials["token"])
        process = subprocess.Popen(list(command), cwd=str(cwd or ROOT), env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                snapshot = session.snapshot()
                if snapshot["closed"]:
                    reason = "submitted"
                    break
                if snapshot["started"] and snapshot["budget"]["wall_clock_remaining_seconds"] <= 0:
                    reason = "timeout"
                    break
                if not snapshot["started"] and time.monotonic()-launched >= startup_seconds:
                    reason = "startup_timeout"
                    break
                time.sleep(.05)
            exit_code = process.poll()
        finally:
            _stop(process, graceful=reason == "submitted")
            exit_code = process.returncode
        finalized = session.finalize(reason)
    result = {"world": str(world), "scope": "public_development", "end_reason": finalized["reason"],
              "agent_exit_code": exit_code, "budget": finalized["budget"],
              "has_valid_submission": finalized["has_valid_submission"],
              "wall_clock_seconds": catalog.WALL_CLOCK_SECONDS, "startup_and_exploration_elapsed_seconds": time.monotonic()-launched}
    if session.latest is not None:
        result["law_sha256"] = session.latest["law_sha256"]
        result["claims_sha256"] = session.latest["claims_sha256"]
        if evaluate:
            pair = Path(session.latest["artifact_directory"])
            claims = json.loads((pair/"claims.json").read_text(encoding="utf-8-sig"))
            result["score"] = evaluate_law(world, pair/"law.py", claims, finalized["budget"]["energy_used"])
    elif evaluate:
        result["score"] = {"total": 0., "reason": "no_valid_submission"}
    (directory/"result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", choices=catalog.WORLDS, default=list(catalog.WORLDS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900.)
    parser.add_argument("--model", default=os.environ.get("SHORE_MODEL", "27b"))
    parser.add_argument("--base-url", default=os.environ.get("SHORE_MODEL_BASE_URL"))
    parser.add_argument("--model-id", default=os.environ.get("SHORE_MODEL_ID"))
    parser.add_argument("--model-root", default=os.environ.get("SHORE_MODEL_ROOT"))
    parser.add_argument("--model-path", default=os.environ.get("SHORE_MODEL_PATH"))
    parser.add_argument("--model-python", default=os.environ.get("SHORE_MODEL_PYTHON"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=3,
                        help="Maximum baseline model calls per universe (default 3); zero removes this cap.")
    parser.add_argument("--agent-dir", type=Path,
                        help="Complete agent/ directory containing strategy.json and run.py.")
    parser.add_argument("--models-file", type=Path,
                        help="Model registry paired with --agent-dir; includes every model or adapter path.")
    parser.add_argument("--agent-command", nargs=argparse.REMAINDER,
                        help="Optional custom command argument list; put this option last.")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.max_turns < 0 or not 0 < args.startup_timeout < float("inf"):
        parser.error("seed and max-turns must be nonnegative; startup timeout must be finite and positive.")
    if bool(args.agent_dir) != bool(args.models_file):
        parser.error('--agent-dir and --models-file must be supplied together.')
    if args.agent_dir and args.agent_command:
        parser.error('--agent-command cannot be combined with the strategy directory options.')
    if args.output.exists():
        parser.error("Choose a new output directory; prior runs are not overwritten.")
    command = args.agent_command or [sys.executable, "-m", "agent.run", "--model", args.model,
        "--tp-size", str(args.tp_size), "--max-tokens", str(args.max_tokens),
        "--startup-timeout", str(args.startup_timeout), "--max-turns", str(args.max_turns)]
    if args.agent_dir:
        from rift.strategy import load_strategy_submission
        try:
            load_strategy_submission(args.agent_dir, args.models_file, check_files=True)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        command = [sys.executable, '-I', str(ROOT/'scripts/run_strategy.py'),
            '--agent-dir', str(args.agent_dir.expanduser().resolve()),
            '--models-file', str(args.models_file.expanduser().resolve()), '--',
            '--max-turns', str(args.max_turns), '--max-tokens', str(args.max_tokens),
            '--startup-timeout', str(args.startup_timeout), '--tp-size', str(args.tp_size)]
    if not args.agent_command:
        for key in ("base_url", "model_id", "model_root", "model_path", "model_python"):
            value = getattr(args, key)
            if value:
                command.extend(["--"+key.replace("_", "-"), value])
    args.output.mkdir(parents=True)
    results = []
    for offset, world in enumerate(args.worlds):
        print(f"Public universe {world}: starting agent", flush=True)
        result = run_episode(world, args.output/f"world-{world}", command,
                             seed=args.seed+offset, startup_seconds=args.startup_timeout)
        results.append(result)
        print(json.dumps({"world": world, "has_valid_submission": result["has_valid_submission"],
                          "score": result.get("score", {}).get("total")}, ensure_ascii=False), flush=True)
    summary = {"scope": "public_development_not_official_ranking", "worlds": results,
               "all_have_valid_submission": all(item["has_valid_submission"] for item in results)}
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding="utf-8")
    return 0 if summary["all_have_valid_submission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
