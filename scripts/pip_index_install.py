"""Prefer the configured pip mirror and explicitly retry download failures once.

This stdlib-only helper never combines indexes. The caller's constraints,
official wheel links and build flags are preserved verbatim on both attempts.
Compiler and dependency-resolution errors are not retried against another index.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


TAIL_BYTES = 192 * 1024
_INDEX_ARGUMENTS = ("--index-url", "--extra-index-url", "--trusted-host", "--no-index")
_BUILD_FAILURES = (
    "resolutionimpossible", "conflicting dependencies", "failed building wheel",
    "failed to build", "subprocess-exited-with-error", "cmake error", "cmakeerror",
    "compiler identification is unknown", "error: command", "error: cannot install",
)
_CONNECTION_FAILURES = (
    "connection refused", "connection reset", "connection aborted", "connectionerror",
    "connecttimeout", "readtimeout", "read timed out", "timed out", "max retries exceeded",
    "failed to establish a new connection", "temporary failure in name resolution",
    "nameresolutionerror", "network is unreachable", "remote end closed connection",
)


def _index(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an HTTP(S) package index URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Package index URLs must not contain queries or fragments")
    return value.rstrip("/")


def mirror_download_failed(output: str, index_url: str) -> bool:
    """Classify only errors tied to this mirror, not arbitrary build downloads."""
    lowered = output.lower()
    if any(marker in lowered for marker in _BUILD_FAILURES):
        return False
    if any(marker in lowered for marker in (
            "certificate_verify_failed", "certificate verify failed",
            "hashes do not match", "do not match the hashes", "error: hash-mismatch")):
        return False
    hostname = urlsplit(index_url).hostname.lower()
    for line in lowered.splitlines():
        if hostname not in line:
            continue
        # Recent pip versions put ``error: connection-failed`` on one line,
        # ``Failed to connect to <host> while fetching ...`` on the next, and
        # the socket reason below it. The target on this line is sufficient:
        # do not combine an earlier mirror URL with an unrelated host's error.
        if re.search(r"\bfailed to connect to\s+" + re.escape(hostname) + r"(?=$|[\s:/])", line):
            return True
        if not any(marker in line for marker in ("error", "warning: retrying", "could not fetch url")):
            continue
        if re.search(r"(?:http(?:s)?(?: error)?[ :]+|status(?: code)?[ :=]+|\b)(404|408|429|500|502|503|504)\b", line):
            return True
        if any(marker in line for marker in _CONNECTION_FAILURES):
            return True
    return False


def _stream_attempt(command, *, env, log_path):
    """Keep compiler output on disk and stream it without retaining it in RAM."""
    with Path(log_path).open("xb") as log:
        try:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as exc:
            log.write((type(exc).__name__ + ": " + str(exc) + "\n").encode("utf-8", errors="replace"))
            return 127
        try:
            while True:
                chunk = process.stdout.read1(32 * 1024)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                stream = getattr(sys.stdout, "buffer", None)
                if stream is not None:
                    stream.write(chunk)
                    stream.flush()
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def _tail(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell()-TAIL_BYTES))
        return stream.read(TAIL_BYTES).decode("utf-8", errors="replace")


def install(python, requirements_args, *, index_url, trusted_host, log_dir,
            fallback_index_url="https://pypi.org/simple", operation="install", runner=None):
    """Run pip and return its status; ``runner(cmd, env=..., log_path=...)`` is injectable.

    An injected runner must write the attempt log and return an integer status.
    Each invocation uses a new receipt directory; prior logs are never replaced.
    """
    if operation not in ("install", "download"):
        raise ValueError("operation must be install or download")
    primary = _index(index_url)
    fallback = _index(fallback_index_url) if fallback_index_url else None
    arguments = list(requirements_args)
    if not python or not arguments:
        raise ValueError("An interpreter and pip requirement arguments are required")
    for argument in arguments:
        if not isinstance(argument, str) or "\0" in argument:
            raise ValueError("Pip arguments must be strings without NUL characters")
        if (argument == "-i" or argument.startswith("-i=") or
                any(argument == flag or argument.startswith(flag+"=") for flag in _INDEX_ARGUMENTS)):
            raise ValueError("Pass package indexes through the helper, never through extra pip arguments")
    if not trusted_host or any(c.isspace() for c in trusted_host):
        raise ValueError("One explicit trusted-host value is required")
    environment = os.environ.copy()
    environment.pop("PIP_INDEX_URL", None)
    environment.pop("PIP_EXTRA_INDEX_URL", None)
    # os.devnull is /dev/null on the target Linux instance (nul on Windows QA).
    environment["PIP_CONFIG_FILE"] = os.devnull
    root = Path(log_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="pip-index-", dir=root))
    attempt_runner = runner or _stream_attempt
    receipt = {"operation": operation, "requirements_args": arguments, "attempts": [],
               "fallback_used": False, "indexes_combined": False}
    indexes = [primary]
    if fallback and fallback != primary:
        indexes.append(fallback)
    code = 1
    for number, index in enumerate(indexes, 1):
        logfile = run/(str(number)+".log")
        command = [str(python), "-m", "pip", operation, *arguments,
            "--index-url", index, "--trusted-host", trusted_host, "--no-cache-dir",
            "--disable-pip-version-check", "--timeout", "600", "--retries", "5"]
        label = "configured index" if number == 1 else "explicit fallback index"
        print("[SHORE pip] " + label + ": " + index, flush=True)
        code = int(attempt_runner(command, env=environment, log_path=logfile))
        if not logfile.exists():
            raise RuntimeError("Pip runner did not create its attempt log")
        receipt["attempts"].append({"index_url": index, "exit_code": code, "log": logfile.name})
        receipt["fallback_used"] = number > 1
        if code == 0:
            break
        if number > 1 or len(indexes) == 1 or not mirror_download_failed(_tail(logfile), primary):
            break
        print("[SHORE pip] Mirror download failed; retrying the same constrained request once with "
              + fallback + ". No additional index is mixed into this request.", flush=True)
    receipt["exit_code"] = code
    (run/"receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("[SHORE pip] Attempt logs: " + str(run), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--trusted-host", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--fallback-index-url", default="https://pypi.org/simple")
    parser.add_argument("--operation", choices=("install", "download"), default="install")
    parser.add_argument("requirements_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    requirements = args.requirements_args
    if requirements and requirements[0] == "--":
        requirements = requirements[1:]
    try:
        return install(args.python, requirements, index_url=args.index_url, trusted_host=args.trusted_host,
            log_dir=args.log_dir, fallback_index_url=args.fallback_index_url, operation=args.operation)
    except (OSError, ValueError, RuntimeError) as exc:
        print("[SHORE pip] " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
