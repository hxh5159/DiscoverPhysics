"""Local public-development adapter for the official authenticated Console API.

Only the six public simulators are used here. Formal unknown environments are
provided by the organizer service, which accepts the same client requests.
"""
from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time

from rift.console import CONSOLE_PROTOCOL_VERSION, ConsoleError
from rift.law import validate_source
from scripts.validate_claims import parse_claims


class DevelopmentSession:
    """One public episode with atomic valid drafts and a monotonic deadline."""

    def __init__(self, console, directory):
        self.console = console
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.mission_data = None
        self.latest = None
        self.closed = False
        self.reason = None
        self._counter = 0

    def _expired(self):
        return self.mission_data is not None and self.console.budget["wall_clock_remaining_seconds"] <= 0

    def snapshot(self):
        # The runner can stop a stuck request without waiting for the request
        # lock. The public SDK calls are owned by one agent per session.
        return {"started": self.console._started_at is not None, "closed": self.closed,
                "expired": self._expired(), "budget": dict(self.console.budget)}

    def finalize(self, reason):
        with self.lock:
            self.closed = True
            if self.reason is None:
                self.reason = reason
            return {"reason": self.reason, "budget": dict(self.console.budget),
                    "has_valid_submission": self.latest is not None}

    def _decode(self, params, name, maximum):
        value = params.get(name+"_b64")
        if not isinstance(value, str) or len(value) > 2*maximum:
            raise ValueError(f"Missing or oversized {name} artifact.")
        raw = base64.b64decode(value, validate=True)
        if not raw or len(raw) > maximum:
            raise ValueError(f"Invalid {name} artifact size.")
        return raw

    def _save(self, params, submit):
        allowed = {"law_b64", "claims_b64"} | ({"report_b64"} if submit else set())
        if set(params)-allowed or not {"law_b64", "claims_b64"}.issubset(params):
            raise ValueError("Provide law_b64 and claims_b64, with optional submit report_b64.")
        law = self._decode(params, "law", 1024*1024)
        claims = self._decode(params, "claims", 65536)
        validate_source(law.decode("utf-8-sig"))
        parse_claims(claims.decode("utf-8-sig"))
        report = self._decode(params, "report", 65536) if "report_b64" in params else None
        if self._expired():
            raise TimeoutError("The exploration window ended during artifact validation.")
        self._counter += 1
        target = self.directory/f"pair-{self._counter:06d}"
        target.mkdir()
        (target/"law.py").write_bytes(law)
        (target/"claims.json").write_bytes(claims)
        if report is not None:
            (target/"report.md").write_bytes(report)
        receipt = {"law_sha256": hashlib.sha256(law).hexdigest(),
                   "claims_sha256": hashlib.sha256(claims).hexdigest(),
                   "artifact_directory": str(target)}
        if self._expired():
            raise TimeoutError("The exploration window ended before the pair could be committed.")
        self.latest = receipt
        if submit:
            self.closed, self.reason = True, "submitted"
        return {"accepted": True, "submitted": submit,
                "law_sha256": receipt["law_sha256"], "claims_sha256": receipt["claims_sha256"],
                "budget": dict(self.console.budget)}

    def dispatch(self, method, params):
        with self.lock:
            if self.closed or self._expired():
                raise ConsoleError("session_closed", "$", "The episode has ended; no late action was accepted.")
            if not isinstance(params, dict):
                raise ValueError("params must be an object.")
            if method == "mission":
                if params:
                    raise ValueError("mission takes no parameters.")
                result = self.console.mission()
                self.mission_data = copy.deepcopy(result)
                return result
            if self.mission_data is None:
                raise ValueError("Call mission before other operations.")
            if method == "observe":
                if set(params) != {"times"}:
                    raise ValueError("observe requires times only.")
                return self.console.observe(params["times"])
            if method == "launch":
                if set(params) != {"probes", "t_launch", "readout", "source_properties"}:
                    raise ValueError("launch requires probes, t_launch, readout and source_properties.")
                return self.console.launch(**params)
            if method == "archive":
                if params:
                    raise ValueError("archive takes no parameters.")
                return self.console.archive()
            if method in ("save_draft", "submit"):
                return self._save(params, method == "submit")
            raise ValueError("Unknown console operation.")


@contextmanager
def serve_session(session):
    """Expose one public session only on loopback, with an unguessable token."""
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            status = 200
            try:
                if self.path != "/v1/console":
                    raise ConsoleError("not_found", "$", "Unknown console path.")
                if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer "+token):
                    raise ConsoleError("unauthorized", "$", "Session authorization failed.")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 3*1024*1024:
                    raise ValueError("Invalid request size.")
                request = json.loads(self.rfile.read(size))
                if (not isinstance(request, dict) or set(request) != {"protocol_version", "method", "params"}
                        or request["protocol_version"] != CONSOLE_PROTOCOL_VERSION):
                    raise ValueError("Invalid console protocol request.")
                response = {"ok": True, "result": session.dispatch(request["method"], request["params"])}
            except ConsoleError as exc:
                status, response = 400, {"ok": False, "error": exc.as_dict()}
            except TimeoutError:
                status, response = 409, {"ok": False, "error": {"code": "session_closed", "path": "$",
                    "message": "The exploration deadline has expired."}}
            except Exception as exc:
                status, response = 400, {"ok": False, "error": {"code": "invalid_request", "path": "$",
                    "message": f"{type(exc).__name__}: {str(exc)[:500]}"}}
            raw = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (OSError, BrokenPipeError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"endpoint": f"http://127.0.0.1:{server.server_port}/v1/console", "token": token}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
