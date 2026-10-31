"""Public transport for the organizer's private experimental console.

The client carries only permitted requests/observations.  It neither imports a
true-world simulator nor evaluates a submitted law.  HTTP and Unix-domain
socket transports carry the same authenticated JSON protocol.
"""
from __future__ import annotations

import base64
import csv
import http.client
import json
import math
import os
from pathlib import Path
import socket
from urllib.parse import unquote, urlsplit


CONSOLE_PROTOCOL_VERSION = "1.0.0"
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_ARTIFACT_FIELDS = ("law", "claims", "report")


class ConsoleError(RuntimeError):
    """An explicit transport/configuration/console error, never a score."""

    def __init__(self, code, path, message):
        self.code, self.path, self.message = code, path, message
        super().__init__(f"{code} at {path}: {message}")

    def as_dict(self):
        return {"code": self.code, "path": self.path, "message": self.message}


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(self.socket_path)
        except BaseException:
            connection.close()
            raise
        self.sock = connection


def _json_default(value):
    # Common NumPy inputs may be used by a team's own experiment planner.
    # No dependency or simulator is imported to serialize their numeric data.
    if type(value).__module__.split(".")[0] == "numpy":
        if hasattr(value, "tolist"):
            return value.tolist()
    raise TypeError("Unsupported JSON value")


def _decode_json(data):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON constant")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite JSON number")
        return result

    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=object_pairs,
                      parse_constant=invalid_constant, parse_float=finite_float)


def _csv_rows(archive):
    """Lossless public records; opaque JSON columns do not guess physics."""
    if isinstance(archive, dict):
        if "first_light" in archive:
            yield "first_light", 0, archive["first_light"]
        records = archive.get("observations", archive.get("records", []))
        if isinstance(records, list):
            for index, record in enumerate(records):
                yield "observation", index, record
        else:
            yield "archive", 0, archive
    elif isinstance(archive, list):
        for index, record in enumerate(archive):
            yield "observation", index, record


def _export_archive(archive, destination):
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    content = json.dumps(archive, ensure_ascii=False, allow_nan=False, indent=2)
    (directory / "archive.json").write_text(content + "\n", encoding="utf-8")
    # One full returned record per row: no rounding, dropped coordinate, or
    # re-noising.  archive.json also retains the budget and all root metadata.
    with (directory / "archive.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_type", "record_index", "record_json"])
        for kind, index, record in _csv_rows(archive):
            writer.writerow([kind, index, json.dumps(record, ensure_ascii=False, allow_nan=False)])


class Console:
    """A client bound by a secret token to one organizer-created session.

    ``endpoint`` is a complete HTTP(S) endpoint for diagnostics, or
    ``unix:///run/rift/console.sock`` in an isolated Linux runner.  The latter
    sends POST /v1/console over the designated Unix socket.  Constructor
    arguments fall back to RIFT_CONSOLE_URL and RIFT_SESSION_TOKEN.

    There are no automatic retries.  ``timeout=None`` leaves transport timing
    to the runner's independent session watchdog.  A caller-specified timeout
    is a network guard only, not a change to the exploration or physical time.
    """

    def __init__(self, endpoint=None, token=None, *, timeout=None):
        endpoint = os.environ.get("RIFT_CONSOLE_URL") if endpoint is None else endpoint
        token = os.environ.get("RIFT_SESSION_TOKEN") if token is None else token
        if not isinstance(token, str) or not token or any(not 33 <= ord(c) <= 126 for c in token):
            raise ConsoleError("configuration_error", "$.token", "A nonempty single-line session token is required.")
        if timeout is not None:
            try:
                valid_timeout = (not isinstance(timeout, bool) and isinstance(timeout, (int, float))
                                 and math.isfinite(timeout) and timeout > 0)
            except OverflowError:
                valid_timeout = False
            if not valid_timeout:
                raise ConsoleError("configuration_error", "$.timeout", "Transport timeout must be positive and finite, or None.")
        if not isinstance(endpoint, str) or not endpoint:
            raise ConsoleError("configuration_error", "$.endpoint", "An official console endpoint is required.")
        try:
            parts = urlsplit(endpoint)
            port = parts.port
        except ValueError:
            raise ConsoleError("configuration_error", "$.endpoint", "Malformed console endpoint.") from None
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ConsoleError("configuration_error", "$.endpoint", "Endpoint credentials, queries and fragments are not permitted.")
        if parts.scheme in ("http", "https"):
            if not parts.hostname:
                raise ConsoleError("configuration_error", "$.endpoint", "HTTP endpoint must contain a host.")
            target = parts.path or "/v1/console"
            socket_path = None
        elif parts.scheme == "unix":
            if parts.netloc or not parts.path.startswith("/") or not hasattr(socket, "AF_UNIX"):
                raise ConsoleError("configuration_error", "$.endpoint", "An absolute Unix socket path and AF_UNIX support are required.")
            socket_path = unquote(parts.path)
            target = "/v1/console"
        else:
            raise ConsoleError("configuration_error", "$.endpoint", "Use the official http, https or unix endpoint.")
        self._scheme, self._host, self._port = parts.scheme, parts.hostname, port
        self._socket_path, self._target = socket_path, target
        self._token, self._timeout = token, timeout

    def __repr__(self):
        return f"Console(transport={self._scheme!r}, token=<redacted>)"

    def _connection(self):
        if self._scheme == "unix":
            return _UnixHTTPConnection(self._socket_path, self._timeout)
        connection_type = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        return connection_type(self._host, self._port, timeout=self._timeout)

    def _safe(self, value):
        return value.replace(self._token, "<redacted>")

    def _call(self, method, params):
        request = {"protocol_version": CONSOLE_PROTOCOL_VERSION, "method": method, "params": params}
        try:
            payload = json.dumps(request, ensure_ascii=False, allow_nan=False,
                                 default=_json_default).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            raise ConsoleError("invalid_request", "$.params", "Request must contain finite JSON-compatible values.") from None
        connection = self._connection()
        try:
            connection.request("POST", self._target, body=payload, headers={
                "Authorization": "Bearer " + self._token,
                "Content-Type": "application/json; charset=utf-8", "Accept": "application/json",
            })
            response = connection.getresponse()
            status = response.status
            data = response.read(_MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raise ConsoleError("transport_error", "$", "Official console transport failed; this is not a scientific score. No automatic retry was made.") from None
        finally:
            connection.close()
        if len(data) > _MAX_RESPONSE_BYTES:
            raise ConsoleError("protocol_error", "$", "Console response exceeds the transport guard.")
        try:
            envelope = _decode_json(data)
        except (ValueError, UnicodeError):
            raise ConsoleError("protocol_error", "$", "Console did not return a valid JSON envelope.") from None
        if not isinstance(envelope, dict) or type(envelope.get("ok")) is not bool:
            raise ConsoleError("protocol_error", "$", "Console response is missing a boolean ok field.")
        if envelope["ok"]:
            if not 200 <= status < 300 or set(envelope) != {"ok", "result"} or not isinstance(envelope["result"], (dict, list)):
                raise ConsoleError("protocol_error", "$", "Console success envelope or HTTP status is invalid.")
            return envelope["result"]
        error = envelope.get("error")
        if set(envelope) != {"ok", "error"} or not isinstance(error, dict) or set(error) != {"code", "path", "message"} or any(not isinstance(error[k], str) for k in error):
            raise ConsoleError("protocol_error", "$", "Console error envelope is invalid.")
        raise ConsoleError(*(self._safe(error[key]) for key in ("code", "path", "message")))

    def mission(self):
        return self._call("mission", {})

    def observe(self, times):
        return self._call("observe", {"times": times})

    def launch(self, *, t_launch, readout, probes, source_properties=None):
        return self._call("launch", {"t_launch": t_launch, "readout": readout,
                                    "probes": probes, "source_properties": {} if source_properties is None else source_properties})

    def archive(self, path=None):
        result = self._call("archive", {})
        if path is not None:
            try:
                _export_archive(result, path)
            except (OSError, TypeError, ValueError):
                raise ConsoleError("archive_export_error", "$.path", "Unable to export returned public records to the requested local directory.") from None
        return result

    def _artifacts(self, method, law, claims, report=None):
        params = {}
        for key, value in zip(_ARTIFACT_FIELDS, (law, claims, report)):
            if key == "report" and value is None:
                continue
            try:
                data = Path(value).read_bytes()
            except (OSError, TypeError, ValueError):
                raise ConsoleError("file_read_error", f"$.{key}", f"Unable to read the local {key} file.") from None
            params[key + "_b64"] = base64.b64encode(data).decode("ascii")
        return self._call(method, params)

    def save_draft(self, law="law.py", claims="claims.json"):
        return self._artifacts("save_draft", law, claims)

    def submit(self, law="law.py", claims="claims.json", report=None):
        return self._artifacts("submit", law, claims, report)


__all__ = ["CONSOLE_PROTOCOL_VERSION", "Console", "ConsoleError"]
