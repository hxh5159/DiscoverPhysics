"""Public transport tests with invented payloads and a local fake service."""
import base64
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rift import Console, ConsoleError
from rift.console import CONSOLE_PROTOCOL_VERSION


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(raw.decode("utf-8"))
        self.server.requests.append((self.path, dict(self.headers), request))
        if self.server.delay:
            time.sleep(self.server.delay)
        status, payload = self.server.responder(request)
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


class FakeConsole:
    def __init__(self, responder=None, *, socket_path=None, delay=0):
        if socket_path is None:
            self.server = HTTPServer(("127.0.0.1", 0), Handler)
            self.endpoint = "http://127.0.0.1:%d/v1/console" % self.server.server_address[1]
        else:
            self.server = socketserver.UnixStreamServer(str(socket_path), Handler)
            self.endpoint = "unix://" + str(socket_path).replace("\\", "/")
        self.server.requests = []
        self.server.delay = delay
        self.server.responder = responder or (lambda request: (200, {"ok": True, "result": {"method": request["method"]}}))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @property
    def requests(self):
        return self.server.requests

    def client(self, **kwargs):
        return Console(endpoint=self.endpoint, token="test-session-token", **kwargs)


class ConsoleClientTests(unittest.TestCase):
    def test_environment_configuration_and_thin_public_import(self):
        with FakeConsole() as fake:
            with patch.dict(os.environ, {"RIFT_CONSOLE_URL": fake.endpoint,
                                        "RIFT_SESSION_TOKEN": "test-session-token"}):
                console = Console()
                self.assertEqual(console.mission(), {"method": "mission"})
                self.assertNotIn("test-session-token", repr(console))
        path, headers, request = fake.requests[0]
        self.assertEqual(path, "/v1/console")
        self.assertEqual(headers["Authorization"], "Bearer test-session-token")
        self.assertEqual(request, {"protocol_version": CONSOLE_PROTOCOL_VERSION,
                                   "method": "mission", "params": {}})

    def test_explicit_configuration_overrides_environment(self):
        with FakeConsole() as fake:
            with patch.dict(os.environ, {"RIFT_CONSOLE_URL": "invalid://wrong", "RIFT_SESSION_TOKEN": "wrong"}):
                fake.client().mission()
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0][1]["Authorization"], "Bearer test-session-token")

    def test_observe_launch_requests_preserve_controls_without_private_fields(self):
        with FakeConsole() as fake:
            console = fake.client()
            console.observe(times=[0.0, .005, 1.0])
            console.launch(t_launch=.5, readout=[0., .1],
                           probes=[{"pos": [1., 2.], "vel": [.2, 0.], "properties": {"p2": -1.}}])
            console.launch(t_launch=.5, readout=[.1], probes=[], source_properties={"p1": 2.})
        self.assertEqual([row[2]["method"] for row in fake.requests], ["observe", "launch", "launch"])
        self.assertEqual(fake.requests[0][2]["params"], {"times": [0., .005, 1.]})
        launch = fake.requests[1][2]["params"]
        self.assertEqual(launch["source_properties"], {})
        self.assertEqual(launch["probes"][0]["properties"], {"p2": -1.})
        self.assertNotIn("world_id", launch)
        self.assertNotIn("session_id", launch)
        self.assertEqual(fake.requests[2][2]["params"]["source_properties"], {"p1": 2.})

    def test_numpy_numeric_inputs_are_serialized_without_changing_values(self):
        import numpy as np
        with FakeConsole() as fake:
            fake.client().observe(times=np.array([0, .005], dtype=np.float64))
        self.assertEqual(fake.requests[0][2]["params"]["times"], [0., .005])

    def test_pair_upload_preserves_raw_law_and_claims_bytes_and_optional_report(self):
        sources = {"law.py": b'\xef\xbb\xbf# unicode\r\nPARAMS={}\r\n',
                   "claims.json": b'{"dynamics_order":2,"hidden_matter":false,"time_dependence":false}\r\n',
                   "report.md": "自己的说明\n".encode("utf-8")}
        with tempfile.TemporaryDirectory() as temporary:
            for name, content in sources.items():
                (Path(temporary) / name).write_bytes(content)
            with FakeConsole() as fake:
                console = fake.client()
                console.save_draft(law=Path(temporary)/"law.py", claims=Path(temporary)/"claims.json")
                console.submit(law=Path(temporary)/"law.py", claims=Path(temporary)/"claims.json",
                               report=Path(temporary)/"report.md")
        self.assertEqual([r[2]["method"] for r in fake.requests], ["save_draft", "submit"])
        self.assertEqual(set(fake.requests[0][2]["params"]), {"law_b64", "claims_b64"})
        params = fake.requests[1][2]["params"]
        for field in ("law", "claims", "report"):
            suffix = {"law": ".py", "claims": ".json", "report": ".md"}[field]
            self.assertEqual(base64.b64decode(params[field+"_b64"]), sources[field+suffix])

    def test_missing_pair_file_does_not_send_partial_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            law = Path(temporary)/"law.py"
            law.write_text("# local candidate", encoding="utf-8")
            with FakeConsole() as fake:
                with self.assertRaises(ConsoleError) as caught:
                    fake.client().submit(law=law, claims=Path(temporary)/"missing.json")
        self.assertEqual(caught.exception.code, "file_read_error")
        self.assertEqual(caught.exception.path, "$.claims")
        self.assertEqual(fake.requests, [])

    def test_archive_exports_same_public_records_without_extra_requests(self):
        archive = {"world_id": "example", "first_light": {"times": [0., 1.], "positions": [[[1., 2.]], [[3., 4.]]]},
                   "observations": [{"experiment_id": "L1", "times": [.5], "velocities": [[[.2, .3]]]}],
                   "budget": {"energy_remaining": 99}}
        def responder(request):
            self.assertEqual(request, {"protocol_version": "1.0.0", "method": "archive", "params": {}})
            return 200, {"ok": True, "result": archive}
        with tempfile.TemporaryDirectory() as temporary:
            with FakeConsole(responder) as fake:
                self.assertEqual(fake.client().archive(temporary), archive)
            self.assertEqual(json.loads((Path(temporary)/"archive.json").read_text(encoding="utf-8")), archive)
            with (Path(temporary)/"archive.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["record_type"] for r in rows], ["first_light", "observation"])
            self.assertEqual(json.loads(rows[1]["record_json"]), archive["observations"][0])
        self.assertEqual(len(fake.requests), 1)

    def test_structured_server_error_and_secret_redaction(self):
        def responder(request):
            return 400, {"ok": False, "error": {"code": "invalid_request", "path": "$.readout[0]",
                                               "message": "Non-grid time. test-session-token"}}
        with FakeConsole(responder) as fake:
            with self.assertRaises(ConsoleError) as caught:
                fake.client().observe([.003])
        self.assertEqual(caught.exception.code, "invalid_request")
        self.assertEqual(caught.exception.path, "$.readout[0]")
        self.assertNotIn("test-session-token", str(caught.exception))
        self.assertEqual(set(caught.exception.as_dict()), {"code", "path", "message"})
        self.assertEqual(len(fake.requests), 1)

    def test_invalid_response_envelopes_fail_without_exposing_raw_body(self):
        invalid = [b"not-json-test-session-token", b'{"ok":true,"ok":false}',
                   b'{"ok":true,"result":{"value":NaN}}', b'{"ok":true,"result":{"value":1e999}}',
                   {"ok": 1, "result": {}}, {"ok": True, "result": 7},
                   {"ok": True, "result": {}, "debug": "private"},
                   {"ok": False, "error": {"code": "bad", "message": "missing path"}}]
        for payload in invalid:
            with self.subTest(payload=payload):
                with FakeConsole(lambda request: (200, payload)) as fake:
                    with self.assertRaises(ConsoleError) as caught:
                        fake.client().mission()
                self.assertEqual(caught.exception.code, "protocol_error")
                self.assertNotIn("test-session-token", str(caught.exception))

    def test_redirect_is_not_followed(self):
        with FakeConsole(lambda request: (302, {"ok": True, "result": {}})) as fake:
            with self.assertRaises(ConsoleError) as caught:
                fake.client().mission()
        self.assertEqual(caught.exception.code, "protocol_error")
        self.assertEqual(len(fake.requests), 1)

    def test_invalid_configuration_is_explicit_and_does_not_echo_token(self):
        for endpoint in ["ftp://localhost/x", "http://name:secret@localhost/x", "http://localhost/x?secret=1",
                         "http://localhost/x#fragment", "http://localhost:bad/x", ""]:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ConsoleError) as caught:
                    Console(endpoint=endpoint, token="test-session-token")
                self.assertEqual(caught.exception.code, "configuration_error")
                self.assertNotIn("test-session-token", str(caught.exception))
        for token in (None, "", "bad\ntoken", "bad token"):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ConsoleError):
                    Console(endpoint="http://localhost/x", token=token)
        for timeout in (False, 0, -1, float("nan"), float("inf"), 10**1000):
            with self.assertRaises(ConsoleError):
                Console(endpoint="http://localhost/x", token="test-session-token", timeout=timeout)

    def test_nonfinite_or_unserializable_requests_do_not_reach_service(self):
        with FakeConsole() as fake:
            for times in ([float("nan")], [float("inf")], [object()]):
                with self.subTest(times=times):
                    with self.assertRaises(ConsoleError) as caught:
                        fake.client().observe(times)
                    self.assertEqual(caught.exception.code, "invalid_request")
        self.assertEqual(fake.requests, [])

    def test_transport_failure_is_not_a_scientific_result_and_is_not_retried(self):
        with FakeConsole(delay=.1) as fake:
            with self.assertRaises(ConsoleError) as caught:
                fake.client(timeout=.005).mission()
        self.assertEqual(caught.exception.code, "transport_error")
        self.assertIn("not a scientific score", caught.exception.message)
        self.assertEqual(len(fake.requests), 1)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX") and os.name != "nt", "Unix deployment transport requires a Unix platform")
    def test_unix_socket_carries_the_same_authenticated_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            with FakeConsole(socket_path=Path(temporary)/"console.sock") as fake:
                self.assertEqual(fake.client().mission(), {"method": "mission"})
        self.assertEqual(fake.requests[0][0], "/v1/console")
        self.assertEqual(fake.requests[0][1]["Authorization"], "Bearer test-session-token")
        self.assertEqual(fake.requests[0][2]["protocol_version"], "1.0.0")


if __name__ == "__main__":
    unittest.main()
