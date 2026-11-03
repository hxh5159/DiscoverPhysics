"""Index recovery checks; no model, CUDA or external package service is used."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.pip_index_install import install, mirror_download_failed


MIRROR = "http://mirror.example/simple"
TRITON_DISCONNECT = """Collecting triton==3.6.0 (from torch==2.11.0+cu128)
  WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'RemoteDisconnected('Remote end closed connection without response')': /repository/pypi/packages/triton/3.6.0/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
error: connection-failed

× Failed to connect to mirror.example while fetching http://mirror.example/repository/pypi/packages/triton/3.6.0/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
╰─> the connection was closed without a reply from the server.

hint: Are you connected to the Internet? If so, check whether your system can connect to mirror.example before trying again. There may be a firewall or proxy that's preventing the connection.
"""


class Runner:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def __call__(self, command, *, env, log_path):
        self.calls.append((command, env.copy()))
        code, text = next(self.responses)
        Path(log_path).write_text(text, encoding="utf-8")
        return code


class PipIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pip-index-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
    def run_install(self, runner, arguments=None, **options):
        return install(sys.executable, arguments or ["example==1"], index_url=MIRROR,
            trusted_host="mirror.example", log_dir=self.root, runner=runner, **options)
    def test_primary_success_is_not_retried_and_pip_configuration_is_cleared(self):
        runner = Runner([(0,"success")])
        with patch.dict(os.environ, {"PIP_INDEX_URL":"http://unexpected.example", "PIP_EXTRA_INDEX_URL":"http://extra.example"}):
            self.assertEqual(self.run_install(runner),0)
        command, environment = runner.calls[0]
        self.assertEqual(len(runner.calls),1)
        self.assertNotIn("PIP_INDEX_URL",environment)
        self.assertNotIn("PIP_EXTRA_INDEX_URL",environment)
        self.assertEqual(environment["PIP_CONFIG_FILE"],os.devnull)
        self.assertEqual(command[command.index("--index-url")+1],MIRROR)
        self.assertIn("--trusted-host",command)
        self.assertIn("--no-cache-dir",command)
        self.assertNotIn("--extra-index-url",command)
    def test_mirror_artifact_404_retries_official_once_preserving_all_constraints(self):
        runner = Runner([(1,"ERROR: HTTP error 404 while getting http://mirror.example/packages/missing.whl"),(0,"success")])
        flags = ["-c","constraints.txt","--find-links","https://download.pytorch.org/whl/cu128/torch/",
                 "--no-binary=vllm","--no-build-isolation","torch==2.11.0+cu128","vllm==0.24.0"]
        self.assertEqual(self.run_install(runner,flags),0)
        self.assertEqual(len(runner.calls),2)
        for command,_ in runner.calls:
            self.assertEqual(command[4:4+len(flags)],flags)
            self.assertNotIn("--extra-index-url",command)
        self.assertEqual(runner.calls[1][0][runner.calls[1][0].index("--index-url")+1],"https://pypi.org/simple")
        receipt = json.loads(next(self.root.glob("pip-index-*/receipt.json")).read_text(encoding="utf-8"))
        self.assertTrue(receipt["fallback_used"])
        self.assertFalse(receipt["indexes_combined"])
        self.assertEqual([row["exit_code"] for row in receipt["attempts"]],[1,0])
    def test_502_and_connection_failures_are_recognized(self):
        for text in ("ERROR: 502 Server Error for url: http://mirror.example/packages/a.whl",
                     "WARNING: Retrying after connection broken by HTTPConnectionPool(host='mirror.example'): Read timed out"):
            self.assertTrue(mirror_download_failed(text,MIRROR),text)
    def test_modern_pip_multiline_triton_disconnect_is_recognized(self):
        # Same output shape as the reported failure; only the mirror host is anonymized.
        self.assertTrue(mirror_download_failed(TRITON_DISCONNECT,MIRROR))
    def test_multiline_connection_failure_retries_official_once_without_changing_pins(self):
        flags = ['-c','constraints-cu128.txt',
                 '--find-links','https://download.pytorch.org/whl/cu128/torch/',
                 'torch==2.11.0+cu128','triton==3.6.0']
        runner = Runner([(1,TRITON_DISCONNECT),(0,'success')])
        self.assertEqual(self.run_install(runner,flags),0)
        self.assertEqual(len(runner.calls),2)
        for command,_ in runner.calls:
            self.assertEqual(command[4:4+len(flags)],flags)
            self.assertNotIn('--extra-index-url',command)
        command = runner.calls[1][0]
        self.assertEqual(command[command.index('--index-url')+1],'https://pypi.org/simple')
        receipt = json.loads(next(self.root.glob('pip-index-*/receipt.json')).read_text(encoding='utf-8'))
        self.assertTrue(receipt['fallback_used'])
        self.assertEqual([item['exit_code'] for item in receipt['attempts']],[1,0])
    def test_modern_connection_errors_require_the_mirror_as_the_exact_target(self):
        for host in ('github.com','mirror.example.evil','notmirror.example'):
            message = ('Looking in indexes: '+MIRROR+'\n'+
                       TRITON_DISCONNECT.replace('mirror.example',host))
            with self.subTest(host=host):
                self.assertFalse(mirror_download_failed(message,MIRROR))
        for failure in ('CMake Error: missing CUDA',
                        'ERROR: THESE PACKAGES DO NOT MATCH THE HASHES',
                        'error: hash-mismatch',
                        'ERROR: CERTIFICATE_VERIFY_FAILED'):
            with self.subTest(failure=failure):
                self.assertFalse(mirror_download_failed(TRITON_DISCONNECT+'\n'+failure,MIRROR))
    def test_dependency_compiler_hash_and_unrelated_network_errors_do_not_fallback(self):
        messages = [
            "ERROR: ResolutionImpossible: dependencies conflict. http://mirror.example/simple",
            "CMake Error: cannot find CUDA. ERROR: 404 http://mirror.example/unused",
            "ERROR: Failed building wheel for vllm. http://mirror.example/404",
            "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES http://mirror.example/a.whl",
            "ERROR: HTTP error 404 https://github.com/other/project/archive.tar.gz",
            "ERROR: No matching distribution found for example==999",
            "ERROR: CERTIFICATE_VERIFY_FAILED at http://mirror.example/simple",
        ]
        for message in messages:
            with self.subTest(message=message):
                runner=Runner([(1,message)])
                self.assertEqual(self.run_install(runner),1)
                self.assertEqual(len(runner.calls),1)
    def test_fallback_failure_is_returned_without_a_third_attempt(self):
        runner=Runner([(1,"ERROR: HTTP error 404 at http://mirror.example/a.whl"),
                       (7,"ERROR: downstream failure")])
        self.assertEqual(self.run_install(runner),7)
        self.assertEqual(len(runner.calls),2)
    def test_disabled_fallback_and_same_index_are_not_retried(self):
        for fallback in ("",MIRROR):
            runner=Runner([(1,"ERROR: HTTP error 404 at http://mirror.example/a.whl")])
            self.assertEqual(self.run_install(runner,fallback_index_url=fallback),1)
            self.assertEqual(len(runner.calls),1)
    def test_existing_logs_are_retained(self):
        self.run_install(Runner([(0,"first")]))
        self.run_install(Runner([(0,"second")]))
        self.assertEqual(len(list(self.root.glob("pip-index-*/receipt.json"))),2)
    def test_extra_index_arguments_are_rejected_before_any_subprocess(self):
        for args in (["--extra-index-url","https://other.example/simple","a"],
                     ["--index-url=https://other.example/simple","a"],["--no-index","a"]):
            runner=Runner([])
            with self.assertRaises(ValueError): self.run_install(runner,args)
            self.assertEqual(runner.calls,[])
    def test_cli_accepts_the_pip_argument_separator_without_importing_gpu_libraries(self):
        script=Path(__file__).resolve().parents[1]/"scripts/pip_index_install.py"
        done=subprocess.run([sys.executable,str(script),"--help"],capture_output=True,text=True,timeout=10)
        self.assertEqual(done.returncode,0)
        self.assertIn("--fallback-index-url",done.stdout)


class LocalIndexRecoveryTest(unittest.TestCase):
    def test_real_pip_download_recovers_from_a_local_bad_artifact_link(self):
        """A tiny invented wheel is downloaded, never installed; no external access."""
        archive=io.BytesIO()
        name="shore_pip_fixture-1.0.0-py3-none-any.whl"
        with ZipFile(archive,"w",ZIP_DEFLATED) as wheel:
            wheel.writestr("shore_pip_fixture/__init__.py", "__version__='1.0.0'\n")
            wheel.writestr("shore_pip_fixture-1.0.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: shore-pip-fixture\nVersion: 1.0.0\n")
            wheel.writestr("shore_pip_fixture-1.0.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nGenerator: test-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            wheel.writestr("shore_pip_fixture-1.0.0.dist-info/RECORD", "")
        payload=archive.getvalue()
        digest=hashlib.sha256(payload).hexdigest()
        requests=[]
        def handler(broken):
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_GET(self):
                    requests.append((broken,self.path))
                    if self.path.rstrip("/")=="/simple/shore-pip-fixture":
                        data=('<a href="/packages/'+name+'#sha256='+digest+'">'+name+'</a>').encode()
                        self.send_response(200);self.send_header("Content-Type","text/html")
                    elif self.path=="/packages/"+name and not broken:
                        data=payload;self.send_response(200);self.send_header("Content-Type","application/octet-stream")
                    else:
                        self.send_error(404);return
                    self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
            return Handler
        servers=[ThreadingHTTPServer(("127.0.0.1",0),handler(broken)) for broken in (True,False)]
        threads=[threading.Thread(target=server.serve_forever,daemon=True) for server in servers]
        for thread in threads:thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="pip-local-recovery-") as folder:
                root=Path(folder)
                code=install(sys.executable,["--no-deps","--dest",str(root/"wheels"),"shore-pip-fixture==1.0.0"],
                    index_url=f"http://127.0.0.1:{servers[0].server_port}/simple",trusted_host="127.0.0.1",
                    fallback_index_url=f"http://127.0.0.1:{servers[1].server_port}/simple",operation="download",log_dir=root/"logs")
                self.assertEqual(code,0)
                self.assertEqual(hashlib.sha256((root/"wheels"/name).read_bytes()).hexdigest(),digest)
                self.assertIn((True,"/packages/"+name),requests)
                self.assertIn((False,"/packages/"+name),requests)
                receipt=json.loads(next((root/"logs").glob("pip-index-*/receipt.json")).read_text(encoding="utf-8"))
                self.assertTrue(receipt["fallback_used"])
        finally:
            for server in servers:server.shutdown();server.server_close()
            for thread in threads:thread.join(timeout=5)


if __name__ == "__main__":unittest.main()
