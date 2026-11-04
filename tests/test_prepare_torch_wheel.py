"""Tiny artifact fixtures; no Torch download, installation or GPU calls."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading

import pytest

from scripts.prepare_torch_wheel import Artifact, DownloadError, prepare


PAYLOAD = b'tiny fixed Torch-wheel transport fixture\n' * 2048
NAME = 'shore_torch_fixture-1.0.0-py3-none-any.whl'
ARTIFACT = Artifact(NAME, hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD),
    tuple('https://' + host + '/whl/cu128/' + NAME
          for host in ('download-r2.pytorch.org', 'download.pytorch.org')))


class FakeCurl:
    def __init__(self, results): self.results, self.calls = iter(results), []
    def __call__(self, command):
        self.calls.append(list(command))
        target = Path(command[command.index('--output') + 1])
        code, data, status = next(self.results)
        with target.open('ab') as stream: stream.write(data)
        return subprocess.CompletedProcess(command, code, status)


def run(root, runner, **kwargs):
    return prepare(root, artifact=ARTIFACT, runner=runner, curl='fixture-curl', sleep=lambda _: None, **kwargs)


def test_valid_download_is_hashed_and_published_after_full_size(tmp_path):
    curl = FakeCurl([(0, PAYLOAD, '200')])
    result = run(tmp_path, curl)
    assert result['status'] == 'prepared' and result['all_bytes_sha256_verified']
    assert (tmp_path/NAME).read_bytes() == PAYLOAD
    assert not (tmp_path/(NAME + '.part')).exists()
    assert result['packages_installed'] is False and result['gpu_or_model_calls'] == 0
    assert '--location' not in curl.calls[0]
    assert curl.calls[0][curl.calls[0].index('--proto') + 1] == '=https'


def test_partial_file_survives_failure_and_is_resumed_on_next_run(tmp_path):
    first = PAYLOAD[:4000]
    curl = FakeCurl([(18, first, '200')])
    with pytest.raises(DownloadError, match='incomplete'): run(tmp_path, curl, attempts=1)
    assert (tmp_path/(NAME + '.part')).read_bytes() == first
    result = run(tmp_path, FakeCurl([(0, PAYLOAD[4000:], '206')]), attempts=1)
    assert result['attempts'][0]['resume_from_bytes'] == 4000
    assert (tmp_path/NAME).read_bytes() == PAYLOAD
    assert len(list(tmp_path.glob('torch-wheel-download-*.json'))) == 2


def test_alternate_official_host_keeps_the_partial_bytes(tmp_path):
    result = run(tmp_path, FakeCurl([(18, PAYLOAD[:100], '200'), (0, PAYLOAD[100:], '206')]), attempts=2)
    assert result['attempts'][1]['resume_from_bytes'] == 100
    assert 'download.pytorch.org' in result['attempts'][1]['url']


def test_existing_verified_wheel_needs_no_curl_or_download(tmp_path):
    target = tmp_path/NAME; target.write_bytes(PAYLOAD)
    result = run(tmp_path, FakeCurl([]), verify_only=True)
    assert result['status'] == 'reused'


def test_existing_wrong_wheel_is_preserved(tmp_path):
    target = tmp_path/NAME; target.write_bytes(b'preserve unknown bytes')
    with pytest.raises(DownloadError, match='Existing wheel'): run(tmp_path, FakeCurl([]))
    assert target.read_bytes() == b'preserve unknown bytes'


def test_download_hash_mismatch_keeps_part_and_never_publishes(tmp_path):
    corrupted = b'x' * len(PAYLOAD)
    with pytest.raises(DownloadError, match='SHA'): run(tmp_path, FakeCurl([(0, corrupted, '200')]))
    assert (tmp_path/(NAME + '.part')).read_bytes() == corrupted
    assert not (tmp_path/NAME).exists()
    receipt = json.loads(next(tmp_path.glob('torch-wheel-download-*.json')).read_text(encoding='utf-8'))
    assert receipt['status'] == 'failed'


def test_404_or_redirect_is_never_success(tmp_path):
    for code, status in ((22, '404'), (0, '302')):
        folder = tmp_path/status
        with pytest.raises(DownloadError): run(folder, FakeCurl([(code, b'', status)]), attempts=1)
        assert not (folder/NAME).exists()


def test_404_with_a_full_sized_body_still_fails(tmp_path):
    with pytest.raises(DownloadError, match='Transfer failed'):
        run(tmp_path, FakeCurl([(22, PAYLOAD, '404')]), attempts=1)
    assert not (tmp_path/NAME).exists()
    assert (tmp_path/(NAME + '.part')).read_bytes() == PAYLOAD


@pytest.mark.parametrize('url', ('http://download.pytorch.org/whl/cu128/' + NAME,
    'https://download.pytorch.org.evil/whl/cu128/' + NAME,
    'https://download.pytorch.org/whl/cu128/' + NAME + '?token=hidden'))
def test_unapproved_source_is_rejected_before_writing(tmp_path, url):
    with pytest.raises(ValueError):
        prepare(tmp_path/'untouched', artifact=Artifact(NAME, ARTIFACT.sha256, len(PAYLOAD), (url,)))
    assert not (tmp_path/'untouched').exists()


def test_real_curl_resumes_a_truncated_http_response(tmp_path):
    curl = shutil.which('curl')
    if not curl: pytest.skip('curl is required for the local Range fixture')
    offsets = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def do_GET(self):
            offset = int(self.headers.get('Range', 'bytes=0-')[6:].split('-')[0])
            offsets.append(offset)
            body = PAYLOAD[offset:]
            self.send_response(206 if offset else 200)
            self.send_header('Content-Length', str(len(body)))
            if offset: self.send_header('Content-Range', f'bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}')
            self.end_headers()
            if len(offsets) == 1:
                self.wfile.write(body[:7000]); self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close(); self.close_connection = True
            else: self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    (tmp_path/(NAME + '.part')).write_bytes(PAYLOAD[:4000])
    def local_runner(command):
        # Production URL/protocol guards remain intact. Only this test runner
        # substitutes an explicitly local, plain-HTTP transport fixture.
        argv = list(command)
        assert argv[-1].startswith('https://download-') or argv[-1].startswith('https://download.pytorch.org/')
        argv[argv.index('--proto') + 1] = '=http'
        argv[-1] = f'http://127.0.0.1:{server.server_port}/fixture.whl'
        return subprocess.run(argv, capture_output=True, text=True, timeout=20)
    try:
        result = prepare(tmp_path, artifact=ARTIFACT, attempts=2, curl=curl,
                         runner=local_runner, sleep=lambda _: None)
        assert offsets == [4000, 11000]
        assert result['status'] == 'prepared'
        assert (tmp_path/NAME).read_bytes() == PAYLOAD
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_help_never_imports_torch_or_downloads(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/prepare_torch_wheel.py'
    done = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp_path,
                          capture_output=True, text=True, timeout=15)
    assert done.returncode == 0 and '--output-dir' in done.stdout
    assert not list(tmp_path.iterdir())
