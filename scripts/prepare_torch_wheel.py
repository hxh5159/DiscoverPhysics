#!/usr/bin/env python3
"""Persist and verify the fixed Python3.12/Linux CUDA12.8 Torch wheel.

Downloads only; never imports Torch, calls a GPU or installs a package. Each
curl invocation resumes the durable .part file, including across script runs.
The official cu128 index supplies the pinned artifact identity below:
https://download.pytorch.org/whl/cu128/torch/
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit
from uuid import uuid4


WHEEL_FILENAME = 'torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
WHEEL_SHA256 = 'd252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997'
WHEEL_BYTES = 820326880
OFFICIAL_HOSTS = frozenset(('download.pytorch.org', 'download-r2.pytorch.org'))


@dataclass(frozen=True)
class Artifact:
    filename: str
    sha256: str
    size: int
    urls: tuple[str, ...]


ARTIFACT = Artifact(WHEEL_FILENAME, WHEEL_SHA256, WHEEL_BYTES, tuple(
    'https://' + host + '/whl/cu128/' + WHEEL_FILENAME.replace('+', '%2B')
    for host in ('download-r2.pytorch.org', 'download.pytorch.org')))


class DownloadError(RuntimeError):
    pass


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _validate(artifact):
    if (not artifact.filename.endswith('.whl') or
            any(c in artifact.filename for c in '/\\:\r\n\0') or
            not re.fullmatch(r'[a-f0-9]{64}', artifact.sha256) or
            type(artifact.size) is not int or artifact.size <= 0 or not artifact.urls):
        raise ValueError('Invalid pinned wheel identity')
    for url in artifact.urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_HOSTS or
                parsed.username or parsed.password or parsed.port not in (None, 443) or
                parsed.query or parsed.fragment or
                unquote(parsed.path.rsplit('/', 1)[-1]) != artifact.filename):
            raise ValueError('Only the pinned wheel on official HTTPS PyTorch hosts is allowed')


def _ordinary(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise DownloadError('Refusing non-regular file: ' + str(path))


def _save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def _lock(path):
    _ordinary(path)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise DownloadError('Another wheel download is active') from exc
            try: yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise DownloadError('Another wheel download is active') from exc
            try: yield
            finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _curl_run(command):
    # curl writes the wheel to disk; only its three-digit HTTP status enters RAM.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True,
                               encoding='utf-8', errors='replace')
    try:
        output, _ = process.communicate()
    except BaseException:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise
    return subprocess.CompletedProcess(command, process.returncode, output)


def prepare(output_dir, *, artifact=ARTIFACT, attempts=6, connect_timeout=30,
            low_speed_limit=65536, low_speed_time=120, curl=None, runner=None,
            sleep=time.sleep, verify_only=False):
    """Download the pinned artifact; test callers may inject a tiny artifact/runner.

    The public CLI provides no arbitrary URL, version or SHA override. Redirects
    are not followed, so curl cannot silently leave the approved origin.
    """
    _validate(artifact)
    for value in (attempts, connect_timeout, low_speed_time):
        if type(value) is not int or value <= 0: raise ValueError('Limits must be positive integers')
    if type(low_speed_limit) is not int or low_speed_limit < 0:
        raise ValueError('The low-speed limit must be a nonnegative integer')
    root = Path(output_dir).expanduser().absolute()
    if root.is_symlink(): raise DownloadError('Use the actual output directory, not a symlink')
    root.mkdir(parents=True, exist_ok=True)
    final, partial = root/artifact.filename, root/(artifact.filename + '.part')
    for path in (final, partial): _ordinary(path)
    receipt = root/('torch-wheel-download-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    + '-' + uuid4().hex[:10] + '.json')
    record = {'schema_version': 'shore-pinned-torch-wheel-1.0', 'status': 'failed',
              'wheel': str(final), 'partial_file': str(partial), 'sha256': artifact.sha256,
              'expected_bytes': artifact.size, 'attempts': [], 'receipt_path': str(receipt),
              'packages_installed': False, 'gpu_or_model_calls': 0}

    def complete(path):
        return path.is_file() and path.stat().st_size == artifact.size and sha256_file(path) == artifact.sha256

    def publish(status):
        record.update(status=status, bytes=artifact.size, all_bytes_sha256_verified=True)
        _save(receipt, record)
        return record

    with _lock(root/(artifact.filename + '.lock')):
        try:
            if final.exists():
                if not complete(final):
                    raise DownloadError('Existing wheel does not match the pin; it was preserved: ' + str(final))
                return publish('reused')
            if verify_only: raise DownloadError('The verified wheel is not present: ' + str(final))
            if partial.exists() and partial.stat().st_size >= artifact.size:
                if not complete(partial):
                    raise DownloadError('The retained .part has the wrong size/SHA; inspect it or choose a new output directory')
                partial.replace(final)
                return publish('prepared')
            executable = curl or shutil.which('curl')
            if not executable: raise DownloadError('curl is required for persistent Range downloads')
            execute = runner or _curl_run
            for number in range(1, attempts + 1):
                url = artifact.urls[(number - 1) % len(artifact.urls)]
                offset = partial.stat().st_size if partial.exists() else 0
                print(f'[SHORE Torch] Attempt {number}/{attempts}; resume {offset}/{artifact.size} bytes; {url}', flush=True)
                command = [str(executable), '--fail', '--show-error', '--progress-bar',
                    '--proto', '=https', '--continue-at', '-', '--retry', '0',
                    '--connect-timeout', str(connect_timeout), '--speed-limit', str(low_speed_limit),
                    '--speed-time', str(low_speed_time), '--output', str(partial),
                    '--write-out', '%{http_code}', url]
                result = execute(command)
                http_status = (result.stdout or '').strip()
                size = partial.stat().st_size if partial.exists() else 0
                record['attempts'].append({'number': number, 'url': url, 'resume_from_bytes': offset,
                    'curl_exit_code': result.returncode, 'http_status': http_status, 'retained_bytes': size})
                _save(receipt, record)
                if size >= artifact.size:
                    if result.returncode != 0 or http_status not in ('200', '206'):
                        raise DownloadError('Transfer failed despite a full-sized .part; retained bytes require verification on the next run')
                    if not complete(partial):
                        raise DownloadError('Downloaded bytes failed pinned size/SHA verification; .part was preserved')
                    # Hash verification, rather than a successful curl status alone,
                    # is the prerequisite for publishing a usable wheel.
                    partial.replace(final)
                    return publish('prepared')
                if result.returncode == 0 and http_status not in ('200', '206'):
                    raise DownloadError('The server did not return a file/Range response; .part was preserved')
                if number < attempts: sleep(min(number * 2, 10))
            raise DownloadError('Download remains incomplete; rerun this same command to resume ' + str(partial))
        except BaseException as exc:
            record.update(error=type(exc).__name__ + ': ' + str(exc),
                          retained_bytes=partial.stat().st_size if partial.exists() else 0)
            _save(receipt, record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/opt/shore-h200-cu128/wheelhouse'))
    parser.add_argument('--attempts', type=int, default=6)
    parser.add_argument('--connect-timeout', type=int, default=30)
    parser.add_argument('--low-speed-limit', type=int, default=65536,
                        help='try the next official URL if speed stays below this many bytes/s (0 disables)')
    parser.add_argument('--low-speed-time', type=int, default=120)
    parser.add_argument('--verify-only', action='store_true', help='verify an existing wheel without invoking curl')
    args = parser.parse_args(argv)
    try:
        value = prepare(args.output_dir, attempts=args.attempts, connect_timeout=args.connect_timeout,
            low_speed_limit=args.low_speed_limit, low_speed_time=args.low_speed_time, verify_only=args.verify_only)
    except (OSError, ValueError, DownloadError) as exc:
        print('[SHORE Torch] ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
