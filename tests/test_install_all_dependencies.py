"""Check the distributed full installer against its actual matching helpers.

No dependencies are installed. Shell tests execute only offline plan/argument
paths; archive download and CUDA compiler behavior have separate tests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/install_all_dependencies.sh'
REQUIRED = {
    'bootstrap_cuda128.sh', 'prepare_h200_offline_cache.py',
    'verify_h200_cuda128.py', 'pip_index_install.py',
    'preflight_cuda128_build.py', 'prepare_vllm_dependencies.py',
}


def shell_path(path):
    value = Path(path).absolute().as_posix()
    if os.name == 'nt' and value[1:3] == ':/':
        return '/' + value[0].lower() + value[2:]
    return value


class InstallerHelperWiringTests(unittest.TestCase):
    def test_every_matching_helper_is_fetched_with_its_current_raw_sha(self):
        source = SCRIPT.read_text(encoding='utf-8')
        entries = re.findall(r'^fetch_helper ([\w.]+) ([a-f0-9]{64})$', source, re.M)
        self.assertEqual(set(name for name, _ in entries), REQUIRED)
        self.assertEqual(len(entries), len(REQUIRED))
        for name, expected in entries:
            # GitHub raw files are LF even in a Windows CRLF checkout.
            raw = (ROOT / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, name)
        offline = dict(entries)['prepare_h200_offline_cache.py']
        self.assertIn("printf '%s  %s\\n' " + offline, source)

    def test_cpu_wrapper_downloads_the_same_transitive_helpers(self):
        installer = SCRIPT.read_text(encoding='utf-8')
        wrapper = (ROOT / 'scripts/prepare_cpu_h200.sh').read_text(encoding='utf-8')
        revision = re.search(r"^SOURCE_REF='([a-f0-9]{40})'$", installer, re.M).group(1)
        self.assertIn("DEPENDENCY_SOURCE_REF='" + revision + "'", wrapper)
        for name in REQUIRED:
            self.assertIn(name, wrapper)


class FullInstallerOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name == 'nt':
            candidate = Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Git/bin/bash.exe'
            cls.bash = str(candidate) if candidate.is_file() else None
        else:
            cls.bash = shutil.which('bash')
        if cls.bash is None:
            raise unittest.SkipTest('Bash is needed for shell checks')

    def test_plan_resume_does_not_create_roots_or_run_installation(self):
        with tempfile.TemporaryDirectory(prefix='.full-installer-check-', dir=ROOT) as folder:
            base = Path(folder)
            paths = [base / name for name in ('env', 'toolchain', 'helpers')]
            command = [self.bash, shell_path(SCRIPT), '--resume', '--plan']
            for flag, path in zip(('--env-root', '--tool-root', '--scripts-dir'), paths):
                command.extend((flag, shell_path(path)))
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(any(path.exists() for path in paths))
            self.assertIn('2.11+cu128', result.stdout)
            self.assertIn('SHA256', result.stdout)

    def test_resume_is_explicit_and_helper_upgrade_preserves_old_directory(self):
        result = subprocess.run([self.bash, shell_path(SCRIPT), '--help'], capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--resume', result.stdout)
        self.assertIn('/opt/shore-image-tools-v3', result.stdout)


if __name__ == '__main__':
    unittest.main()
