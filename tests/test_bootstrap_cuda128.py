"""Offline checks for the CUDA bootstrap's argument and filesystem boundary.

These tests run only the plan/argument paths. They never install a package,
initialize CUDA, or download a model.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_cuda128.sh"


def find_bash() -> str | None:
    if os.name == "nt":
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Git" / "bin" / "bash.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs" / "Git" / "bin" / "bash.exe",
        ):
            if candidate.is_file():
                return str(candidate)
        # Windows' system32/bash.exe is a WSL launcher, not necessarily an
        # installed shell, so do not make it a test prerequisite.
        return None
    return shutil.which("bash")


def shell_path(path: Path) -> str:
    value = path.absolute().as_posix()
    if os.name == "nt" and len(value) > 2 and value[1:3] == ":/":
        return "/" + value[0].lower() + value[2:]
    return value


class BootstrapCuda128PathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("Bash is required for offline shell checks")
        if not SCRIPT.is_file():
            raise RuntimeError(f"Bootstrap script is missing: {SCRIPT}")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".bootstrap-cu128-test-", dir=ROOT)
        self.base = Path(self.temporary.name)
        # Keep every fixture and recursive cleanup within this checkout.
        self.assertEqual(self.base.resolve().parent, ROOT.resolve())
        self.addCleanup(self.temporary.cleanup)
        self.bin = self.base / "poisoned-tools"
        self.bin.mkdir()
        self.calls = self.base / "unexpected-tool-calls.txt"
        for name in (
            "uv", "git", "gcc", "g++", "nvcc", "nvidia-smi",
            "python", "python3", "python3.12", "pip", "curl", "wget",
        ):
            stub = self.bin / name
            stub.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' {shlex.quote(name)} >> "
                f"{shlex.quote(shell_path(self.calls))}\n"
                "exit 97\n",
                encoding="utf-8",
                newline="\n",
            )
            stub.chmod(0o755)

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        # Set PATH inside Bash so native Windows PATH separators do not hide
        # the poisoned tools when these tests run in Git Bash.
        command = [
            self.bash, "-c",
            'PATH="$1:$PATH"; export PATH; exec "$BASH" "$2" "${@:3}"',
            "bootstrap-offline-test", shell_path(self.bin), shell_path(SCRIPT),
            *arguments,
        ]
        environment = os.environ.copy()
        environment["CUDA_HOME"] = shell_path(self.base / "missing-cuda")
        environment["SHORE_CUDA128_HOME"] = shell_path(
            self.base / "missing-cuda-override"
        )
        return subprocess.run(
            command, cwd=ROOT, env=environment, text=True,
            encoding="utf-8", errors="replace", capture_output=True,
            timeout=15, check=False,
        )

    def assert_no_tools(self) -> None:
        self.assertFalse(
            self.calls.exists(),
            self.calls.read_text(encoding="utf-8") if self.calls.exists() else "",
        )

    def assert_rejected(self, *arguments: str) -> None:
        result = self.invoke(*arguments)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_tools()

    def make_venv_structure(self, location: Path) -> None:
        location.mkdir(parents=True, exist_ok=True)
        (location / "pyvenv.cfg").write_text("home = /offline-test\n", encoding="utf-8")
        python = location / "bin" / "python"
        python.parent.mkdir()
        # A plan may inspect a venv's structure, but must not execute it.
        shutil.copyfile(self.bin / "python", python)
        python.chmod(0o755)

    def test_plan_for_new_target_is_offline_and_creates_nothing(self) -> None:
        target = self.base / "new environment with spaces"
        before = set(self.base.iterdir())
        result = self.invoke("--env-root", shell_path(target), "--plan")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(target.exists())
        self.assertEqual(set(self.base.iterdir()), before)
        self.assertIn(".venv-trial", result.stdout)
        self.assertIn(".venv-vllm-cu128", result.stdout)
        self.assertIn("2.11.0+cu128", result.stdout)
        self.assertIn("0.24.0", result.stdout)
        self.assert_no_tools()

    def test_plan_preserves_old_environment_results_and_cache(self) -> None:
        target = self.base / "existing"
        for name in (".venv-trial", ".venv-vllm-cu128"):
            self.make_venv_structure(target / name)
        expected = {}
        for relative in (
            ".venv-vllm/keep", ".venv-trial/keep",
            ".venv-vllm-cu128/keep", "results/keep", ".cache/keep",
        ):
            item = target / relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(relative.encode())
            expected[relative] = item.read_bytes()
        before = sorted(p.relative_to(target).as_posix() for p in target.rglob("*"))
        result = self.invoke("--env-root", shell_path(target), "--plan")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = sorted(p.relative_to(target).as_posix() for p in target.rglob("*"))
        self.assertEqual(after, before)
        for relative, content in expected.items():
            self.assertEqual((target / relative).read_bytes(), content)
        self.assert_no_tools()

    def test_invalid_cli_and_unsafe_roots_are_rejected_before_tools(self) -> None:
        target = self.base / "existing"
        target.mkdir()
        for arguments in (
            (), ("--plan",), ("--env-root",),
            ("--env-root", "", "--plan"),
            ("--env-root", "relative/path", "--plan"),
            ("--env-root", "/", "--plan"),
            ("--env-root", shell_path(target) + "/..", "--plan"),
            ("--env-root", shell_path(target) + "/.", "--plan"),
            ("--env-root", shell_path(self.base / "missing" / "child"), "--plan"),
            ("--env-root", shell_path(target), "--unknown"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)

    def test_file_root_is_rejected_and_preserved(self) -> None:
        target = self.base / "ordinary-file"
        target.write_bytes(b"preserve this file")
        self.assert_rejected("--env-root", shell_path(target), "--plan")
        self.assertEqual(target.read_bytes(), b"preserve this file")

    def test_install_requires_explicit_package_source_before_writes(self) -> None:
        target = self.base / "must-not-be-created"
        self.assert_rejected("--env-root", shell_path(target))
        self.assertFalse(target.exists())

    def test_existing_user_activation_file_is_rejected_and_preserved(self) -> None:
        target = self.base / "existing"
        target.mkdir()
        activation = target / "activate_cuda128.sh"
        activation.write_bytes(b"# user-managed activation\nexport USER_SETTING=keep\n")
        before = activation.read_bytes()
        self.assert_rejected("--env-root", shell_path(target), "--plan")
        self.assertEqual(activation.read_bytes(), before)

    def test_generated_activation_file_allows_an_offline_plan(self) -> None:
        target = self.base / "existing"
        target.mkdir()
        activation = target / "activate_cuda128.sh"
        activation.write_text(
            "# Generated by bootstrap_cuda128.sh.\n# original contents\n",
            encoding="utf-8", newline="\n",
        )
        before = activation.read_bytes()
        result = self.invoke("--env-root", shell_path(target), "--plan")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(activation.read_bytes(), before)
        self.assert_no_tools()

    def test_incomplete_environment_directory_is_rejected_and_preserved(self) -> None:
        target = self.base / "existing"
        incomplete = target / ".venv-vllm-cu128"
        incomplete.mkdir(parents=True)
        sentinel = incomplete / "keep"
        sentinel.write_bytes(b"incomplete environment remains available for inspection")
        self.assert_rejected("--env-root", shell_path(target), "--plan")
        self.assertEqual(
            sentinel.read_bytes(),
            b"incomplete environment remains available for inspection",
        )

    def test_symlinked_environment_is_rejected_without_touching_destination(self) -> None:
        target = self.base / "existing"
        target.mkdir()
        destination = self.base / "other-environment"
        destination.mkdir()
        sentinel = destination / "keep"
        sentinel.write_bytes(b"external environment")
        for name in (".venv-trial", ".venv-vllm-cu128"):
            link = target / name
            try:
                link.symlink_to(destination, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlinks unavailable: {exc}")
            try:
                self.assert_rejected("--env-root", shell_path(target), "--plan")
                self.assertEqual(sentinel.read_bytes(), b"external environment")
            finally:
                link.unlink()


if __name__ == "__main__":
    unittest.main()
