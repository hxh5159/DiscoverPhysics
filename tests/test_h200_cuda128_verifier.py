"""Offline verifier wiring/inventory checks; no Torch, GPU or model is loaded."""
from contextlib import nullcontext, redirect_stdout
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import verify_h200_cuda128 as verifier


class H200VerifierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.cache_dirs = {}
        for field, name in (("flashinfer_cubin_dir", "cubins"),
                            ("flashinfer_aot_dir", "aot")):
            directory = self.root / name
            directory.mkdir()
            self.cache_dirs[field] = str(directory)
        self.cache = {"schema_version": "1.0.0", "status": "prepared",
                      "flashinfer_version": "0.6.12", **self.cache_dirs, "files": []}
        for index, directory in enumerate(self.cache_dirs.values()):
            path = Path(directory) / ("test" + str(index) + ".bin")
            data = b"local synthetic kernel inventory" + bytes([index])
            path.write_bytes(data)
            self.cache["files"].append({"path": str(path), "size": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest()})
        self.manifest = {"schema_version": "1.0.0", "target": "h200", "target_cuda_arch": "9.0",
                         "torch_cuda": "12.8", "build_host": "cpu", "cpu_stage_completed": True,
                         "gpu_validation_performed": False, "model_loaded": False,
                         "installed_versions": {"torch": "2.11.0+cu128", "vllm": "0.24.0+cu128",
                                                "transformers": "5.8.0"},
                         "cuda_home": str(self.root / "cuda"), "cxx": str(self.root / "g++"),
                         "cache_environment": {key: str(self.root / key) for key in verifier.CACHE_KEYS},
                         "offline_cache_manifest": "offline-kernel-cache.json"}
        self.interpreter = self.root / ".venv-vllm-cu128/bin/python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text("not executed by unit tests", encoding="utf-8")
        self.save()

    def save(self):
        (self.root / verifier.MANIFEST_NAME).write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.root / "offline-kernel-cache.json").write_text(json.dumps(self.cache), encoding="utf-8")

    def invoke(self, *args, probe=None):
        if probe is None:
            probe = {"passed": True, "gpu_probe_attempted": True, "gpu_compute_passed": True,
                     "checks": [{"name": "simulated_h200_probe", "passed": True}]}
        completed = subprocess.CompletedProcess([], 0, json.dumps(probe), "")
        output = io.StringIO()
        with patch.object(verifier.subprocess, "run", return_value=completed) as run:
            with redirect_stdout(output):
                code = verifier.main(["--env-root", str(self.root), *args])
        return code, json.loads(output.getvalue()), run

    def test_plan_never_executes_target_python_even_without_image(self):
        code, report, run = self.invoke("--plan")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "plan")
        run.assert_not_called()
        self.assertFalse(report["gpu_probe_attempted"])

    def test_help_does_not_launch_a_probe(self):
        with patch.object(verifier.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                verifier.main(["--help"])
        self.assertEqual(exited.exception.code, 0)
        run.assert_not_called()

    def test_manifest_requires_completed_cpu_build_and_real_boolean_flags(self):
        for field, value in (("cpu_stage_completed", False), ("cpu_stage_completed", 1),
                             ("gpu_validation_performed", True), ("model_loaded", True),
                             ("target", "4090"), ("target_cuda_arch", "8.9"),
                             ("torch_cuda", "12.9")):
            with self.subTest(field=field, value=value):
                changed = {**self.manifest, field: value}
                with self.assertRaises(ValueError):
                    verifier.build_config(self.root, changed)

    def test_inventory_verifies_both_cache_roots_and_detects_same_size_corruption(self):
        before = verifier.verify_cache_inventory(self.root, self.manifest)
        self.assertTrue(before["passed"])
        self.assertEqual(before["file_count"], 2)
        file = Path(self.cache["files"][0]["path"])
        file.write_bytes(b"X" * file.stat().st_size)
        after = verifier.verify_cache_inventory(self.root, self.manifest)
        self.assertFalse(after["passed"])
        self.assertEqual(after["status"], "corrupt_or_missing")
        self.assertEqual(after["failures"][0]["error"], "SHA256 mismatch")

    def test_missing_or_empty_inventory_is_explicitly_not_prepared(self):
        missing = verifier.verify_cache_inventory(self.root, {**self.manifest, "offline_cache_manifest": "missing.json"})
        self.assertEqual(missing["status"], "not_prepared")
        self.cache["files"] = []
        self.save()
        empty = verifier.verify_cache_inventory(self.root, self.manifest)
        self.assertFalse(empty["passed"])
        self.assertEqual(empty["status"], "not_prepared")

    def test_offline_inventory_rejects_other_flashinfer_versions(self):
        for version in (None, "0.6.11", "0.6.12.post1", "0.6.13"):
            with self.subTest(version=version):
                self.cache["flashinfer_version"] = version
                self.save()
                with self.assertRaisesRegex(ValueError, "FlashInfer 0.6.12"):
                    verifier.verify_cache_inventory(self.root, self.manifest)

    def test_independent_toolchain_restoration_replaces_stale_compilers_and_stub_paths(self):
        self.manifest["cc"] = str(self.root / "gcc")
        config = verifier.build_config(self.root, self.manifest)
        cuda = Path(self.manifest["cuda_home"])
        libraries = [cuda / "lib64", cuda / "lib", cuda / "targets/x86_64-linux/lib"]
        for directory in libraries:
            directory.mkdir(parents=True, exist_ok=True)
        inherited = self.root / "host-driver-libs"
        stubs = cuda / "lib64/stubs"
        previous = {"PATH": str(self.root / "previous-bin"), "CC": "/unrelated/gcc",
                    "CXX": "/unrelated/g++", "CUDACXX": "/wrong/nvcc",
                    "CUDAHOSTCXX": "/wrong/c++", "NVCC_CCBIN": "/wrong/c++",
                    "LD_LIBRARY_PATH": os.pathsep.join(map(str, (stubs, inherited, libraries[0])))}
        environment = verifier.restore_toolchain_environment(previous, config, str(cuda))
        self.assertEqual(environment["CC"], self.manifest["cc"])
        for field in ("CXX", "CUDAHOSTCXX", "NVCC_CCBIN"):
            self.assertEqual(environment[field], self.manifest["cxx"])
        self.assertEqual(environment["CUDACXX"], str(cuda / "bin/nvcc"))
        self.assertEqual(environment["TRITON_PTXAS_PATH"], str(cuda / "bin/ptxas"))
        self.assertEqual(environment["LD_LIBRARY_PATH"].split(os.pathsep),
                         [*map(str, libraries), str(inherited)])
        self.assertEqual(previous["CC"], "/unrelated/gcc")

    def test_old_manifest_without_cc_only_uses_an_existing_gcc_sibling(self):
        missing = verifier.build_config(self.root, self.manifest)
        self.assertIsNone(missing["cc"])
        environment = verifier.restore_toolchain_environment({"CC": "/unrelated/gcc"},
                                                              missing, self.manifest["cuda_home"])
        self.assertNotIn("CC", environment)
        gcc = self.root / "gcc"
        gcc.write_text("synthetic compiler path; never executed", encoding="utf-8")
        gcc.chmod(0o755)
        existing = verifier.build_config(self.root, self.manifest)
        self.assertEqual(existing["cc"], str(gcc))

    def test_gpu_probe_loop_rejects_small_h200_mig_devices_and_reports_memory(self):
        # Execute the actual embedded device-check loop with a tiny fake Torch
        # interface. This checks the branch without loading Torch or a GPU.
        loops = [node for node in ast.walk(ast.parse(verifier.PROBE_CODE))
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                 and node.target.id == "index"]
        self.assertEqual(len(loops), 1)
        program = compile(ast.Module(body=loops, type_ignores=[]), "<simulated_gpu_loop>", "exec")

        class Tensor:
            def __matmul__(self, other):
                return 128

        for size, expected in ((18, False), (129, False), (130, True), (140, True)):
            with self.subTest(total_memory_gib=size):
                checks = []
                cuda = SimpleNamespace(
                    get_device_name=lambda _: "NVIDIA H200 MIG" if size < 130 else "NVIDIA H200",
                    get_device_capability=lambda _: (9, 0),
                    get_device_properties=lambda _: SimpleNamespace(total_memory=size * 1024 ** 3),
                    device=lambda _: nullcontext(), is_bf16_supported=lambda: True,
                    synchronize=lambda _: None)
                torch = SimpleNamespace(cuda=cuda, bfloat16=object(),
                                        ones=lambda *a, **kw: Tensor(),
                                        all=lambda value: SimpleNamespace(item=lambda: value))
                namespace = {"count": 4, "torch": torch, "gpu_success": True,
                             "check": lambda name, passed, **kw: checks.append(
                                 {"name": name, "passed": passed, **kw}),
                             "fail_text": lambda error: str(error)}
                exec(program, namespace)
                self.assertEqual(namespace["gpu_success"], expected)
                devices = [check for check in checks if check["name"].startswith("h200_device_")]
                self.assertEqual(len(devices), 4)
                self.assertTrue(all(check["total_memory_gib"] == size for check in devices))
                self.assertTrue(all(check["passed"] is expected for check in devices))

    def test_ptxas_path_and_version_are_explicitly_checked_instead_of_silent_fallback(self):
        custom = str(self.root / "toolchain/ptxas")
        self.manifest["triton_ptxas"] = custom
        config = verifier.build_config(self.root, self.manifest)
        self.assertEqual(config["ptxas"], custom)
        environment = verifier.restore_toolchain_environment({}, config, self.manifest["cuda_home"])
        self.assertEqual(environment["TRITON_PTXAS_PATH"], custom)
        loops = [node for node in ast.walk(ast.parse(verifier.PROBE_CODE))
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                 and [entry.id for entry in node.target.elts] == ["label", "command"]]
        self.assertEqual(len(loops), 1)
        program = compile(ast.Module(body=loops, type_ignores=[]), "<simulated_compiler_probe>", "exec")
        for version, expected in (("12.8", True), ("12.9", False), ("13.0", False)):
            with self.subTest(version=version):
                checks = []
                process = subprocess.CompletedProcess([], 0, "Cuda compilation tools, release " + version + ", V" + version, "")
                namespace = {"compiler_commands": [("ptx_assembler_12_8", [custom, "--version"])],
                             "subprocess": SimpleNamespace(run=lambda *a, **kw: process),
                             "re": __import__("re"),
                             "check": lambda name, passed, **kw: checks.append(
                                 {"name": name, "passed": passed, **kw}),
                             "fail_text": lambda error: str(error)}
                exec(program, namespace)
                self.assertEqual(checks[0]["passed"], expected)

    def test_inventory_cannot_verify_an_unrelated_file_or_duplicate_a_record(self):
        self.cache["files"][0]["path"] = str(self.root / "outside-cache.bin")
        self.save()
        with self.assertRaises(ValueError):
            verifier.verify_cache_inventory(self.root, self.manifest)
        self.cache["files"] = [self.cache["files"][1], copy.deepcopy(self.cache["files"][1])]
        self.save()
        with self.assertRaises(ValueError):
            verifier.verify_cache_inventory(self.root, self.manifest)

    def test_single_probe_forces_offline_and_target_environment(self):
        code, report, run = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed_basic_offline_checks")
        self.assertFalse(report["model_loaded"])
        self.assertTrue(report["unverified"])
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], str(self.interpreter))
        self.assertEqual(argv[1:3], ["-I", "-c"])
        environment = run.call_args.kwargs["env"]
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(environment[name], "1")
        self.assertEqual(environment["FLASHINFER_CUDA_ARCH_LIST"], "9.0a")
        self.assertEqual(environment["TVM_FFI_CUDA_ARCH_LIST"], "9.0a")
        self.assertEqual(environment["FLASHINFER_NO_DOWNLOAD"], "1")
        self.assertEqual(environment["VLLM_NO_USAGE_STATS"], "1")
        self.assertNotIn("PYTHONPATH", environment)
        config = json.loads(argv[-1])
        self.assertEqual(config["flashinfer_version"], "0.6.12")
        self.assertEqual(config["flashinfer_cache_directories"], self.cache_dirs)
        compile(verifier.PROBE_CODE, "<offline_h200_probe>", "exec")

    def test_gpu_unavailable_and_unattempted_reports_cannot_pass(self):
        for attempted, compute in ((False, False), (True, False)):
            probe = {"passed": True, "gpu_probe_attempted": attempted,
                     "gpu_compute_passed": compute, "checks": []}
            code, report, _ = self.invoke(probe=probe)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")

    def test_corrupt_cache_does_not_pass_with_successful_gpu_probe(self):
        Path(self.cache["files"][0]["path"]).unlink()
        code, report, _ = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checks"][1]["status"], "corrupt_or_missing")

    def test_incomplete_build_fails_before_target_execution(self):
        self.manifest["cpu_stage_completed"] = False
        self.save()
        code, report, run = self.invoke()
        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn("cpu_stage_completed", report["error"])


if __name__ == "__main__":
    unittest.main()
