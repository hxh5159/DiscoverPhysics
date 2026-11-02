"""Verify a CPU-built CUDA 12.8 environment on an offline four-H200 node.

This standard-library launcher never installs packages or starts a model.
--plan and --help do not import Torch, inspect a GPU, or execute a subprocess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MANIFEST_NAME = "cuda128-build-manifest.json"
UNVERIFIED = [
    "No model weights, tokenizer or model-specific execution were loaded.",
    "Four-GPU NCCL/model tensor parallelism and FP8/MoE kernels were not exercised.",
    "Cache inventory hashes do not prove that every model-specific JIT kernel is present.",
    "Project SDK/source integrity is outside this build/cache manifest.",
    "No generation, competition agent, or physics evaluation was run.",
]
CACHE_KEYS = ("HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
              "FLASHINFER_WORKSPACE_BASE")
FLASHINFER_VERSION = "0.6.12"


# Exactly one target-environment Python invocation returns this JSON probe.
# Package logging is captured rather than copied to the public result.
PROBE_CODE = r'''
import contextlib
import importlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile

config = json.loads(sys.argv[1])
checks = []
result = {"checks": checks, "gpu_probe_attempted": False,
          "gpu_compute_passed": False, "network_disabled_in_python": True}

def check(name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})

def fail_text(error):
    return type(error).__name__ + ": " + str(error)[:400]

def no_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled during offline verification")

# Offline environment variables are the normal library mechanism. Also deny
# Python socket/DNS access in this probe so an import cannot fetch a kernel.
socket.create_connection = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.getaddrinfo = no_network

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    check("linux_runtime", platform.system() == "Linux")
    check("huggingface_offline", all(os.environ.get(k) == "1" for k in
          ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")))
    for key, directory in config["cache_environment"].items():
        try:
            path = Path(directory)
            if not path.is_dir():
                raise ValueError("Prepared cache directory is missing")
            with tempfile.TemporaryFile(dir=path) as sample:
                sample.write(b"shore-offline-cache-write-check")
                sample.flush()
            check("writable_cache_" + key, True, path=str(path))
        except Exception as exc:
            check("writable_cache_" + key, False, error=fail_text(exc))
    for package, expected in config["installed_versions"].items():
        try:
            actual = md.version(package)
            check("version_" + package, actual == expected,
                  expected=expected, actual=actual)
        except Exception as exc:
            check("version_" + package, False, error=fail_text(exc))
    if config.get("flashinfer_version"):
        try:
            actual = md.version("flashinfer-python")
            check("version_flashinfer_cache", actual == config["flashinfer_version"],
                  expected=config["flashinfer_version"], actual=actual)
        except Exception as exc:
            check("version_flashinfer_cache", False, error=fail_text(exc))
    compiler_commands = [
        ("cuda_compiler_12_8", [config["nvcc"], "--version"]),
        ("ptx_assembler_12_8", [config["ptxas"], "--version"]),
        ("cpp_compiler", [config["cxx"], "--version"]),
    ]
    if config.get("cc"):
        compiler_commands.append(("c_compiler", [config["cc"], "--version"]))
    for label, command in compiler_commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if label in ("cuda_compiler_12_8", "ptx_assembler_12_8"):
                passed = completed.returncode == 0 and bool(re.search(
                    r"release\s+12\.8(?:[\s,]|$)", completed.stdout))
            else:
                passed = completed.returncode == 0
            check(label, passed, executable=command[0])
        except Exception as exc:
            check(label, False, error=fail_text(exc))
    try:
        pip_result = subprocess.run([sys.executable, "-m", "pip", "check"],
                                    capture_output=True, text=True, timeout=90)
        check("pip_dependency_consistency", pip_result.returncode == 0,
              output=pip_result.stdout.strip()[:2000])
    except Exception as exc:
        check("pip_dependency_consistency", False, error=fail_text(exc))
    torch = None
    try:
        import torch
        check("torch_import", True)
        check("torch_cuda_12_8", torch.version.cuda == "12.8",
              actual=torch.version.cuda)
        result["gpu_probe_attempted"] = True
        available = bool(torch.cuda.is_available())
        count = torch.cuda.device_count() if available else 0
        check("cuda_driver_initialization", available,
              error=None if available else "CUDA initialization unavailable; check allocated GPUs and host driver")
        check("four_visible_gpus", available and count == 4, actual=count)
        gpu_success = available and count == 4
        for index in range(count):
            try:
                name = torch.cuda.get_device_name(index)
                capability = tuple(torch.cuda.get_device_capability(index))
                memory_bytes = int(torch.cuda.get_device_properties(index).total_memory)
                memory_gib = memory_bytes / (1024 ** 3)
                is_h200 = ("H200" in name.upper() and capability == (9, 0)
                           and memory_bytes >= 130 * (1024 ** 3))
                check("h200_device_" + str(index), is_h200,
                      name_on_device=name, compute_capability=list(capability),
                      total_memory_gib=memory_gib, minimum_total_memory_gib=130)
                if not is_h200:
                    gpu_success = False
                    continue
                with torch.cuda.device(index):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("BF16 support was not reported")
                    x = torch.ones((128, 128), device="cuda:" + str(index), dtype=torch.bfloat16)
                    product = x @ x
                    torch.cuda.synchronize(index)
                    correct = bool(torch.all(product == 128).item())
                    check("bf16_matrix_compute_" + str(index), correct)
                    gpu_success = gpu_success and correct
                    del product, x
            except Exception as exc:
                check("gpu_execution_" + str(index), False, error=fail_text(exc))
                gpu_success = False
        result["gpu_compute_passed"] = gpu_success
    except Exception as exc:
        check("torch_or_cuda_runtime", False, error=fail_text(exc))
    for module in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        try:
            importlib.import_module(module)
            check("import_" + module, True)
        except Exception as exc:
            check("import_" + module, False, error=fail_text(exc))
    if config.get("flashinfer_cache_directories"):
        try:
            from flashinfer.jit import env as flashinfer_env
            for field, attribute in (("flashinfer_cubin_dir", "FLASHINFER_CUBIN_DIR"),
                                     ("flashinfer_aot_dir", "FLASHINFER_AOT_DIR")):
                actual = str(Path(getattr(flashinfer_env, attribute)).resolve())
                expected = str(Path(config["flashinfer_cache_directories"][field]).resolve())
                check("active_" + field, actual == expected, actual=actual, expected=expected)
        except Exception as exc:
            check("flashinfer_cache_directory_resolution", False, error=fail_text(exc))
    try:
        from vllm.reasoning import ReasoningParserManager
        ReasoningParserManager.get_reasoning_parser("qwen3")
        check("qwen3_reasoning_parser_import", True)
    except Exception as exc:
        check("qwen3_reasoning_parser_import", False, error=fail_text(exc))

result["passed"] = all(c["passed"] for c in checks)
print(json.dumps(result, ensure_ascii=True, allow_nan=False))
'''


def load_json(path):
    with path.open(encoding="utf-8-sig") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("JSON manifest must be an object")
    return value


def build_config(env_root, manifest):
    expected = {"schema_version": "1.0.0", "target": "h200", "torch_cuda": "12.8",
                "build_host": "cpu", "cpu_stage_completed": True,
                "gpu_validation_performed": False, "model_loaded": False}
    for name, value in expected.items():
        if type(manifest.get(name)) is not type(value) or manifest[name] != value:
            raise ValueError("Build manifest field is missing or incompatible: " + name)
    if str(manifest.get("target_cuda_arch")) != "9.0":
        raise ValueError("Build manifest target_cuda_arch must be 9.0")
    versions = manifest.get("installed_versions")
    if not isinstance(versions, dict) or any(
            not isinstance(versions.get(name), str) or not versions[name]
            for name in ("torch", "vllm", "transformers")):
        raise ValueError("Build manifest must declare Torch, vLLM and Transformers versions")
    caches = manifest.get("cache_environment")
    if not isinstance(caches, dict) or any(not isinstance(caches.get(key), str)
                                          or not Path(caches[key]).is_absolute()
                                          for key in CACHE_KEYS):
        raise ValueError("Build manifest requires four absolute cache_environment paths")
    cuda_home = manifest.get("cuda_home")
    if not isinstance(cuda_home, str) or not Path(cuda_home).is_absolute():
        raise ValueError("Build manifest requires an absolute cuda_home containing bin/nvcc")
    ptxas = manifest.get("triton_ptxas", str(Path(cuda_home) / "bin/ptxas"))
    if not isinstance(ptxas, str) or not Path(ptxas).is_absolute():
        raise ValueError("Build manifest triton_ptxas must be an absolute executable path")
    cxx = manifest.get("cxx")
    if not isinstance(cxx, str) or not Path(cxx).is_absolute():
        raise ValueError("Build manifest requires an absolute cxx executable")
    cc = manifest.get("cc")
    if cc is not None and (not isinstance(cc, str) or not Path(cc).is_absolute()):
        raise ValueError("Build manifest cc, when declared, must be an absolute executable path")
    if cc is None:
        # Older manifests only declared CXX. Infer a sibling GCC only when it
        # actually exists; never retain an unrelated inherited CC silently.
        for compiler in (Path(cxx), Path(cxx).resolve()):
            gcc_name, replaced = re.subn(r"g\+\+(?=(?:-\d+(?:\.\d+)*)?$)",
                                         "gcc", compiler.name)
            candidate = compiler.with_name(gcc_name)
            if replaced and candidate.is_file() and os.access(candidate, os.X_OK):
                cc = str(candidate)
                break
    return {"installed_versions": {k: versions[k] for k in ("torch", "vllm", "transformers")},
            "cache_environment": {key: caches[key] for key in CACHE_KEYS +
                                  tuple(k for k in ("DG_JIT_CACHE_DIR", "TVM_FFI_CACHE_DIR")
                                        if isinstance(caches.get(k), str) and Path(caches[k]).is_absolute())},
            "nvcc": str(Path(cuda_home) / "bin/nvcc"), "ptxas": ptxas,
            "cxx": cxx, "cc": cc}


def restore_toolchain_environment(environment, config, cuda_home):
    """Restore the prepared compiler/runtime paths without activation helpers."""
    environment = environment.copy()
    environment.update(CUDA_HOME=cuda_home, CUDACXX=config["nvcc"],
                       CXX=config["cxx"], CUDAHOSTCXX=config["cxx"],
                       NVCC_CCBIN=config["cxx"], TRITON_PTXAS_PATH=config["ptxas"])
    if config.get("cc"):
        environment["CC"] = config["cc"]
    else:
        environment.pop("CC", None)
    executable_dirs = [str(Path(cuda_home) / "bin"), str(Path(config["cxx"]).parent)]
    if config.get("cc"):
        executable_dirs.append(str(Path(config["cc"]).parent))
    environment["PATH"] = os.pathsep.join(dict.fromkeys(
        executable_dirs + [p for p in environment.get("PATH", "").split(os.pathsep) if p]))
    toolkit_libraries = [Path(cuda_home) / "lib64", Path(cuda_home) / "lib",
                         Path(cuda_home) / "targets/x86_64-linux/lib"]
    inherited_libraries = [Path(p) for p in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    library_dirs = []
    for path in [p for p in toolkit_libraries if p.is_dir()] + inherited_libraries:
        if ("stubs" in {part.lower() for part in path.parts}
                or "stubs" in {part.lower() for part in path.resolve().parts}):
            continue
        if str(path) not in library_dirs:
            library_dirs.append(str(path))
    if library_dirs:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return environment


def verify_cache_inventory(env_root, manifest):
    relative = manifest.get("offline_cache_manifest")
    if not isinstance(relative, str) or not relative:
        return {"name": "offline_cache_inventory", "passed": False,
                "status": "not_prepared", "error": "No offline_cache_manifest declared"}
    cache_manifest_path = (env_root / relative).resolve()
    if not cache_manifest_path.is_relative_to(env_root):
        raise ValueError("offline_cache_manifest must remain within env_root")
    if not cache_manifest_path.is_file():
        return {"name": "offline_cache_inventory", "passed": False,
                "status": "not_prepared", "error": "Offline cache manifest is missing"}
    cache = load_json(cache_manifest_path)
    if cache.get("schema_version") != "1.0.0" or cache.get("status") != "prepared":
        return {"name": "offline_cache_inventory", "passed": False,
                "status": "not_prepared", "error": "Offline cache manifest is not prepared"}
    cache_roots = []
    for key in ("flashinfer_cubin_dir", "flashinfer_aot_dir"):
        directory = cache.get(key)
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise ValueError("Offline cache requires an absolute " + key)
        cache_root = Path(directory).resolve()
        if not cache_root.is_relative_to(env_root):
            raise ValueError("Prepared cache directory must remain within env_root")
        if not cache_root.is_dir():
            return {"name": "offline_cache_inventory", "passed": False,
                    "status": "not_prepared", "error": "Prepared cache directory is missing: " + key}
        cache_roots.append(cache_root)
    if cache.get("flashinfer_version") != FLASHINFER_VERSION:
        raise ValueError("Offline cache must use FlashInfer " + FLASHINFER_VERSION)
    files = cache.get("files")
    if not isinstance(files, list) or not files:
        return {"name": "offline_cache_inventory", "passed": False,
                "status": "not_prepared", "error": "Offline cache file inventory is empty"}
    failures, seen, total_size = [], set(), 0
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Offline cache inventory entry requires a path")
        path = Path(item["path"])
        if not path.is_absolute():
            raise ValueError("Offline cache inventory paths must be absolute")
        path = path.resolve()
        if path in seen or not any(path.is_relative_to(root) for root in cache_roots):
            raise ValueError("Offline cache paths must be unique and within a declared FlashInfer directory")
        seen.add(path)
        digest, size = item.get("sha256"), item.get("size")
        if not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None:
            raise ValueError("Offline cache entry requires a SHA256 digest")
        if type(size) is not int or size < 0:
            raise ValueError("Offline cache entry requires a nonnegative byte size")
        try:
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError("Missing file or byte-size mismatch")
            hasher = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest.lower():
                raise ValueError("SHA256 mismatch")
            total_size += size
        except (OSError, ValueError) as exc:
            if len(failures) < 10:
                failures.append({"path": str(path), "error": str(exc)[:200]})
    return {"name": "offline_cache_inventory", "passed": not failures,
            "status": "prepared_inventory_verified" if not failures else "corrupt_or_missing",
            "flashinfer_version": cache.get("flashinfer_version"),
            "directories": dict(zip(("flashinfer_cubin_dir", "flashinfer_aot_dir"),
                                     map(str, cache_roots))),
            "file_count": len(files), "verified_bytes": total_size, "failures": failures,
            "scope": "Only listed cache files were verified; model-specific JIT coverage is untested"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--env-root", type=Path, required=True,
                        help="CPU-built image directory containing the manifest and .venv-vllm-cu128")
    result.add_argument("--plan", action="store_true", help="Print the plan without subprocess or GPU access")
    result.add_argument("--timeout", type=int, default=300, help="Target Python probe timeout in seconds")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    env_root = args.env_root.expanduser().resolve()
    report = {"schema_version": "1.0.0", "env_root": str(env_root), "status": "failed",
              "model_loaded": False, "checks": [], "unverified": UNVERIFIED}
    if args.plan:
        report.update(status="plan", gpu_probe_attempted=False,
                      actions=["Read completed CPU build manifest", "Verify prepared offline cache inventory",
                               "Run one offline target Python probe for CUDA, four H200s, BF16, imports and tools"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.timeout < 1:
        report["error"] = "--timeout must be positive"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    try:
        manifest = load_json(env_root / MANIFEST_NAME)
        config = build_config(env_root, manifest)
        report["checks"].append({"name": "cpu_build_manifest", "passed": True,
                                 "gpu_validation_performed_during_build": False})
        cache_check = verify_cache_inventory(env_root, manifest)
        report["checks"].append(cache_check)
        config["flashinfer_version"] = FLASHINFER_VERSION
        config["flashinfer_cache_directories"] = cache_check.get("directories")
        interpreter = env_root / ".venv-vllm-cu128/bin/python"
        if not interpreter.is_file():
            raise ValueError("Target environment Python is missing: " + str(interpreter))
        environment = os.environ.copy()
        environment.update(config["cache_environment"])
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", VLLM_USE_RUST_FRONTEND="0",
                           FLASHINFER_CUDA_ARCH_LIST="9.0a", TVM_FFI_CUDA_ARCH_LIST="9.0a",
                           FLASHINFER_NO_DOWNLOAD="1", VLLM_NO_USAGE_STATS="1")
        environment = restore_toolchain_environment(environment, config, manifest["cuda_home"])
        # Avoid importing packages from an inherited source tree instead of the
        # installed image. Neither credentials nor unrelated environment values
        # are included in the JSON report.
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([str(interpreter), "-I", "-c", PROBE_CODE,
                                    json.dumps(config, ensure_ascii=True)],
                                   cwd=env_root, env=environment, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
        if completed.returncode != 0:
            raise ValueError("Target Python probe failed; exit code " + str(completed.returncode))
        try:
            probe = json.loads(completed.stdout)
        except (ValueError, TypeError):
            raise ValueError("Target Python probe did not return valid JSON; package stderr was not disclosed") from None
        if not isinstance(probe, dict) or not isinstance(probe.get("checks"), list):
            raise ValueError("Target Python probe returned an invalid result object")
        report["checks"].extend(probe["checks"])
        report["gpu_probe_attempted"] = bool(probe.get("gpu_probe_attempted"))
        report["gpu_compute_passed"] = bool(probe.get("gpu_compute_passed"))
        report["status"] = ("passed_basic_offline_checks" if (
            probe.get("passed") is True and report["gpu_probe_attempted"]
            and report["gpu_compute_passed"] and all(
                item.get("passed") is True for item in report["checks"])) else "failed")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)[:500]
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed_basic_offline_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
