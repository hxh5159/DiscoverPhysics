#!/usr/bin/env python3
"""Check the CUDA 12.8 source-build toolchain on a CPU-only Linux instance.

No GPU is initialized and no model, kernel or compiled executable is run. Each
invocation keeps an independent diagnostic directory, including failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4


HEADERS = ("cuda.h", "cuda_runtime.h", "device_functions.h")
TAG = "SHORE_CUDA_BUILD_INFO="
CUDA_128 = re.compile(r"release\s+12\.8(?:[,\s]|$)")


class PreflightError(RuntimeError):
    pass


def _absolute(path: str | Path) -> Path:
    # A venv's python may be a symlink. Resolving it before execution bypasses
    # pyvenv.cfg and can silently invoke an entirely different environment.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _executable(path: str | Path, name: str) -> str:
    text = str(path)
    found = text if Path(text).is_file() else shutil.which(text)
    if found is None:
        raise PreflightError(f"Missing {name} executable: {text}")
    return str(_absolute(found))


def layout(cuda_home: str | Path) -> dict[str, str]:
    """Locate real flat/Conda target headers and cudart; do not run anything."""
    root = _absolute(cuda_home)
    includes = (root / "targets/x86_64-linux/include", root / "include")
    include = next((p for p in includes if all((p / h).is_file() for h in HEADERS)), None)
    if include is None:
        raise PreflightError("CUDA development headers are incomplete: expected cuda.h, "
                             "cuda_runtime.h and device_functions.h in one include directory under " + str(root))
    libraries = (root / "targets/x86_64-linux/lib", root / "lib64", root / "lib")
    library = next((p for p in libraries if (p / "libcudart.so").is_file()
                    and "stubs" not in (p / "libcudart.so").resolve().parts), None)
    if library is None:
        raise PreflightError("CUDA development runtime library libcudart.so is missing under " + str(root))
    for name in ("nvcc", "ptxas"):
        if not (root / "bin" / name).is_file():
            raise PreflightError(f"CUDA compiler component is missing: {root / 'bin' / name}")
    return {"cuda_home": str(root), "include_dir": str(include), "library_dir": str(library),
            "cudart": str(library / "libcudart.so"), "nvcc": str(root / "bin/nvcc"),
            "ptxas": str(root / "bin/ptxas")}


def build_environment(found: dict[str, str], cc: str, cxx: str,
                      environment: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environment is None else environment)
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="", CUDA_HOME=found["cuda_home"],
               CUDA_INC_PATH=found["include_dir"], CUDA_LIB_PATH=found["library_dir"],
               CUDA_BIN_PATH=found["cuda_home"], CUDA_PATH=found["cuda_home"],
               CUDA_NVCC_EXECUTABLE=found["nvcc"],
               CUDAToolkit_ROOT=found["cuda_home"], CUDACXX=found["nvcc"],
               CUDAHOSTCXX=cxx, NVCC_CCBIN=cxx, CC=cc, CXX=cxx,
               PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8")
    root = Path(found["cuda_home"])
    env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
    inherited = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                 if p and "stubs" not in Path(p).parts]
    real_libraries = [str(p) for p in (Path(found["library_dir"]), root / "lib64", root / "lib") if p.is_dir()]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(real_libraries + inherited))
    prefixes = [str(root / "targets/x86_64-linux"), str(root)]
    if env.get("CMAKE_PREFIX_PATH"):
        prefixes.append(env["CMAKE_PREFIX_PATH"])
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(prefixes)
    return env


TARGET_PROBE = r'''
import json, pathlib, platform, sys, sysconfig
import torch
import cmake
root = pathlib.Path(torch.__file__).parent / "share/cmake"
preferred = root / "Caffe2/Modules_CUDA_fix/FindCUDA.cmake"
if not preferred.is_file():
    candidates = sorted(p for p in root.rglob("FindCUDA.cmake") if p.parent.name == "Modules_CUDA_fix")
    if not candidates:
        raise RuntimeError("The installed PyTorch lacks its vendored FindCUDA.cmake")
    preferred = candidates[0]
value = {"python": sys.executable, "python_version": list(sys.version_info[:3]),
         "system": platform.system(), "machine": platform.machine(),
         "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
         "torch_find_cuda": str(preferred), "python_include": sysconfig.get_path("include"),
         "cmake_module": str(pathlib.Path(cmake.__file__))}
print("SHORE_CUDA_BUILD_INFO=" + json.dumps(value))
'''


def _bracket(value: str) -> str:
    equals = "="
    while "]" + equals + "]" in value:
        equals += "="
    return "[" + equals + "[" + value + "]" + equals + "]"


def cmake_source(module_dir: str) -> str:
    return '''cmake_minimum_required(VERSION 3.26.1)
list(PREPEND CMAKE_MODULE_PATH MODULE_DIR)
set(CMAKE_SIZEOF_VOID_P 8)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA 12.8 EXACT REQUIRED)
if(NOT CUDA_VERSION_STRING STREQUAL "12.8")
  message(FATAL_ERROR "Expected CUDA 12.8; found ${CUDA_VERSION_STRING}")
endif()
set(_shore_headers_ok FALSE)
foreach(_shore_include IN LISTS CUDA_INCLUDE_DIRS)
  if(EXISTS "${_shore_include}/cuda.h" AND EXISTS "${_shore_include}/cuda_runtime.h"
     AND EXISTS "${_shore_include}/device_functions.h")
    set(_shore_headers_ok TRUE)
  endif()
endforeach()
if(NOT _shore_headers_ok OR NOT EXISTS "${CUDA_CUDART_LIBRARY}")
  message(FATAL_ERROR "CUDA include/runtime detection did not match the checked toolkit")
endif()
message(STATUS "SHORE_FINDCUDA_OK=${CUDA_VERSION_STRING};${CUDA_INCLUDE_DIRS};${CUDA_CUDART_LIBRARY}")
'''.replace("MODULE_DIR", _bracket(Path(module_dir).as_posix()))


def cmake_command(python: str, found: dict[str, str], source: Path) -> list[str]:
    return [python, "-m", "cmake",
            "-DCUDA_TOOLKIT_ROOT_DIR:PATH=" + found["cuda_home"],
            "-DCUDAToolkit_ROOT:PATH=" + found["cuda_home"],
            "-DCUDA_TOOLKIT_INCLUDE:PATH=" + found["include_dir"],
            "-DCUDA_CUDART_LIBRARY:FILEPATH=" + found["cudart"],
            "-DCUDA_NVCC_EXECUTABLE:FILEPATH=" + found["nvcc"],
            "-DCUDA_USE_STATIC_CUDA_RUNTIME:BOOL=OFF", "-P", str(source)]


def nvcc_command(found: dict[str, str], cxx: str, source: Path, output: Path) -> list[str]:
    return [found["nvcc"], "-ccbin", cxx, "-std=c++17", "-arch=sm_90",
            "-I" + found["include_dir"], "-c", str(source), "-o", str(output)]


def run_preflight(python: str | Path, cuda_home: str | Path, output: str | Path,
                  cc: str | Path, cxx: str | Path, *, runner=subprocess.run) -> dict:
    """Keep stage logs and a JSON result; errors stop before the heavy build."""
    parent = _absolute(output)
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise PreflightError("The diagnostic output must be a directory")
    directory = parent / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10])
    directory.mkdir(mode=0o700)
    report = {"schema_version": "shore-cuda128-build-preflight-1.0", "status": "failed",
              "checks": [], "diagnostic_directory": str(directory),
              "gpu_initialized": False, "model_loaded": False,
              "compiled_program_executed": False, "packages_installed": False,
              "report_path": str(directory / "report.json")}
    stage = "cuda_layout"

    def run(name: str, command: list[str], *, timeout=180):
        nonlocal stage
        stage = name
        try:
            done = runner(command, cwd=directory, env=env, text=True,
                          encoding="utf-8", errors="replace", capture_output=True,
                          timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            text = (getattr(exc, "stdout", None) or "")
            error = (getattr(exc, "stderr", None) or "")
            (directory / (name + ".stdout.log")).write_text(text.decode(errors="replace") if isinstance(text, bytes) else text, encoding="utf-8")
            (directory / (name + ".stderr.log")).write_text(error.decode(errors="replace") if isinstance(error, bytes) else error, encoding="utf-8")
            raise PreflightError(f"{name} could not finish: {exc}") from exc
        (directory / (name + ".stdout.log")).write_text(done.stdout or "", encoding="utf-8")
        (directory / (name + ".stderr.log")).write_text(done.stderr or "", encoding="utf-8")
        if done.returncode:
            raise PreflightError(f"{name} failed with exit code {done.returncode}; see {directory}")
        report["checks"].append({"name": name, "passed": True})
        return done.stdout or ""

    try:
        found = layout(cuda_home)
        report["layout"] = found
        report["checks"].append({"name": "cuda_layout", "passed": True})
        stage = "configured_executables"
        python, cc, cxx = (_executable(python, "target Python"), _executable(cc, "C compiler"),
                           _executable(cxx, "C++ compiler"))
        env = build_environment(found, cc, cxx)
        report["checks"].append({"name": stage, "passed": True})
        for name in ("nvcc", "ptxas"):
            version = run(name + "_version", [found[name], "--version"])
            if not CUDA_128.search(version):
                raise PreflightError(f"Expected CUDA 12.8 {name}; inspect the version log")
        for name, executable in (("cc", cc), ("cxx", cxx)):
            version = run(name + "_version", [executable, "-dumpfullversion", "-dumpversion"]).strip()
            match = re.match(r"(\d+)\.(\d+)", version)
            if not match or not (11 <= int(match[1]) <= 14) or (int(match[1]) == 11 and int(match[2]) < 3):
                raise PreflightError(f"Unsupported {name} version for this CUDA12.8 profile: {version}")
        raw = run("target_python_probe", [python, "-I", "-c", TARGET_PROBE])
        records = [line[len(TAG):] for line in raw.splitlines() if line.startswith(TAG)]
        if len(records) != 1:
            raise PreflightError("Target Python did not return exactly one toolchain record")
        target = json.loads(records[0])
        report["target"] = target
        if target.get("system") != "Linux" or target.get("machine") not in ("x86_64", "AMD64"):
            raise PreflightError("The build target must be Linux x86_64")
        if target.get("python_version", [])[:2] != [3, 12]:
            raise PreflightError("The inference build must use Python 3.12")
        if target.get("torch_version") != "2.11.0+cu128" or target.get("torch_cuda") != "12.8":
            raise PreflightError("The installed inference environment must contain torch2.11.0+cu128")
        if not (Path(target["python_include"]) / "Python.h").is_file():
            raise PreflightError("The target Python is missing its development header Python.h")
        module = Path(target["torch_find_cuda"])
        if not module.is_file():
            raise PreflightError("The target Torch FindCUDA.cmake file is missing")
        find_source = directory / "find-cuda.cmake"
        find_source.write_text(cmake_source(str(module.parent)), encoding="utf-8")
        result = run("torch_find_cuda", cmake_command(python, found, find_source))
        if "SHORE_FINDCUDA_OK=12.8;" not in result:
            raise PreflightError("Torch FindCUDA did not confirm CUDA12.8 headers and runtime")
        source = directory / "nvcc-smoke.cu"
        source.write_text("#include <cuda_runtime.h>\n"
                          "__global__ void shore_cuda_build_probe(float *x) { x[threadIdx.x] = 1.0f; }\n", encoding="utf-8")
        object_file = directory / "nvcc-smoke.o"
        run("nvcc_compile_only", nvcc_command(found, cxx, source, object_file))
        if not object_file.is_file() or object_file.stat().st_size == 0:
            raise PreflightError("nvcc exited without producing a nonempty object file")
        report.update(status="passed", sm90_compile_only=True,
                      object_sha256=hashlib.sha256(object_file.read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError, PreflightError) as exc:
        report.update(failed_stage=stage, error=str(exc))
        report["checks"].append({"name": stage, "passed": False})
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="actual inference-venv Python; do not resolve its symlink")
    parser.add_argument("--cuda-home", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="reusable diagnostics parent; each run creates a new subdirectory")
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.python, args.cuda_home, args.output, args.cc, args.cxx)
    except (OSError, ValueError, PreflightError) as exc:
        print(json.dumps({"status": "failed", "stage": "diagnostic_output", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
