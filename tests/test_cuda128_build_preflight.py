"""CPU-only preflight boundaries; synthetic commands never install or use CUDA."""
from pathlib import Path
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cuda_build_preflight", ROOT / "scripts/preflight_cuda128_build.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def fixture(tmp_path, *, conda=True):
    root = tmp_path / "cuda-12.8"
    include = root / ("targets/x86_64-linux/include" if conda else "include")
    library = root / ("targets/x86_64-linux/lib" if conda else "lib64")
    include.mkdir(parents=True)
    library.mkdir(parents=True)
    for name in preflight.HEADERS:
        (include / name).write_text("// fixture CUDA header\n")
    (library / "libcudart.so").write_bytes(b"fixture runtime")
    for name in ("nvcc", "ptxas", "gcc", "g++", "python"):
        target = root / "bin" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"fixture executable, not run")
    module = tmp_path / "target-torch/share/cmake/Caffe2/Modules_CUDA_fix/FindCUDA.cmake"
    module.parent.mkdir(parents=True)
    module.write_text("# synthetic module")
    python_include = tmp_path / "python-include"
    python_include.mkdir()
    (python_include / "Python.h").write_text("// fixture Python header")
    info = {"python_version": [3, 12, 16], "system": "Linux", "machine": "x86_64",
            "torch_version": "2.11.0+cu128", "torch_cuda": "12.8",
            "torch_find_cuda": str(module), "python_include": str(python_include)}
    return root, include, library, info


class FakeRunner:
    def __init__(self, info, *, fail=None, version="12.8", write_object=True):
        self.info, self.fail = info, fail
        self.version, self.write_object = version, write_object
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        output = ""
        if "--version" in argv:
            stage = Path(argv[0]).name + "_version"
            output = f"Cuda compilation tools, release {self.version}, V{self.version}.93\n"
        elif "-dumpversion" in argv:
            stage = "cc_version" if Path(argv[0]).name == "gcc" else "cxx_version"
            output = "12.4.0\n"
        elif "-c" in argv and "SHORE_CUDA_BUILD_INFO=" in argv[-1]:
            stage = "target_python_probe"
            output = preflight.TAG + json.dumps(self.info) + "\n"
        elif "cmake" in argv:
            stage = "torch_find_cuda"
            output = "-- SHORE_FINDCUDA_OK=12.8;fixture/include;fixture/libcudart.so\n"
        else:
            stage = "nvcc_compile_only"
            if self.write_object:
                Path(argv[argv.index("-o") + 1]).write_bytes(b"SM90 fixture object, not real compilation")
        code = 1 if self.fail == stage else 0
        return subprocess.CompletedProcess(argv, code, output, "synthetic build failure" if code else "")


@pytest.mark.parametrize("conda", [False, True])
def test_detects_actual_flat_and_conda_layout_files(tmp_path, conda):
    root, include, library, _ = fixture(tmp_path, conda=conda)
    found = preflight.layout(root)
    assert found["include_dir"] == str(include)
    assert found["library_dir"] == str(library)


@pytest.mark.parametrize("missing", ["cuda.h", "cuda_runtime.h", "device_functions.h", "libcudart.so", "nvcc", "ptxas"])
def test_missing_toolchain_component_fails_before_any_process(tmp_path, missing):
    root, include, library, _ = fixture(tmp_path)
    path = (library / missing if missing == "libcudart.so" else
            root / "bin" / missing if missing in ("nvcc", "ptxas") else include / missing)
    path.unlink()
    with pytest.raises(preflight.PreflightError):
        preflight.layout(root)


def test_mocked_full_preflight_uses_explicit_paths_and_compile_only(tmp_path):
    root, include, library, info = fixture(tmp_path)
    run = FakeRunner(info)
    result = preflight.run_preflight(root / "bin/python", root, tmp_path / "diagnostics",
                                    root / "bin/gcc", root / "bin/g++", runner=run)
    assert result["status"] == "passed"
    assert result["sm90_compile_only"] is True
    assert not result["gpu_initialized"] and not result["model_loaded"]
    assert not result["compiled_program_executed"]
    cmake = next(argv for argv, _ in run.calls if "cmake" in argv)
    assert "-DCUDA_TOOLKIT_INCLUDE:PATH=" + str(include) in cmake
    assert "-DCUDA_CUDART_LIBRARY:FILEPATH=" + str(library / "libcudart.so") in cmake
    assert "-DCUDA_TOOLKIT_ROOT_DIR:PATH=" + str(root) in cmake
    compile_command = run.calls[-1][0]
    assert "-c" in compile_command and "-arch=sm_90" in compile_command
    assert "--run" not in compile_command and "-run" not in compile_command
    for argv, kwargs in run.calls:
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "nvidia-smi" not in argv
    script = (Path(result["diagnostic_directory"]) / "find-cuda.cmake").read_text(encoding="utf-8")
    assert "find_package(CUDA 12.8 EXACT REQUIRED)" in script
    assert "find_package(Torch" not in script and "enable_language" not in script
    assert json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))["status"] == "passed"


@pytest.mark.parametrize("stage", ["target_python_probe", "torch_find_cuda", "nvcc_compile_only"])
def test_failure_preserves_stage_logs_and_stops_following_steps(tmp_path, stage):
    root, _, _, info = fixture(tmp_path)
    runner = FakeRunner(info, fail=stage)
    result = preflight.run_preflight(root / "bin/python", root, tmp_path / "diagnostics",
                                    root / "bin/gcc", root / "bin/g++", runner=runner)
    assert result["status"] == "failed" and result["failed_stage"] == stage
    assert "synthetic build failure" in result["error"]
    assert Path(result["report_path"]).is_file()
    assert "synthetic build failure" in (Path(result["diagnostic_directory"]) / (stage + ".stderr.log")).read_text()
    if stage != "nvcc_compile_only":
        assert not (Path(result["diagnostic_directory"]) / "nvcc-smoke.o").exists()


def test_wrong_cuda_version_is_not_accepted(tmp_path):
    root, _, _, info = fixture(tmp_path)
    runner = FakeRunner(info, version="13.2")
    result = preflight.run_preflight(root / "bin/python", root, tmp_path / "diagnostics",
                                    root / "bin/gcc", root / "bin/g++", runner=runner)
    assert result["status"] == "failed" and len(runner.calls) == 1
    assert "12.8" in result["error"]


def test_wrong_torch_or_python_stops_before_cmake(tmp_path):
    root, _, _, info = fixture(tmp_path)
    info["torch_cuda"] = "13.0"
    runner = FakeRunner(info)
    result = preflight.run_preflight(root / "bin/python", root, tmp_path / "diagnostics",
                                    root / "bin/gcc", root / "bin/g++", runner=runner)
    assert result["status"] == "failed" and result["failed_stage"] == "target_python_probe"
    assert not any("cmake" in argv for argv, _ in runner.calls)


def test_reused_diagnostic_parent_keeps_previous_files(tmp_path):
    root, _, _, info = fixture(tmp_path)
    output = tmp_path / "diagnostics"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve me")
    reports = [preflight.run_preflight(root / "bin/python", root, output,
                root / "bin/gcc", root / "bin/g++", runner=FakeRunner(info)) for _ in range(2)]
    assert reports[0]["diagnostic_directory"] != reports[1]["diagnostic_directory"]
    assert (output / "user-file.txt").read_text() == "preserve me"
    assert all(Path(row["report_path"]).is_file() for row in reports)


def test_nonempty_compile_output_is_required(tmp_path):
    root, _, _, info = fixture(tmp_path)
    result = preflight.run_preflight(root / "bin/python", root, tmp_path / "diagnostics",
        root / "bin/gcc", root / "bin/g++", runner=FakeRunner(info, write_object=False))
    assert result["status"] == "failed" and "object" in result["error"]


def test_environment_uses_real_libraries_and_preserves_venv_path(tmp_path):
    root, _, library, _ = fixture(tmp_path)
    env = preflight.build_environment(preflight.layout(root), "gcc", "g++", {
        "PATH": "ordinary-bin", "LD_LIBRARY_PATH": os.pathsep.join(["/bad/stubs", "/safe/lib"]),
        "PYTHONHOME": "foreign-python", "PYTHONPATH": "foreign-imports"})
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    assert "/bad/stubs" not in env["LD_LIBRARY_PATH"]
    assert str(library) in env["LD_LIBRARY_PATH"]
    relative = tmp_path / "venv/bin/python"
    assert str(preflight._absolute(relative)) == str(relative)


def test_help_is_side_effect_free_and_does_not_import_torch(tmp_path):
    process = subprocess.run([sys.executable, str(ROOT / "scripts/preflight_cuda128_build.py"), "--help"],
                             cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0
    assert "--cuda-home" in process.stdout
    assert not list(tmp_path.iterdir())


def real_cmake_executable():
    """Use an explicitly supplied/PATH CMake, without downloading test tools."""
    executable = os.environ.get("SHORE_TEST_CMAKE") or shutil.which("cmake")
    if not executable:
        pytest.skip("Real CMake is not installed; set SHORE_TEST_CMAKE to run this regression")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=15)
    match = re.search(r"cmake version (\d+)\.(\d+)", version.stdout)
    if version.returncode or not match or tuple(map(int, match.groups())) < (3, 26):
        pytest.skip("The real CMake regression requires CMake >=3.26")
    return executable


@pytest.mark.parametrize("missing_header", [False, True])
def test_real_cmake_script_initialization_and_header_validation(tmp_path, missing_header):
    """Run real CMake; the FindCUDA fixture isolates its required initializer.

    This is not a CUDA/nvcc simulation advertised as real toolchain discovery:
    it tests the script-mode initialization contract that FakeRunner cannot.
    """
    executable = real_cmake_executable()
    root, include, _, _ = fixture(tmp_path)
    found = preflight.layout(root)
    modules = tmp_path / "cmake-modules"
    modules.mkdir()
    (modules / "FindCUDA.cmake").write_text(
        'cmake_initialize_per_config_variable(CUDA_NVCC_FLAGS "Probe configuration")\n'
        'set(CUDA_VERSION_STRING "12.8")\n'
        'set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_INCLUDE}")\n',
        encoding="utf-8")
    source = tmp_path / "find-cuda.cmake"
    generated = preflight.cmake_source(str(modules))
    flags = preflight.cmake_command(sys.executable, found, source)[3:]
    command = [executable, *flags]
    # The previous generated script fails in the real interpreter even though
    # all file fixtures are present; this is the reported initialization bug.
    source.write_text(generated.replace("include(CMakeInitializeConfigs)\n", ""), encoding="utf-8")
    before = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert before.returncode != 0
    assert 'Unknown CMake command "cmake_initialize_per_config_variable"' in before.stderr

    if missing_header:
        (include / "device_functions.h").unlink()
    source.write_text(generated, encoding="utf-8")
    after = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30)
    assert 'Unknown CMake command "cmake_initialize_per_config_variable"' not in after.stderr
    if missing_header:
        assert after.returncode != 0
        assert "CUDA include/runtime detection did not match the checked toolkit" in after.stderr
    else:
        assert after.returncode == 0, after.stdout + after.stderr
        assert "SHORE_FINDCUDA_OK=12.8;" in after.stdout
