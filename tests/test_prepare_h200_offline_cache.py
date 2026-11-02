"""Offline-asset integrity and command construction, without CUDA or network."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_h200_offline_cache",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_h200_offline_cache.py",
)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def fixture_tree(tmp_path):
    root = tmp_path / "image"
    root.mkdir()
    build = {"target": "h200", "target_cuda_arch": "9.0", "cpu_stage_completed": True,
             "gpu_validation_performed": False, "keep_this_field": "preserved",
             "cuda_home": str(tmp_path / "cuda-12.8"), "cxx": str(tmp_path / "g++")}
    cache.write_json(root / cache.BUILD_MANIFEST_NAME, build)
    cubins, aot = root / "cubins", root / "aot"
    cubins.mkdir()
    aot.mkdir()
    (cubins / "kernel.cubin").write_bytes(b"prebuilt-cubin")
    (aot / "module.so").write_bytes(b"prebuilt-aot")
    records = [{"path": str(p), "size": p.stat().st_size, "sha256": cache.sha256_file(p)}
               for p in (cubins / "kernel.cubin", aot / "module.so")]
    manifest = {"schema_version": "1.0.0", "status": "prepared",
                "flashinfer_version": "0.6.12", "flashinfer_cubin_dir": str(cubins),
                "flashinfer_aot_dir": str(aot), "files": records}
    return root, build, manifest


def test_verify_never_launches_subprocess_or_download(tmp_path):
    root, _, manifest = fixture_tree(tmp_path)
    cache.write_json(root / cache.MANIFEST_NAME, manifest)
    with patch.object(cache.subprocess, "run", side_effect=AssertionError("No child process allowed")):
        assert cache.main(["--env-root", str(root), "--verify"]) == 0


@pytest.mark.parametrize("damage", ["changed", "missing", "duplicate", "escape", "wrong_version"])
def test_verify_rejects_incomplete_or_changed_assets(tmp_path, damage):
    root, _, manifest = fixture_tree(tmp_path)
    if damage == "changed":
        Path(manifest["files"][0]["path"]).write_bytes(b"other-contents")
    elif damage == "missing":
        Path(manifest["files"][1]["path"]).unlink()
    elif damage == "duplicate":
        manifest["files"].append(manifest["files"][0].copy())
    elif damage == "escape":
        outside = root / "unapproved.cubin"
        outside.write_bytes(b"unapproved")
        manifest["files"][0] = {"path": str(outside), "size": outside.stat().st_size,
                                "sha256": cache.sha256_file(outside)}
    else:
        manifest["flashinfer_version"] = "0.6.13"
    with pytest.raises(ValueError):
        cache.verify_manifest(root, manifest)


def test_cpu_download_environment_does_not_modify_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://not-used.invalid/simple")
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    result = cache.cpu_environment(tmp_path, {"cache_environment": {"VLLM_CACHE_ROOT": str(tmp_path)},
                                             "cuda_home": str(tmp_path / "cuda-12.8"),
                                             "cxx": str(tmp_path / "g++")})
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    assert result["FLASHINFER_CUDA_ARCH_LIST"] == "9.0a"
    assert result["TVM_FFI_CUDA_ARCH_LIST"] == "9.0a"
    assert result["CUDA_HOME"] == str(tmp_path / "cuda-12.8")
    assert result["CUDACXX"] == str(tmp_path / "cuda-12.8" / "bin" / "nvcc")
    assert result["CXX"] == result["NVCC_CCBIN"] == str(tmp_path / "g++")
    assert "PIP_EXTRA_INDEX_URL" not in result
    assert "FLASHINFER_NO_DOWNLOAD" not in result
    assert "FLASHINFER_DISABLE_JIT" not in result
    assert cache.os.environ["CUDA_VISIBLE_DEVICES"] == "2,3,4,5"
    assert cache.os.environ["FLASHINFER_NO_DOWNLOAD"] == "1"


def test_prepare_uses_exact_official_wheel_and_preserves_build_fields(tmp_path):
    root, build, fixture = fixture_tree(tmp_path)
    binary = root / ".venv-vllm-cu128" / "bin"
    binary.mkdir(parents=True)
    (binary / "python").write_bytes(b"placeholder")
    (binary / "flashinfer").write_bytes(b"placeholder")
    probe = {"flashinfer_cubin_dir": fixture["flashinfer_cubin_dir"],
             "flashinfer_aot_dir": fixture["flashinfer_aot_dir"],
             "official_artifact_count": 1, "aot_library_count": 1,
             "paths": [r["path"] for r in fixture["files"]]}
    versions = {"flashinfer-python": "0.6.12", "flashinfer-cubin": "0.6.12",
                "torch": "2.11.0+cu128", "vllm": "0.24.0", "transformers": "5.8.0"}
    constraint_contents = []
    def inspect_commands(command, **kwargs):
        if "install" in command:
            constraint_contents.append(Path(command[command.index("--constraint") + 1]).read_text())
    with patch.object(cache.platform, "system", return_value="Linux"), \
            patch.object(cache.platform, "machine", return_value="x86_64"), \
            patch.object(cache, "run_json", side_effect=[versions, probe]), \
            patch.object(cache.subprocess, "run", side_effect=inspect_commands) as run:
        result = cache.prepare(root, build, "http://mirror.invalid/simple", "mirror.invalid")
    assert result["status"] == "prepared"
    pip = run.call_args_list[0].args[0]
    assert "flashinfer-jit-cache==0.6.12+cu128" in pip
    assert cache.OFFICIAL_JIT_LINKS in pip
    assert "--no-deps" not in pip and "--no-cache-dir" in pip
    assert "torch==2.11.0+cu128\n" in constraint_contents[0]
    assert "flashinfer-cubin==0.6.12\n" in constraint_contents[0]
    assert not list(root.glob(".flashinfer-constraints-*"))
    assert run.call_args_list[1].args[0] == [str(binary / "python"), "-m", "pip", "check"]
    assert run.call_args_list[2].args[0] == [str(binary / "flashinfer"), "download-cubin"]
    saved = json.loads((root / cache.BUILD_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved["keep_this_field"] == "preserved"
    assert saved["offline_cache_prepared"] is True
    assert saved["offline_cache_manifest"] == cache.MANIFEST_NAME
    assert cache.verify_manifest(root, cache.load_json(root / cache.MANIFEST_NAME))["files"] == 2


def test_reject_unfinished_or_other_arch_build(tmp_path):
    root, build, _ = fixture_tree(tmp_path)
    for change in ({"target": "4090"}, {"target_cuda_arch": "8.9"},
                   {"cpu_stage_completed": False}):
        cache.write_json(root / cache.BUILD_MANIFEST_NAME, {**build, **change})
        with pytest.raises(ValueError):
            cache.read_build_manifest(root)


def test_dependency_failure_stops_before_download_and_cleans_only_own_constraints(tmp_path):
    root, build, _ = fixture_tree(tmp_path)
    binary = root / ".venv-vllm-cu128" / "bin"
    binary.mkdir(parents=True)
    for name in ("python", "flashinfer"):
        (binary / name).write_bytes(b"placeholder")
    existing = root / ".flashinfer-constraints-user.txt"
    existing.write_text("keep")
    versions = {"flashinfer-python": "0.6.12", "flashinfer-cubin": "0.6.12",
                "torch": "2.11.0+cu128", "vllm": "0.24.0", "transformers": "5.8.0"}
    def fail_check(command, **kwargs):
        if command[-1] == "check":
            raise cache.subprocess.CalledProcessError(1, command)
    with patch.object(cache.platform, "system", return_value="Linux"), \
            patch.object(cache.platform, "machine", return_value="x86_64"), \
            patch.object(cache, "run_json", return_value=versions), \
            patch.object(cache.subprocess, "run", side_effect=fail_check) as run:
        with pytest.raises(cache.subprocess.CalledProcessError):
            cache.prepare(root, build, "http://mirror.invalid/simple", "mirror.invalid")
    assert run.call_count == 2
    assert existing.read_text() == "keep"
    assert list(root.glob(".flashinfer-constraints-*")) == [existing]
    assert not (root / cache.MANIFEST_NAME).exists()


@pytest.mark.parametrize("kernel_available", [True, False])
def test_empty_upstream_directory_cannot_hide_checksum_listed_kernel(tmp_path, kernel_available):
    import hashlib

    root = tmp_path / "cubins"
    root.mkdir()
    kernel, header = b"actual-kernel", b"actual-header"
    checksum_text = (f"{hashlib.sha256(kernel).hexdigest()}  kernel.cubin\n"
                     f"{hashlib.sha256(header).hexdigest()}  include/meta.h\n").encode()
    upstream = {"release/checksums.txt": checksum_text,
                "release/kernel.cubin": kernel, "release/include/meta.h": header}
    # Simulate the real upstream failure mode: directory discovery returns no
    # files, even though the authenticated checksum manifest contains a kernel.
    def get_available_cubin_files(*args):
        return ()
    assert get_available_cubin_files("release/") == ()
    requested = []
    def get_artifact(name, checksum):
        requested.append(name)
        if name == "release/kernel.cubin" and not kernel_available:
            return b""
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(upstream[name])
        return upstream[name]
    pinned = {"release/checksums.txt": hashlib.sha256(checksum_text).hexdigest()}
    if kernel_available:
        actual = cache.populate_pinned_artifacts(root, pinned, get_artifact)
        assert set(actual) == {str((root / name).resolve()) for name in upstream}
        assert "release/kernel.cubin" in requested
        assert (root / "release/kernel.cubin").read_bytes() == kernel
    else:
        with pytest.raises(ValueError, match="Missing or corrupt pinned artifact"):
            cache.populate_pinned_artifacts(root, pinned, get_artifact)
