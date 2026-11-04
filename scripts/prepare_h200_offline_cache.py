#!/usr/bin/env python3
"""Prepare FlashInfer's pinned offline assets on the networked CPU image.

This does not initialize a GPU, load a model, or warm every runtime shape.
The verify operation uses only the local manifest and ordinary file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = "1.0.0"
FLASHINFER_VERSION = "0.6.12"
JIT_CACHE_VERSION = "0.6.12+cu128"
MANIFEST_NAME = "offline-kernel-cache.json"
BUILD_MANIFEST_NAME = "cuda128-build-manifest.json"
OFFICIAL_JIT_LINKS = "https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a manifest symlink: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def read_build_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / BUILD_MANIFEST_NAME)
    if manifest.get("target") != "h200":
        raise ValueError("Build manifest target must be h200.")
    if str(manifest.get("target_cuda_arch")) != "9.0":
        raise ValueError("Build manifest target_cuda_arch must be 9.0.")
    if manifest.get("cpu_stage_completed") is not True:
        raise ValueError("Complete the CPU build stage before preparing assets.")
    if manifest.get("gpu_validation_performed") is not False:
        raise ValueError("Expected the networked CPU-stage build manifest.")
    return manifest


def verify_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "prepared":
        raise ValueError("Offline asset manifest schema/status is invalid.")
    if manifest.get("flashinfer_version") != FLASHINFER_VERSION:
        raise ValueError("Offline assets must match FlashInfer 0.6.12.")
    roots: list[Path] = []
    for field in ("flashinfer_cubin_dir", "flashinfer_aot_dir"):
        candidate = Path(manifest.get(field, ""))
        if not candidate.is_absolute() or not contained(candidate, root) or not candidate.is_dir():
            raise ValueError(f"Invalid or missing cache directory: {field}")
        roots.append(candidate.resolve())
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Offline asset manifest has no files.")
    seen: set[Path] = set()
    total = 0
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Malformed asset file record.")
        path = Path(record["path"])
        resolved = path.resolve()
        if not path.is_absolute() or not any(contained(path, allowed) for allowed in roots):
            raise ValueError(f"Asset path escapes declared cache directories: {path}")
        if resolved in seen:
            raise ValueError(f"Duplicate asset record: {path}")
        seen.add(resolved)
        size, checksum = record.get("size"), record.get("sha256")
        if type(size) is not int or size < 0 or not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Invalid asset size or checksum: {path}")
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != checksum:
            raise ValueError(f"Missing or changed offline asset: {path}")
        total += size
    return {"status": "verified", "files": len(records), "bytes": total,
            "network_calls": 0, "gpu_validation_performed": False,
            "model_loaded": False}


def cpu_environment(root: Path, build: dict[str, Any]) -> dict[str, str]:
    result = os.environ.copy()
    cache = build.get("cache_environment", {})
    if not isinstance(cache, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in cache.items()):
        raise ValueError("Build cache_environment must map strings to strings.")
    result.update(cache)
    for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "FLASHINFER_NO_DOWNLOAD",
                "FLASHINFER_DISABLE_JIT", "FLASHINFER_DISABLE_VERSION_CHECK"):
        result.pop(key, None)
    result.update({"PIP_CONFIG_FILE": "/dev/null", "CUDA_VISIBLE_DEVICES": "",
                   "FLASHINFER_CUDA_ARCH_LIST": "9.0a", "TVM_FFI_CUDA_ARCH_LIST": "9.0a",
                   "HF_HUB_OFFLINE": "1",
                   "VLLM_NO_USAGE_STATS": "1"})
    cuda_home, cxx = build.get("cuda_home"), build.get("cxx")
    if not isinstance(cuda_home, str) or not Path(cuda_home).is_absolute():
        raise ValueError("Build manifest must record absolute cuda_home.")
    if not isinstance(cxx, str) or not Path(cxx).is_absolute():
        raise ValueError("Build manifest must record the absolute C++ compiler path.")
    result.update({"CUDA_HOME": cuda_home, "CUDACXX": str(Path(cuda_home) / "bin" / "nvcc"),
                   "CXX": cxx, "CUDAHOSTCXX": cxx, "NVCC_CCBIN": cxx})
    result["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + result.get("PATH", "")
    result.setdefault("FLASHINFER_WORKSPACE_BASE", str(root))
    return result


def run_json(python: Path, source: str, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run([str(python), "-c", source], env=env,
                               check=True, text=True, stdout=subprocess.PIPE)
    # Imported libraries may print informational lines; only our tagged record is JSON.
    prefix = "SHORE_ASSET_JSON="
    records = [line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise ValueError("The environment probe did not return exactly one JSON record.")
    return json.loads(records[0])


def populate_pinned_artifacts(cubin_dir, checksum_map, get_artifact):
    """Fetch every file in SHA-pinned manifests, without directory listings."""
    import hashlib
    from pathlib import Path, PurePosixPath
    import re

    root = Path(cubin_dir).resolve()
    expected = {}
    manifests = dict(checksum_map)
    if not manifests:
        raise ValueError("The pinned FlashInfer checksum registry is empty.")

    def checked_name(name):
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError(f"Invalid artifact path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or any(part in (".", "..") for part in path.parts):
            raise ValueError(f"Artifact path must remain relative: {name}")
        local = root.joinpath(*path.parts)
        if not local.resolve().is_relative_to(root):
            raise ValueError(f"Artifact path escapes the cubin directory: {name}")
        return path, local

    def verified_bytes(name, checksum):
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ValueError(f"Invalid SHA256 for artifact: {name}")
        _, local = checked_name(name)
        data = get_artifact(name, checksum)
        if (hashlib.sha256(data).hexdigest() != checksum
                or not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != checksum):
            raise ValueError(f"Missing or corrupt pinned artifact: {name}")
        return data

    for manifest_name, manifest_checksum in sorted(manifests.items()):
        manifest_path, _ = checked_name(manifest_name)
        data = verified_bytes(manifest_name, manifest_checksum)
        expected[manifest_name] = manifest_checksum
        entries = 0
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed pinned checksum manifest: {manifest_name}")
            checksum, relative_name = fields
            relative_path, _ = checked_name(relative_name)
            full_name = str(manifest_path.parent / relative_path)
            checked_name(full_name)
            if full_name in expected and expected[full_name] != checksum:
                raise ValueError(f"Conflicting artifact checksums: {full_name}")
            expected[full_name] = checksum
            entries += 1
        if not entries:
            raise ValueError(f"Pinned checksum manifest has no files: {manifest_name}")
    # get_artifact performs a cache lookup first and downloads only a missing or
    # invalid file. Incomplete directory listings cannot omit any manifest entry.
    for name, checksum in sorted(expected.items()):
        verified_bytes(name, checksum)
    return [str(root.joinpath(*PurePosixPath(name).parts).resolve())
            for name in sorted(expected)]


VERSION_PROBE = r'''
import importlib.metadata as md, json
names = ('torch', 'vllm', 'transformers', 'flashinfer-python', 'flashinfer-cubin')
versions = {name: md.version(name) for name in names}
assert versions['torch'] == '2.11.0+cu128', versions
assert versions['vllm'].split('+')[0] == '0.24.0', versions
assert versions['transformers'] == '5.8.0', versions
assert versions['flashinfer-python'] == versions['flashinfer-cubin'] == '0.6.12', versions
print('SHORE_ASSET_JSON=' + json.dumps(versions))
'''


ASSET_PROBE = inspect.getsource(populate_pinned_artifacts) + r'''
import importlib.metadata as md, json
from pathlib import Path
from flashinfer.jit import env
from flashinfer.artifacts import CheckSumHash
from flashinfer.jit.cubin_loader import get_artifact
assert md.version('flashinfer-jit-cache') == '0.6.12+cu128'
# This may download missing files, but runs only in the ONLINE CPU stage.
registry = populate_pinned_artifacts(env.FLASHINFER_CUBIN_DIR,
                                     CheckSumHash.map_checksums, get_artifact)
aot = list(env.FLASHINFER_AOT_DIR.rglob('*.so'))
assert aot, 'The JIT-cache package contains no shared libraries'
paths = set(registry)
paths.update(str(path.resolve()) for path in aot)
print('SHORE_ASSET_JSON=' + json.dumps({
  'flashinfer_cubin_dir': str(env.FLASHINFER_CUBIN_DIR.resolve()),
  'flashinfer_aot_dir': str(env.FLASHINFER_AOT_DIR.resolve()),
  'official_artifact_count': len(registry), 'aot_library_count': len(aot),
  'flashinfer_jit_cache_version': md.version('flashinfer-jit-cache'),
  'paths': sorted(paths)}))
'''


def prepare(root: Path, build: dict[str, Any], index: str, host: str, *,
            mirror_only: bool = False) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Preparation requires the Linux x86_64 CPU build image.")
    python = root / ".venv-vllm-cu128" / "bin" / "python"
    cli = python.with_name("flashinfer")
    if not python.is_file() or not cli.is_file():
        raise ValueError("Build the pinned model environment first.")
    env = cpu_environment(root, build)
    versions = run_json(python, VERSION_PROBE, env)
    # Resolve normally, while preventing a companion wheel from replacing the
    # pinned model stack. Only this invocation's temporary constraint is removed.
    constraint_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".flashinfer-constraints-", suffix=".txt",
                                         delete=False) as constraints:
            constraint_name = constraints.name
            for name in ("torch", "vllm", "transformers", "flashinfer-python", "flashinfer-cubin"):
                constraints.write(f"{name}=={versions[name]}\n")
        pip_helper = Path(__file__).with_name("pip_index_install.py")
        if not pip_helper.is_file():
            raise ValueError("Matching pip_index_install.py is missing; use the complete installer helpers.")
        command = [
            str(python), str(pip_helper), "--python", str(python),
            "--index-url", index, "--trusted-host", host,
            "--log-dir", str(root / "pip-logs"), "--operation", "install",
        ]
        if mirror_only:
            command.append("--mirror-only")
        command.extend(["--", "--only-binary=:all:", "--constraint", constraint_name,
                        f"flashinfer-jit-cache=={JIT_CACHE_VERSION}"])
        if not mirror_only:
            command.extend(["--find-links", OFFICIAL_JIT_LINKS])
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError:
            if mirror_only:
                print("[SHORE offline cache] Mirror-only package installation failed. The configured index "
                      f"must provide flashinfer-jit-cache=={JIT_CACHE_VERSION} and its dependencies; "
                      "no official JIT-cache package link or alternate index was used.", file=sys.stderr)
            raise
        subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    finally:
        if constraint_name is not None:
            Path(constraint_name).unlink(missing_ok=True)
    # v0.6.12 enumerates all release artifacts; no GPU or per-device allocation.
    subprocess.run([str(cli), "download-cubin"], env=env, check=True)
    probe = run_json(python, ASSET_PROBE, env)
    records = []
    for name in probe.pop("paths"):
        path = Path(name)
        if not contained(path, root):
            raise ValueError(f"Asset is outside captured environment root: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path),
                        "size": path.stat().st_size})
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "prepared", "target": "h200",
        "cuda": "12.8", "flashinfer_version": FLASHINFER_VERSION, "versions": versions,
        **probe, "files": records,
        "unverified": ["GPU initialization", "all runtime shapes", "model loading"],
        "preparation_used_network": True, "verification_uses_network": False,
        "package_index": index, "mirror_only": bool(mirror_only),
        "gpu_validation_performed": False, "model_loaded": False,
        "notes": [
            "Installed flashinfer-cubin takes precedence over FLASHINFER_CUBIN_DIR in 0.6.12.",
            "download-cubin and artifact registry enumeration ran only in the online CPU stage.",
            "Local verification does not execute show-config, artifact-status or download-cubin.",
            "Offline runtime requires FLASHINFER_NO_DOWNLOAD=1; local JIT remains allowed.",
        ],
    }
    verify_manifest(root, manifest)
    write_json(root / MANIFEST_NAME, manifest)
    build["offline_cache_manifest"] = MANIFEST_NAME
    build["offline_cache_prepared"] = True
    write_json(root / BUILD_MANIFEST_NAME, build)
    return {"status": "prepared", "manifest": str(root / MANIFEST_NAME),
            "files": len(records), "bytes": sum(x["size"] for x in records),
            "gpu_validation_performed": False, "model_loaded": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--index-url")
    parser.add_argument("--trusted-host")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Install Python packages only from the configured index, without fallback")
    parser.add_argument("--verify", action="store_true",
                        help="Verify recorded files locally; no imports, downloads, GPU or model calls.")
    args = parser.parse_args(argv)
    try:
        if not args.env_root.is_absolute() or args.env_root.resolve() == Path(args.env_root.anchor):
            raise ValueError("--env-root must be an absolute directory other than filesystem root.")
        root = args.env_root.resolve(strict=True)
        build = read_build_manifest(root)
        if args.verify:
            report = verify_manifest(root, load_json(root / MANIFEST_NAME))
        else:
            if not args.index_url or not args.trusted_host:
                parser.error("Preparation requires --index-url and --trusted-host.")
            report = prepare(root, build, args.index_url, args.trusted_host, mirror_only=args.mirror_only)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
