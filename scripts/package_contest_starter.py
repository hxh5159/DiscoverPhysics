"""Build only the explicitly public starter files, never a repository archive."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "LICENSE",
    "rift/__init__.py", "rift/console.py", "rift/submission.py", "rift/law.py",
    "rift/observations.py", "rift/prediction.py", "rift/worker.py",
    "rift_numerics/__init__.py", "rift_numerics/rollout.py",
    "scripts/validate_claims.py", "scripts/validate_law.py", "scripts/validate_submission.py",
    "docs/LAW_API.txt", "docs/CONSOLE_API.txt", "docs/SUBMISSION_API.txt", "docs/CONTEST_WORKFLOW.txt",
    "docs/contracts/claims.schema.json", "docs/contracts/claims.example.json",
    "docs/contracts/law_api.json", "docs/contracts/law_case.schema.json",
    "docs/contracts/numerics.json", "docs/contracts/submission.schema.json", "docs/contracts/runtime.json",
    "examples/agent.py", "examples/submission.json", "examples/law.py", "examples/law_case.json", "examples/Dockerfile.agent",
)
PUBLIC_NUMERICS_FIELDS = (
    "numerics_version", "integrator", "dt", "dtype", "sampling", "softening",
    "shared_implementation", "candidate_inputs", "private_world_access",
    "freeze_policy", "time_meaning", "background_propagation", "shared_module",
    "prediction_program_protocol", "state_callback",
)
START_HERE = '''攻坚组接口起步包

请先阅读docs/SUBMISSION_API.txt、CONSOLE_API.txt和LAW_API.txt。
在已安装NumPy与SciPy的官方环境中，可运行：
  python scripts/validate_submission.py examples/submission.json
  python scripts/validate_law.py examples/law.py --claims docs/contracts/claims.example.json
  python scripts/validate_law.py examples/law.py --case examples/law_case.json --execute

启动入口示例位于examples/agent.py，只验证交互和提交格式，不进行模型推理或规律发现。
要按examples/submission.json启动，应把examples目录作为提交工作目录，并确保公开SDK可导入。
控制台地址、会话令牌和产物目录由可信运行器提供，不在这个包中预置。
正式模型权重、基础镜像、测试配置和完整比赛接入方式由指挥部随环境发布。
本包不含正式宇宙的真实模拟器或隐藏验证真值，接口示例不新增参赛练习阶段。
公开数值说明仅包含有效操作约定；原研究记录和历史比较不随包发布。
'''


def _public_bytes(relative, data):
    if relative == "docs/contracts/numerics.json":
        source = json.loads(data.decode("utf-8-sig"))
        public = {key: source[key] for key in PUBLIC_NUMERICS_FIELDS if key in source}
        if not {"numerics_version", "integrator", "dt", "sampling"}.issubset(public):
            raise ValueError("The public numerical contract is incomplete.")
        return (json.dumps(public, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    if relative.startswith("docs/contracts/") and relative.endswith(".schema.json"):
        source = json.loads(data.decode("utf-8-sig"))
        identifier = source.get("$id")
        if isinstance(identifier, str) and identifier.startswith("urn:discoverphysics:"):
            source["$id"] = identifier.replace("urn:discoverphysics:", "urn:shore:", 1)
            return (json.dumps(source, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    return data


def build_starter(output, *, root=ROOT, overwrite=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Explicit output must have a .zip extension.")
    if output.exists() and not overwrite:
        raise FileExistsError("Output already exists; choose another path or use --overwrite.")
    payloads = {}
    for relative in PUBLIC_FILES:
        lexical = root / relative
        if lexical.is_symlink() or not lexical.resolve().is_relative_to(root) or not lexical.is_file():
            raise ValueError(f"Required public file is missing or not a regular in-root file: {relative}")
        current = root
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"Public source path must not contain symlinks: {relative}")
        payloads[relative] = _public_bytes(relative, lexical.read_bytes())
    payloads["START_HERE.txt"] = START_HERE.encode("utf-8")
    inventory = [{"path": path, "sha256": hashlib.sha256(data).hexdigest()}
                 for path, data in sorted(payloads.items())]
    manifest = {"starter_schema_version": "1.0.0", "contains_formal_world_simulators": False,
                "contains_model_weights": False, "files": inventory}
    manifest["public_contract_projection"] = "Numerics includes normative public fields only; schema identifiers use the participant namespace. Mathematical and validation rules are unchanged."
    payloads["starter-manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return {"file_count": len(payloads), "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "contains_formal_world_simulators": False, "contains_model_weights": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_starter(args.output, overwrite=args.overwrite)
        print(json.dumps({"ok": True, **result}))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
