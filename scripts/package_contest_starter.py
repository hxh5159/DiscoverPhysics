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
    "rift/model_registry.py", "rift/model_catalog.json",
    "rift_numerics/__init__.py", "rift_numerics/rollout.py",
    "shore_agent/__init__.py", "shore_agent/agent.py", "shore_agent/models.py",
    "shore_agent/tools.py", "shore_agent/tool_worker.py",
    "development_worlds/__init__.py", "development_worlds/worlds.py",
    "development_worlds/catalog.py", "development_worlds/console.py",
    "development_worlds/evaluation.py", "development_worlds/service.py",
    "development_worlds/runner.py", "development_worlds/README.md",
    "scripts/__init__.py", "scripts/validate_claims.py", "scripts/validate_law.py", "scripts/validate_submission.py",
    "scripts/serve_model.py", "scripts/build_runtime_images.py", "scripts/make_submission.py",
    "scripts/evaluate_submission.py",
    "scripts/bootstrap_cuda128.sh", "scripts/prepare_h200_offline_cache.py",
    "scripts/pip_index_install.py", "scripts/prepare_vllm_dependencies.py", "scripts/preflight_cuda128_build.py",
    "scripts/prepare_torch_wheel.py",
    "scripts/verify_h200_cuda128.py", "requirements.txt",
    "docs/LAW_API.txt", "docs/CONSOLE_API.txt", "docs/SUBMISSION_API.txt", "docs/CONTEST_WORKFLOW.txt",
    "docs/QUICKSTART.txt", "docs/PARTICIPANT_FILES.txt", "docs/MODEL_RUNTIME.txt", "docs/MODELS.txt",
    "docs/CODE_AND_SUBMISSION_SPEC.txt", "docs/彼岸物理规律发现挑战赛_赛题说明_更新版.pdf",
    "docs/H200_OFFLINE_SETUP.txt", "docs/CUDA128_SETUP.txt",
    "docs/contracts/claims.schema.json", "docs/contracts/claims.example.json",
    "docs/contracts/law_api.json", "docs/contracts/law_case.schema.json",
    "docs/contracts/numerics.json", "docs/contracts/submission.schema.json", "docs/contracts/runtime.json",
    "docs/contracts/exploration_protocol.json", "docs/contracts/observation_noise.json",
    "docs/contracts/scoring.json", "docs/contracts/feedback.schema.json",
    "examples/agent.py", "examples/submission.json", "examples/law.py", "examples/law_case.json", "examples/Dockerfile.agent",
    "examples/Dockerfile.candidate",
)
PUBLIC_NUMERICS_FIELDS = (
    "numerics_version", "integrator", "dt", "dtype", "sampling", "softening",
    "shared_implementation", "candidate_inputs", "private_world_access",
    "freeze_policy", "time_meaning", "background_propagation", "shared_module",
    "prediction_program_protocol", "state_callback",
)
START_HERE = '''攻坚组开发与实验起步包

先阅读docs/QUICKSTART.txt，再看PARTICIPANT_FILES.txt和MODEL_RUNTIME.txt。
本包包含六公开宇宙、可修改LLM基础框架、固定候选积分器及公开计分程序。
在已安装NumPy与SciPy的官方环境中，可运行：
  python scripts/validate_submission.py examples/submission.json
  python scripts/validate_law.py examples/law.py --claims docs/contracts/claims.example.json
  python scripts/validate_law.py examples/law.py --case examples/law_case.json --execute

模型准备后，从本包根目录运行python -m development_worlds.runner --help。
实际默认入口python -m shore_agent.agent会加载所选本地模型并自动探索；没有模型就明确报错。
公开开发驱动和正式驱动使用相同Console及law.py/claims.json提交接口。
提交根目录为包含shore_agent、rift和提交清单的完整工作区，不再只复制examples目录。
正式测评直接运行本队当前项目；测评期间停止编辑代码、配置及所用权重。
模型从本队配置的实际目录加载，正式控制台地址及会话令牌由指挥部运行器提供。
九宇宙环境由指挥部接入后，使用scripts/evaluate_submission.py统一启动；
正式模拟、固定积分评分和隐藏运行材料由指挥部管理，不要求重新封装本队镜像。
本包不含正式九宇宙的真实模拟器、定律参数或隐藏验证真值，也不附带权重和凭证。
六公开运行的观测及日志可保留；九隐藏运行材料由指挥部保留，赛中只返回规定成绩和必要错误。
'''

PUBLIC_EXPLORATION_FIELDS = (
    "protocol_version", "updated", "scoring_contract", "feedback_contract",
    "evaluation_feedback", "mission", "time_model", "first_light", "launch_request",
    "probe_model", "readout_semantics", "exploration_limits", "fitting", "submission",
    "common_claim_semantics", "submission_evaluation", "evaluation_resources",
    "development_support", "communication_policy", "experimental_error_handling",
    "background_prediction", "candidate_metadata_explanation", "private_simulator_boundary",
    "observation_noise_contract", "observation_noise_version", "runtime_contract",
    "outer_submission_contract",
)


def _public_bytes(relative, data):
    if relative == "docs/contracts/exploration_protocol.json":
        source = json.loads(data.decode("utf-8-sig"))
        public = {key: source[key] for key in PUBLIC_EXPLORATION_FIELDS if key in source}
        if "launch_request" in public:
            public["launch_request"]["semantics"] = "数组每项是共享顶层时间安排及源侧控制的一枚探针；不同顶层安排采用不同调用"
        return (json.dumps(public, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
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
    listed = sorted(set(payloads) | {"FILE_LIST.txt", "starter-manifest.json"})
    payloads["FILE_LIST.txt"] = ("攻坚组开发包文件清单（不含九宇宙真实模拟器/真值）\n"
        + "文件总数：" + str(len(listed)) + "\n\n" + "\n".join(listed) + "\n").encode("utf-8")
    inventory = [{"path": path, "sha256": hashlib.sha256(data).hexdigest()}
                 for path, data in sorted(payloads.items())]
    manifest = {"starter_schema_version": "1.2.0", "contains_formal_world_simulators": False,
                "contains_public_development_worlds": True, "public_worlds": ["1", "2", "3", "4", "5", "6"],
                "contains_editable_llm_baseline": True,
                "contains_model_weights": False,
                "formal_execution_mode": "direct_project",
                "formal_entrypoint": "scripts/evaluate_submission.py",
                "requires_external_organizer_runtime": True,
                "files": inventory}
    manifest["public_contract_projection"] = "Normative public contracts only; internal design reviews excluded; schema identifiers use the participant namespace. Scoring and numerical rules are unchanged."
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
