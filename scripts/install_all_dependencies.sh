#!/usr/bin/env bash
# One command for all application dependencies of the current CUDA12.8/H200 baseline.
# Run on a connected Linux x86_64 CPU instance with an existing Conda installation.
# Covers inference, experiment/fitting/scoring libraries and offline inference assets.
# Does not change drivers, container daemons, project/model files or platform images.
# Custom SFT/RL stacks are outside the baseline dependency profile.
set -Eeuo pipefail
umask 077

ENV_ROOT='/opt/shore-h200-cu128'
TOOL_ROOT='/opt/shore-h200-toolchain'
SCRIPT_ROOT='/opt/shore-image-tools'
INDEX_URL="${SHORE_PIP_MIRROR:-}"
TRUSTED_HOST="${SHORE_PIP_MIRROR_HOST:-}"
CONDA_BIN="${CONDA_EXE:-}"
JOBS="${SHORE_BUILD_JOBS:-2}"
PLAN=0
CHECK_ONLY=0
PHASE='参数检查'
LOG_FILE=''
SOURCE_REF='31b9e13bfb55455d0fc99f6fbcfcbe99ea075871'

die() { printf 'ERROR [%s]: %s\n' "$PHASE" "$*" >&2; exit 2; }
stage() { PHASE="$1"; printf '\n========== %s ==========\n' "$PHASE"; }
failed() {
    local code="$1" line="$2"
    printf '\n安装未完成：阶段“%s”，脚本第%s行，退出码%s。\n' "$PHASE" "$line" "$code" >&2
    if [ -n "$LOG_FILE" ]; then printf '安装日志：%s\n' "$LOG_FILE" >&2; fi
    printf '已停止后续步骤；保留已有文件和环境，不会自动删除。\n' >&2
    exit "$code"
}
trap 'failed "$?" "$LINENO"' ERR

usage() {
    cat <<'USAGE'
Usage:
  bash install_all_dependencies.sh --index-url URL --trusted-host HOST
  bash install_all_dependencies.sh --plan
  bash install_all_dependencies.sh --check

Options:
  --env-root PATH     Python environments/cache root (default /opt/shore-h200-cu128)
  --tool-root PATH    Conda CUDA/compiler toolchain (default /opt/shore-h200-toolchain)
  --scripts-dir PATH  Verified environment helpers (default /opt/shore-image-tools)
  --conda-exe PATH    Existing Conda executable; otherwise discover CONDA_EXE/PATH
  --jobs NUMBER       CPU build jobs (default SHORE_BUILD_JOBS or 2)
  --plan              Print the complete installation plan; no filesystem/network changes
  --check             Check installed packages/cache locally; no installation or GPU calls

The index may also be supplied by SHORE_PIP_MIRROR/SHORE_PIP_MIRROR_HOST.
Installs Python3.12, NumPy2.1.3, SciPy1.15.3, pytest, CUDA Toolkit12.8.1,
GCC/G++12, PyTorch2.11.0+cu128, torchvision/torchaudio0.26/2.11,
vLLM0.24.0 from source, Transformers5.8.0 and resolved dependencies,
plus FlashInfer0.6.12/cu128 offline kernel resources and verification tools.

Only the /opt roots above are used. No project/model mount or GPU is required.
Conda and network access must already be available for installation. Dedicated
training stacks, host GPU drivers and Docker/NVIDIA container services are not installed.
Protected incomplete/user environments are preserved. Use a fresh --env-root if
the existing build is rejected; the script does not erase or silently replace it.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --index-url|--trusted-host|--env-root|--tool-root|--scripts-dir|--conda-exe|--jobs)
            [ "$#" -ge 2 ] && [ -n "$2" ] && [[ "$2" != --* ]] || die "缺少参数值：$1"
            case "$1" in
                --index-url) INDEX_URL="$2" ;;
                --trusted-host) TRUSTED_HOST="$2" ;;
                --env-root) ENV_ROOT="$2" ;;
                --tool-root) TOOL_ROOT="$2" ;;
                --scripts-dir) SCRIPT_ROOT="$2" ;;
                --conda-exe) CONDA_BIN="$2" ;;
                --jobs) JOBS="$2" ;;
            esac
            shift 2 ;;
        --plan) PLAN=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "未知选项：$1；使用--help查看用法。" ;;
    esac
done

valid_root() {
    local path="$1"
    [[ "$path" == /* && "$path" != / ]] || die "目录必须是非根绝对路径：$path"
    [[ "$path" != *//* ]] || die '目录不能包含重复斜杠。'
    case "/${path#/}/" in */../*|*/./*) die '目录不能包含.或..路径分量。' ;; esac
    [[ "$path" != *$'\n'* && "$path" != *$'\r'* ]] || die '目录不能含换行。'
}
while [[ "$ENV_ROOT" == */ && "$ENV_ROOT" != / ]]; do ENV_ROOT="${ENV_ROOT%/}"; done
while [[ "$TOOL_ROOT" == */ && "$TOOL_ROOT" != / ]]; do TOOL_ROOT="${TOOL_ROOT%/}"; done
while [[ "$SCRIPT_ROOT" == */ && "$SCRIPT_ROOT" != / ]]; do SCRIPT_ROOT="${SCRIPT_ROOT%/}"; done
for path in "$ENV_ROOT" "$TOOL_ROOT" "$SCRIPT_ROOT"; do valid_root "$path"; done
[ "$ENV_ROOT" != "$TOOL_ROOT" ] && [ "$ENV_ROOT" != "$SCRIPT_ROOT" ] && \
    [ "$TOOL_ROOT" != "$SCRIPT_ROOT" ] || die '环境、工具链、脚本目录必须不同。'
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die '--jobs必须为正整数。'
TRIAL_PY="$ENV_ROOT/.venv-trial/bin/python"
MODEL_PY="$ENV_ROOT/.venv-vllm-cu128/bin/python"

if [ "$PLAN" -eq 1 ]; then
    printf '%s\n' \
        '完整安装计划（未安装、未联网、未创建目录）：' \
        '1. 检查Linux/已有Conda，准备独立CUDA12.8/GCC12/Python3.12工具链。' \
        '2. 获取固定版本且校验SHA256的环境安装、离线缓存、H200检查脚本。' \
        '3. 检查或安装uv，创建实验和推理两套Python3.12环境。' \
        '4. 安装NumPy/SciPy/pytest以及PyTorch2.11+cu128。' \
        '5. 安装构建依赖，从源码编译vLLM0.24，安装Transformers5.8及其依赖。' \
        '6. 下载并校验FlashInfer0.6.12/cu128离线推理资源。' \
        '7. 检查两环境版本、pip依赖、CPU计算、构建清单和离线文件完整性。' \
        "实验Python：$TRIAL_PY" "推理Python：$MODEL_PY" \
        "工具链：$TOOL_ROOT" "检查脚本：$SCRIPT_ROOT" \
        '无需项目共享盘或GPU；不下载模型、不修改宿主驱动/容器服务。'
    exit 0
fi

verify_all() {
    local program
    for program in nvcc ptxas x86_64-conda-linux-gnu-gcc x86_64-conda-linux-gnu-g++ uv git curl; do
        [ -x "$TOOL_ROOT/bin/$program" ] || die "工具链组件缺失：$TOOL_ROOT/bin/$program"
    done
    "$TOOL_ROOT/bin/nvcc" --version | grep -Eq 'release 12\.8([,[:space:]]|$)' || die 'nvcc版本不是CUDA12.8。'
    "$TOOL_ROOT/bin/ptxas" --version | grep -Eq 'release 12\.8([,[:space:]]|$)' || die 'ptxas版本不是CUDA12.8。'
    "$TOOL_ROOT/bin/x86_64-conda-linux-gnu-gcc" -dumpversion | grep -Eq '^12([.]|$)' || die 'GCC版本不是12。'
    "$TOOL_ROOT/bin/x86_64-conda-linux-gnu-g++" -dumpversion | grep -Eq '^12([.]|$)' || die 'G++版本不是12。'
    [ -x "$TRIAL_PY" ] || die "实验环境不存在或不可执行：$TRIAL_PY"
    [ -x "$MODEL_PY" ] || die "推理环境不存在或不可执行：$MODEL_PY"
    [ -f "$ENV_ROOT/activate_h200_offline.sh" ] || die '缺少H200离线激活文件。'
    [ -f "$SCRIPT_ROOT/prepare_h200_offline_cache.py" ] || die '缺少离线资源校验脚本。'
    printf '%s  %s\n' 925e21a9cef1ec297f03c5136ff6020078fd59d8a2765fa69e3e54cfd0cade7a \
        "$SCRIPT_ROOT/prepare_h200_offline_cache.py" | sha256sum -c -
    # Preserve the CUDA12.8/compiler runtime libraries; never use Toolkit stubs.
    local directory inherited
    local -a library_paths=() old_paths=()
    for directory in "$TOOL_ROOT/lib64" "$TOOL_ROOT/lib" "$TOOL_ROOT/targets/x86_64-linux/lib"; do
        [ ! -d "$directory" ] || library_paths+=("$directory")
    done
    IFS=: read -r -a old_paths <<< "${LD_LIBRARY_PATH:-}"
    for inherited in "${old_paths[@]}"; do
        case "$inherited/" in */stubs/*|/) ;; *) library_paths+=("$inherited") ;; esac
    done
    if [ "${#library_paths[@]}" -gt 0 ]; then
        LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${library_paths[*]}")"; export LD_LIBRARY_PATH
    else
        unset LD_LIBRARY_PATH
    fi
    "$TRIAL_PY" -m pip check
    "$MODEL_PY" -m pip check
    "$TRIAL_PY" -c 'import sys,numpy,scipy; from scipy.special import k1; assert sys.version_info[:2]==(3,12); assert numpy.__version__=="2.1.3"; assert scipy.__version__=="1.15.3"; assert numpy.isfinite(k1(1.)); print("实验环境:",sys.executable,"NumPy",numpy.__version__,"SciPy",scipy.__version__)'
    # CPU tensors only. No vLLM import, CUDA initialization or model loading.
    "$MODEL_PY" - <<'PY_MODEL'
import importlib.metadata as md
import sys
import torch

assert sys.version_info[:2] == (3, 12)
versions = {name: md.version(name) for name in (
    'torch', 'vllm', 'transformers', 'flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache')}
assert versions['torch'] == '2.11.0+cu128', versions
assert torch.version.cuda == '12.8', torch.version.cuda
assert (torch.ones((2, 2), device='cpu') @ torch.ones((2, 2), device='cpu')).tolist() == [[2., 2.], [2., 2.]]
assert versions['vllm'].split('+')[0] == '0.24.0', versions
assert versions['transformers'] == '5.8.0', versions
assert versions['flashinfer-python'] == versions['flashinfer-cubin'] == '0.6.12', versions
assert versions['flashinfer-jit-cache'] == '0.6.12+cu128', versions
print('推理环境:', sys.executable)
for name, version in versions.items():
    print(name + '=' + version)
PY_MODEL
    "$TRIAL_PY" - "$ENV_ROOT" <<'PY_BUILD'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
build = json.loads((root / 'cuda128-build-manifest.json').read_text())
expected = {'target': 'h200', 'target_cuda_arch': '9.0', 'torch_cuda': '12.8',
            'cpu_stage_completed': True, 'gpu_validation_performed': False, 'model_loaded': False}
assert all(type(build.get(k)) is type(v) and build.get(k) == v for k, v in expected.items()), 'CPU构建清单不完整或不匹配'
records = build.get('stable_extensions')
assert isinstance(records, list) and records, '缺少vLLM编译扩展清单'
stems = set()
model_root = (root / '.venv-vllm-cu128').resolve()
for item in records:
    path = (root / item['path']).resolve()
    assert path.is_relative_to(model_root) and path.is_file(), '编译扩展路径错误'
    assert path.name.startswith(item['stem']) and path.suffix == '.so', '编译扩展名称错误'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    assert path.stat().st_size == item['size'] and digest.hexdigest() == item['sha256'], '编译扩展文件已变化'
    stems.add(item['stem'])
assert stems == {'_C_stable_libtorch', '_moe_C_stable_libtorch'}, 'vLLM编译扩展不全'
print('CPU构建清单和vLLM编译扩展SHA256通过。')
PY_BUILD
    "$TRIAL_PY" "$SCRIPT_ROOT/prepare_h200_offline_cache.py" --env-root "$ENV_ROOT" --verify
}

export PYTHONDONTWRITEBYTECODE=1
if [ "$CHECK_ONLY" -eq 1 ]; then
    stage '检查现有依赖（不安装、不联网、不加载GPU或模型）'
    verify_all
    printf '\n现有应用依赖与离线资源检查通过；GPU和真实模型运行另行验收。\n'
    exit 0
fi

[ -n "$INDEX_URL" ] && [ -n "$TRUSTED_HOST" ] || die '请提供--index-url和--trusted-host，或设置对应SHORE_PIP环境变量。'
case "$INDEX_URL" in http://*|https://*) ;; *) die '--index-url必须为HTTP/HTTPS地址。' ;; esac
[[ "$INDEX_URL$TRUSTED_HOST" != *$'\n'* && "$INDEX_URL$TRUSTED_HOST" != *$'\r'* ]] || \
    die '索引参数不能含换行。'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die '实际安装仅适用于Linux x86_64 CPU实例。'
command -v getconf >/dev/null || die '缺少系统基础命令getconf。'
HOST_GLIBC="$(getconf GNU_LIBC_VERSION)" || die '无法读取系统glibc版本。'
[[ "$HOST_GLIBC" =~ ^glibc[[:space:]]+([0-9]+)\.([0-9]+) ]] || die '需要glibc>=2.28的Linux实例。'
(( BASH_REMATCH[1] > 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] >= 28) )) || \
    die '需要glibc>=2.28；请在符合条件的实例中安装。'

stage '1/7 检查已有Conda及安装目录'
if [ ! -x "$CONDA_BIN" ]; then CONDA_BIN="$(type -P conda || true)"; fi
if [ ! -x "$CONDA_BIN" ]; then
    for candidate in "${CONDA_PREFIX:+$CONDA_PREFIX/bin/conda}" \
        "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" /opt/conda/bin/conda /root/miniconda3/bin/conda \
        /inspire/hdd/project/urbanlowaltitude/fengkairui-25026/houwenzhe/miniconda3/bin/conda; do
        if [ -x "$candidate" ]; then CONDA_BIN="$candidate"; break; fi
    done
fi
[ -x "$CONDA_BIN" ] || die '找不到已有Conda；可用--conda-exe指定本实例实际Conda可执行文件。'
for directory in "$ENV_ROOT" "$TOOL_ROOT" "$SCRIPT_ROOT"; do
    [ ! -L "$directory" ] || die "安装根目录是符号链接，请使用其实际路径：$directory"
    [ ! -e "$directory" ] || [ -d "$directory" ] || die "安装路径不是目录：$directory"
done
mkdir -p -- "$ENV_ROOT" "$SCRIPT_ROOT"
command -v tee >/dev/null || die '缺少系统基础命令tee。'
LOG_FILE="$(mktemp "$ENV_ROOT/install-all-dependencies.XXXXXXXX.log")"
exec > >(tee -a "$LOG_FILE") 2>&1
printf '完整安装日志：%s\n实验环境：%s\n推理环境：%s\n' "$LOG_FILE" "$TRIAL_PY" "$MODEL_PY"
# Prevent two installer processes from changing the same environments concurrently.
if command -v flock >/dev/null; then
    exec 9> "$ENV_ROOT/install-all-dependencies.lock"
    flock -n 9 || die '相同环境已有安装进程运行，请等待该进程结束。'
fi

stage '2/7 安装CUDA12.8.1、GCC/G++12及基础构建工具'
if [ -f "$TOOL_ROOT/conda-meta/history" ]; then CONDA_ACTION=install; else CONDA_ACTION=create; fi
CONDA_ARGS=()
if [ "$CONDA_ACTION" = create ]; then CONDA_ARGS+=(--no-default-packages); fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
CONDA_SUBDIR=linux-64 CONDA_OVERRIDE_CUDA=12.8 "$CONDA_BIN" "$CONDA_ACTION" -y --prefix "$TOOL_ROOT" "${CONDA_ARGS[@]}" \
    --override-channels --channel https://conda.anaconda.org/nvidia/label/cuda-12.8.1 \
    --channel https://conda.anaconda.org/conda-forge \
    'cuda-toolkit=12.8.1' 'gcc_linux-64=12' 'gxx_linux-64=12' \
    'python=3.12' pip uv git curl make pkg-config
export SHORE_CUDA128_HOME="$TOOL_ROOT"
export CC="$TOOL_ROOT/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$TOOL_ROOT/bin/x86_64-conda-linux-gnu-g++"
export PATH="$TOOL_ROOT/bin:$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$ENV_ROOT/.tools/python"
export UV_CACHE_DIR="$ENV_ROOT/.cache/uv"
export UV_PYTHON_PREFERENCE=only-managed
export SHORE_BUILD_JOBS="$JOBS"
export GIT_TERMINAL_PROMPT=0
"$TOOL_ROOT/bin/nvcc" --version
"$TOOL_ROOT/bin/ptxas" --version
"$CC" --version
"$CXX" --version
command -v sha256sum >/dev/null || die '缺少系统基础命令sha256sum。'

stage '3/7 自动准备固定版本安装组件并校验SHA256'
fetch_helper() {
    local name="$1" expected="$2" target="$SCRIPT_ROOT/$1" temporary
    [ ! -L "$target" ] || die "组件路径是符号链接，已保留：$target"
    if [ -e "$target" ]; then
        [ -f "$target" ] || die "组件路径不是普通文件：$target"
        if printf '%s  %s\n' "$expected" "$target" | sha256sum -c - >/dev/null 2>&1; then
            printf '复用已校验组件：%s\n' "$name"
            return
        fi
        die "现有组件与固定版本不同，已保留；请用新的--scripts-dir：$target"
    fi
    temporary="$(mktemp "$SCRIPT_ROOT/.${name}.download.XXXXXXXX")"
    curl -fL --retry 3 --connect-timeout 30 \
        "https://raw.githubusercontent.com/hxh5159/DiscoverPhysics/$SOURCE_REF/scripts/$name" \
        -o "$temporary"
    printf '%s  %s\n' "$expected" "$temporary" | sha256sum -c -
    if [[ "$name" == *.sh ]]; then
        bash -n "$temporary"
    else
        "$TOOL_ROOT/bin/python" -c 'import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' "$temporary"
    fi
    mv -- "$temporary" "$target"
}
fetch_helper bootstrap_cuda128.sh 0da6d45886f6e67e3247b49bbc8977ac00a91c997cfece4d9551f42cafc78443
fetch_helper prepare_h200_offline_cache.py 925e21a9cef1ec297f03c5136ff6020078fd59d8a2765fa69e3e54cfd0cade7a
fetch_helper verify_h200_cuda128.py 5a1ebc8e5fbe1fc726701a8748f53e665ee05854befadccf334cad2ebae1ffc8

stage '4/7 安装uv/Python3.12、NumPy/SciPy、PyTorch和vLLM/Transformers'
printf '%s\n' '此阶段会真实执行pip安装及vLLM源码编译；首次安装可能耗时较长，输出持续写入日志。'
bash "$SCRIPT_ROOT/bootstrap_cuda128.sh" \
    --cpu-build-for h200 --env-root "$ENV_ROOT" \
    --index-url "$INDEX_URL" --trusted-host "$TRUSTED_HOST"

stage '5/7 安装并下载FlashInfer/cu128离线内核资源'
"$TRIAL_PY" "$SCRIPT_ROOT/prepare_h200_offline_cache.py" --env-root "$ENV_ROOT" \
    --index-url "$INDEX_URL" --trusted-host "$TRUSTED_HOST"

stage '6/7 检查两套环境版本、pip依赖和离线文件'
verify_all

stage '7/7 写入本次成功凭据'
"$TRIAL_PY" - "$ENV_ROOT" "$MODEL_PY" "$TOOL_ROOT" "$SCRIPT_ROOT" "$LOG_FILE" "$SOURCE_REF" <<'PY'
import importlib.metadata as md
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

root, model_python, tool_root, script_root, log_file, revision = sys.argv[1:]
code = 'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ("torch","vllm","transformers","flashinfer-python","flashinfer-jit-cache")}))'
model = json.loads(subprocess.check_output([model_python, '-c', code], text=True))
receipt = {
    'status': 'application_dependencies_installed_cpu_checks_passed',
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'target': 'Linux x86_64 / CUDA12.8 / H200 SM90',
    'trial_python': sys.executable, 'model_python': model_python,
    'trial_versions': {n: md.version(n) for n in ('numpy', 'scipy', 'pytest')},
    'model_versions': model, 'tool_root': tool_root, 'script_root': script_root,
    'helper_source_revision': revision, 'log': log_file,
    'gpu_or_model_execution_performed': False,
}
path = Path(root) / 'install-all-dependencies-success.json'
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
temporary.replace(path)
print('成功凭据：' + str(path))
PY
printf '\n全部基础应用依赖及离线资源安装完成（CPU检查通过）。\n'
printf '实验/拟合/评分Python：%s\n模型推理Python：%s\n安装日志：%s\n' "$TRIAL_PY" "$MODEL_PY" "$LOG_FILE"
printf '再次只读核查可执行本脚本加 --check；GPU与实际模型加载在H200中另行验收。\n'
