#!/usr/bin/env bash
# Connected CPU preparation only: build CUDA12.8 resources for an offline H200 image.
# No GPU, model weights or API calls are needed on this instance.
set -Eeuo pipefail
umask 077

BASE='/inspire/hdd/project/26-the-other-shore/fengkairui-25026'
ENVROOT='/opt/shore-h200-cu128'
TOOLROOT='/opt/shore-h200-toolchain'
SCRIPTS='/opt/shore-image-tools'
INDEX_URL="${SHORE_PIP_MIRROR:-}"
TRUSTED_HOST="${SHORE_PIP_MIRROR_HOST:-}"
PLAN=0
DEPENDENCIES_ONLY=0
# These independent environment helpers share a fixed, previously published revision.
DEPENDENCY_SOURCE_REF='20a87a71128be6e7ce577133f825b31149884ca8'
PHASE='解析参数'

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
stage() { PHASE="$1"; printf '\n[SHORE CPU] %s\n' "$PHASE"; }
failed() {
    local status="$1" line="$2"
    # Do not print BASH_COMMAND or the environment: either may contain credentials.
    printf '\nERROR: 阶段“%s”在脚本第%s行失败（退出码%s）。已停止，未进入后续步骤。\n' \
        "$PHASE" "$line" "$status" >&2
    exit "$status"
}
trap 'failed "$?" "$LINENO"' ERR

usage() {
    printf '%s\n' \
        'Usage: bash prepare_cpu_h200.sh --index-url URL --trusted-host HOST' \
        '       [--dependencies-only] [--base-dir /absolute/project/directory] [--plan]' \
        '--dependencies-only installs under /opt without accessing project/model paths or test bundles.' \
        'The default base contains sibling DiscoverPhysics and test repositories.' \
        'Mirror values may also come from SHORE_PIP_MIRROR/SHORE_PIP_MIRROR_HOST.' \
        '--plan prints paths without network, mkdir, package installation or GPU checks.' \
        'Requires Linux and an existing Conda installation; full project preparation also requires findmnt.'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --index-url|--trusted-host|--base-dir)
            [ "$#" -ge 2 ] && [ -n "$2" ] && [[ "$2" != --* ]] || \
                die "缺少参数值：$1"
            case "$1" in
                --index-url) INDEX_URL="$2" ;;
                --trusted-host) TRUSTED_HOST="$2" ;;
                --base-dir) BASE="$2" ;;
            esac
            shift 2 ;;
        --plan) PLAN=1; shift ;;
        --dependencies-only) DEPENDENCIES_ONLY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "未知选项：$1；使用--help查看用法。" ;;
    esac
done

[[ "$BASE" == /* ]] || die '--base-dir必须是绝对路径。'
while [[ "$BASE" == */ && "$BASE" != / ]]; do BASE="${BASE%/}"; done
[ "$BASE" != / ] || die '不能把文件系统根目录作为项目目录。'
case "/${BASE#/}/" in
    */../*|*/./*) die '--base-dir不能包含.或..路径分量。' ;;
esac
[[ "$BASE" != *$'\n'* && "$BASE" != *$'\r'* ]] || die '项目路径不能含换行。'

PROJECT="$BASE/DiscoverPhysics"
ASSETS="$BASE/test"
TESTROOT="$BASE/shore-organizer-open-model-test-20261101"

printf '%s\n' \
    "主项目：$PROJECT" \
    "测试仓库：$ASSETS" \
    "测试包：$TESTROOT" \
    "环境：$ENVROOT" \
    "工具链：$TOOLROOT" \
    "检查脚本：$SCRIPTS"
if [ "$DEPENDENCIES_ONLY" -eq 1 ]; then
    printf '模式：仅安装依赖和离线资源，不访问或创建项目/模型目录。\n'
else
    printf '模式：完整项目准备，需要项目共享盘。\n'
fi

if [ "$PLAN" -eq 1 ]; then
    printf '\nPLAN ONLY: 未检查GPU、未创建目录、未联网、未安装依赖。\n'
    exit 0
fi

[ -n "$INDEX_URL" ] || die '请传入--index-url或设置SHORE_PIP_MIRROR。'
[ -n "$TRUSTED_HOST" ] || die '请传入--trusted-host或设置SHORE_PIP_MIRROR_HOST。'
case "$INDEX_URL" in
    http://*|https://*) ;;
    *) die '--index-url必须使用http://或https://。' ;;
esac
[[ "$INDEX_URL" != *$'\n'* && "$INDEX_URL" != *$'\r'* &&
   "$TRUSTED_HOST" != *$'\n'* && "$TRUSTED_HOST" != *$'\r'* ]] || \
    die '索引参数不能含换行。'

if [ "$DEPENDENCIES_ONLY" -eq 0 ]; then
stage '1/7 检查项目共享盘'
command -v findmnt >/dev/null || die '缺少findmnt，请在CPU实例安装或加载util-linux。'

check_shared_mount() {
    local path="$1" filesystem mount_target
    filesystem="$(findmnt -n -T "$path" -o FSTYPE)" || \
        die "无法查询挂载：$path"
    case "$filesystem" in
        ''|overlay|tmpfs|ramfs)
            die "路径未处于所需共享磁盘：$path（文件系统：${filesystem:-未知}）。请先挂载平台项目盘，不会在临时文件系统创建同名项目。" ;;
    esac
    mount_target="$(findmnt -n -T "$path" -o TARGET)" || \
        die "无法查询挂载点：$path"
    [ -n "$mount_target" ] && [ "$mount_target" != / ] || \
        die "路径只属于实例根文件系统：$path。请先挂载平台项目盘，不会创建同名项目目录。"
    findmnt -T "$path" -o TARGET,SOURCE,FSTYPE
}

# A missing user subdirectory is fine only if its nearest existing ancestor is mounted.
ANCESTOR="$BASE"
while [ ! -d "$ANCESTOR" ]; do
    if [ -e "$ANCESTOR" ] || [ -L "$ANCESTOR" ]; then
        die "项目路径中的现有对象不是可用目录：$ANCESTOR"
    fi
    ANCESTOR="$(dirname -- "$ANCESTOR")"
done
check_shared_mount "$ANCESTOR"
if [ ! -d "$BASE" ]; then
    printf '共享盘已核验，创建项目子目录：%s\n' "$BASE"
    mkdir -p -- "$BASE"
fi
check_shared_mount "$BASE"
else
    stage '1/7 仅准备依赖，跳过项目共享盘和模型目录检查'
fi
mkdir -p -- "$SCRIPTS"

stage '2/7 查找Conda并安装CUDA12.8工具链'
CONDA_BIN="${CONDA_EXE:-}"
if [ ! -x "$CONDA_BIN" ]; then CONDA_BIN="$(type -P conda || true)"; fi
if [ ! -x "$CONDA_BIN" ]; then
    # Repository relocation does not relocate the user's original Conda installation.
    CONDA_BIN='/inspire/hdd/project/urbanlowaltitude/fengkairui-25026/houwenzhe/miniconda3/bin/conda'
fi
[ -x "$CONDA_BIN" ] || die '找不到已有Conda；请将CONDA_EXE设为实际conda可执行文件的绝对路径。'

if [ -f "$TOOLROOT/conda-meta/history" ]; then CONDA_ACTION=install; else CONDA_ACTION=create; fi
CONDA_OVERRIDE_CUDA=12.8 "$CONDA_BIN" "$CONDA_ACTION" -y --prefix "$TOOLROOT" \
    --override-channels --channel nvidia/label/cuda-12.8.1 --channel conda-forge \
    'cuda-toolkit=12.8.1' 'gcc_linux-64=12' 'gxx_linux-64=12' \
    'python=3.12' pip git curl make pkg-config

export SHORE_CUDA128_HOME="$TOOLROOT"
export CC="$TOOLROOT/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$TOOLROOT/bin/x86_64-conda-linux-gnu-g++"
export PATH="$TOOLROOT/bin:$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$ENVROOT/.tools/python"
export UV_CACHE_DIR="$ENVROOT/.cache/uv"
export UV_PYTHON_PREFERENCE=only-managed
export SHORE_BUILD_JOBS=2
export GIT_TERMINAL_PROMPT=0
"$TOOLROOT/bin/nvcc" --version
"$TOOLROOT/bin/ptxas" --version
"$CC" --version
"$CXX" --version
if command -v uv >/dev/null; then uv --version; fi

if [ "$DEPENDENCIES_ONLY" -eq 1 ]; then
    stage '3/7 获取独立环境安装与检查脚本'
    for name in bootstrap_cuda128.sh prepare_h200_offline_cache.py verify_h200_cuda128.py pip_index_install.py prepare_vllm_dependencies.py preflight_cuda128_build.py; do
        temporary="$(mktemp "$SCRIPTS/.${name}.download.XXXXXXXX")"
        curl -fL --retry 3 --connect-timeout 30 \
            "https://raw.githubusercontent.com/hxh5159/DiscoverPhysics/$DEPENDENCY_SOURCE_REF/scripts/$name" \
            -o "$temporary"
        if [[ "$name" == *.sh ]]; then
            bash -n "$temporary"
        else
            "$TOOLROOT/bin/python" -c \
                'import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' \
                "$temporary"
        fi
        mv -- "$temporary" "$SCRIPTS/$name"
    done
    printf '%s\n' "$DEPENDENCY_SOURCE_REF" > "$SCRIPTS/dependency-source-commit.txt"
    stage '4/7 跳过仓库同步和测试包解压，仅安装运行依赖'
else
stage '3/7 克隆或更新主项目和测试资料仓库'
sync_public_repo() {
    local url="$1" target="$2"
    if [ -d "$target/.git" ] || [ -f "$target/.git" ]; then
        if ! git -C "$target" diff --quiet || ! git -C "$target" diff --cached --quiet; then
            die "仓库有未提交的本地修改，请先保存后重试；不会覆盖：$target"
        fi
        git -c credential.helper= -C "$target" pull --ff-only
    elif [ ! -e "$target" ] && [ ! -L "$target" ]; then
        git -c credential.helper= clone "$url" "$target"
    else
        die "目标已有非Git内容，请保留并检查，不会自动覆盖：$target"
    fi
}
cd "$BASE"
sync_public_repo https://github.com/hxh5159/DiscoverPhysics.git "$PROJECT"
sync_public_repo https://github.com/hxh5159/test.git "$ASSETS"

stage '4/7 校验并准备指挥部H200测试包'
if [ -f "$TESTROOT/.test-bundle-receipt.json" ]; then
    python3 "$ASSETS/prepare_test_bundle.py" --bundle h200 --target "$TESTROOT" --refresh
else
    python3 "$ASSETS/prepare_test_bundle.py" --bundle h200 --target "$TESTROOT"
fi

for name in bootstrap_cuda128.sh prepare_h200_offline_cache.py verify_h200_cuda128.py pip_index_install.py prepare_vllm_dependencies.py preflight_cuda128_build.py; do
    [ -f "$PROJECT/scripts/$name" ] || die "主仓库缺少准备脚本：scripts/$name"
    cp -- "$PROJECT/scripts/$name" "$SCRIPTS/"
done
git -C "$PROJECT" rev-parse HEAD > "$SCRIPTS/project-commit.txt"
fi

stage '5/7 安装实验依赖并从源码构建H200推理环境'
if [ "$DEPENDENCIES_ONLY" -eq 1 ]; then cd "$SCRIPTS"; else cd "$PROJECT"; fi
bash "$SCRIPTS/bootstrap_cuda128.sh" \
    --cpu-build-for h200 --env-root "$ENVROOT" \
    --index-url "$INDEX_URL" --trusted-host "$TRUSTED_HOST"

stage '6/7 下载并校验离线内核资源'
PY="$ENVROOT/.venv-trial/bin/python"
"$PY" "$SCRIPTS/prepare_h200_offline_cache.py" --env-root "$ENVROOT" \
    --index-url "$INDEX_URL" --trusted-host "$TRUSTED_HOST"
"$PY" "$SCRIPTS/prepare_h200_offline_cache.py" --env-root "$ENVROOT" --verify

stage '7/7 检查CPU准备结果'
[ -f "$ENVROOT/activate_h200_offline.sh" ] || die '缺少H200离线激活脚本，不能视为准备完成。'
"$PY" -c 'import sys,numpy,scipy; print(sys.executable); print(numpy.__version__,scipy.__version__)'
printf '\n依赖与离线资源准备完成。环境和检查脚本目录：\n'
printf '%s\n' "$ENVROOT" "$TOOLROOT" "$SCRIPTS"
printf '下一步可在H200中执行环境验收和模型推理；当前CPU步骤未运行GPU或模型。\n'
