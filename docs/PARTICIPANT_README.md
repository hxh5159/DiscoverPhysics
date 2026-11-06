# 彼岸：物理规律发现挑战赛

本文件帮助攻坚组了解任务、安装环境和开始自测。完整赛题见 [赛题说明](docs/彼岸物理规律发现挑战赛_赛题说明_更新版.pdf)；操作细节见 [提交规范](docs/STRATEGY_SUBMISSION.txt)、[控制台说明](docs/CONSOLE_API.txt) 和 [规律程序说明](docs/LAW_API.txt)。开发包解压后，本文件位于项目根目录 `README.md`。

## 任务与开发环境

裂隙连接着规律未知的彼岸。攻坚组需要让白名单开源LLM自主安排实验、理解带噪观测、提出候选规律并验证，最终自动生成可预测探针运动的程序和三项结构化声明。各宇宙均为二维；同一宇宙的定律参数固定，待拟合常数由本队方法确定，评分端不替队伍拟合。

- **六个开发宇宙**：编号1—6，模拟器和标准评测程序公开，供反复自测、生成合法训练数据及改进策略。开发成绩不计正式排名。
- **九个测评宇宙**：Adam、Lilith、Tabris；Sachiel、Ramiel、Matriel；Leliel、Arael、Lilin，依次分为三档，档位权重为30%、40%、30%，档内等权。真实模拟器和验证真值不交给攻坚组，正式运行时智能体仍可经控制台自动实验。开发与测评宇宙之间没有公开的规律对应关系，不能据开发宇宙直接确定测评规律。
- **统一限制**：开发和测评均固定每宇宙3轮探索迭代，不得修改或绕过；每宇宙10点能量、最多20次免费 `observe`，首次成功 `mission` 后最多5分钟现实经过时间，模型加载另计。每枚新探针耗1点，单次 `launch` 最多8枚，每枚最多50个采样时刻。
- **初始数据**：`mission` 提供操作范围、可见天体初态和免费“第一束光”（2枚标准探针各8个时刻的位置、速度及同期可见背景位置）。后续使用 `observe`、`launch`、`archive` 继续探索，位置和速度观测含噪声，评分真值无观测噪声。

探针是可自由运动的被动测试粒子，不影响背景或其他探针；可在允许范围内选择初始位置、速度、投放与采样时刻，以及任务开放的标量属性。例如，三个固定可见作用源可产生 `a(r)=0.1*Σ(b_i-r)`，探针轨迹由加速度积分得到。这个教学例仅解释“力场—探针—观测”的形式，并非九个测评宇宙的真实规律。

## 环境与依赖

正式运行面向Linux x86_64，实验/拟合与模型推理使用分开的Python环境。推荐使用指挥部提供的基础环境和完整本地模型目录，不在无网运行节点临时下载依赖或权重。

| 用途 | 基础配置 |
| --- | --- |
| 实验、拟合、固定积分与计分 | Python 3.12、NumPy 2.1.3、SciPy 1.15.3，清单为根目录 `requirements.txt` |
| 基础模型推理 | Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、torchaudio 2.11.0+cu128、vLLM 0.24.0、Transformers 5.8.0 |
| 模型环境构建 | CUDA Toolkit 12.8及完整开发头文件、GCC/G++ 11.3—14、git、curl；构建脚本还会安装并检查配套依赖 |
| 模型运行 | 可用GPU、兼容驱动及足够显存；`nvidia-smi` 显示的CUDA版本不等于已安装Toolkit版本 |

本包不要求安装其它研究代码的完整依赖。自行增加的策略依赖应在提交前安装，并在同一环境完成自测；不得因此修改固定接口或评测程序。

### 已有基础环境

在开发包根目录执行，下列变量分别指向实验Python与模型Python：

```bash
PY='/opt/shore-h200-cu128/.venv-trial/bin/python'
MPY='/opt/shore-h200-cu128/.venv-vllm-cu128/bin/python'

"$PY" -c 'import sys,numpy,scipy; print(sys.executable,numpy.__version__,scipy.__version__)'
"$MPY" -c 'import torch,transformers,importlib.metadata as m; print(torch.__version__,torch.version.cuda,m.version("vllm"),transformers.__version__)'
```

若已由指挥部准备上述环境，可直接进入模型配置与运行步骤。包版本检查不等于模型已成功加载，仍需完成实际启动与自测。

### 自行安装

仅安装实验依赖时，在已有Python 3.12的联网实例执行：

```bash
python3.12 -m venv .venv-trial
PY="$PWD/.venv-trial/bin/python"
env -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL \
  -u PIP_FIND_LINKS -u PIP_NO_INDEX \
  PIP_CONFIG_FILE=/dev/null \
  "$PY" -m pip install -r requirements.txt \
    --index-url 'http://nexus.sii.shaipower.online/repository/pypi/simple' \
    --trusted-host 'nexus.sii.shaipower.online' --no-cache-dir
```

需要从头安装模型推理环境时，先准备上表所列完整CUDA Toolkit与编译工具，再运行包内脚本。以下命令在联网CPU实例中为H200安装环境，不要求CPU实例配有GPU：

```bash
# 改为本实例实际存在的CUDA 12.8 Toolkit目录。
export SHORE_CUDA128_HOME='/opt/shore-h200-toolchain'
export CC="$SHORE_CUDA128_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$SHORE_CUDA128_HOME/bin/x86_64-conda-linux-gnu-g++"

ENVROOT="$PWD/.runtime-h200-cu128"
bash scripts/bootstrap_cuda128.sh \
  --cpu-build-for h200 --env-root "$ENVROOT" \
  --index-url 'http://nexus.sii.shaipower.online/repository/pypi/simple' \
  --trusted-host 'nexus.sii.shaipower.online' --mirror-only

python3 scripts/prepare_h200_offline_cache.py \
  --env-root "$ENVROOT" \
  --index-url 'http://nexus.sii.shaipower.online/repository/pypi/simple' \
  --trusted-host 'nexus.sii.shaipower.online' --mirror-only

PY="$ENVROOT/.venv-trial/bin/python"
MPY="$ENVROOT/.venv-vllm-cu128/bin/python"
```

若编译器不在上述目录，应将 `CC`、`CXX` 改为本实例实际GCC/G++路径。Python包从命令指定的内部镜像下载；镜像必须包含所需版本及对应CUDA 12.8的安装包，缺包或网络失败时脚本会停止，不会自动更换CUDA或模型。编译时还可能下载第三方源码及离线推理文件。环境准备细节见 [H200离线环境说明](docs/H200_OFFLINE_SETUP.txt)。

在H200运行实例使用相同环境目录，先执行生成的 `source "$ENVROOT/activate_h200_offline.sh"`，再用 `python3 scripts/verify_h200_cuda128.py --env-root "$ENVROOT"` 检查本地GPU与依赖。模型权重须已完整保存在可读的本地路径。

## 模型白名单与配置

白名单仅保留以下Qwen模型及列出的精度版本，移除397B系列及其它系列模型：

| 模型 | 基础框架预设ID | 预置目录名 |
| --- | --- | --- |
| Qwen3.8-27B（BF16） | `27b` | `Qwen3.8-27B` |
| Qwen3.6-35B-A3B（BF16） | `35b` | `Qwen3.6-35B-A3B` |
| Qwen3.6-35B-A3B-FP8 | `35b-fp8` | `Qwen-3.6-35B-A3B-FP8`，兼容不带额外连字符的别名 |
| Qwen3.5-122B-A10B（BF16） | `122b` | `Qwen3.5-122B-A10B` |
| Qwen3.5-122B-A10B-FP8 | `122b-fp8` | `Qwen3.5-122B-A10B-FP8` |

为避免在模型选择上带来分歧和差距，故保留以上几个模型，同时考虑算力资源和微调所需算力，不再保留需要4卡才能加载的模型。本说明针对赛题的H200资源配置；实际卡数还取决于精度、上下文和推理/训练方式。基础程序默认使用4卡并行，并不表示每个保留模型都必须使用4卡才能加载。

公共模型目录为 `/inspire/hdd/project/26-the-other-shore/public/models`。根目录 `models.json` 默认使用其中的Qwen3.8-27B。选用其它模型、训练后的完整权重或LoRA时，在清单中逐项填写唯一 `id`、白名单模型名称 `base_model`、完整权重路径 `path`，以及需要时的LoRA目录 `adapter_path`，并用 `primary_model_id` 指定主模型。示例见 `examples/models.json`、`examples/models.multi.json`。更换权重后应重新加载模型，避免继续调用旧权重。

同一个白名单模型可以加载多个实例，分别承担规划、拟合、复核等任务。多个实例使用相同的一份权重时，可以复用同一个路径和清单ID，无需重复复制；使用不同的微调权重或LoRA时，必须在 `models.json` 中分别列出，并设置不同的ID。多个模型、多个实例的调用和显存安排均由本队 `agent/run.py` 统一协调，权重文件不必复制到策略目录。

## 开始一次完整自测

以下在开发包根目录执行，`PY`、`MPY` 使用前文准备好的两个解释器。先编辑 `agent/strategy.json` 填写本队身份与版本，并检查 `models.json` 的实际路径：

```bash
"$PY" scripts/validate_strategy.py \
  --agent-dir "$PWD/agent" --models-file "$PWD/models.json" --check-files

"$PY" -m development_worlds.runner \
  --agent-dir "$PWD/agent" --models-file "$PWD/models.json" \
  --model-python "$MPY" --max-turns 3 \
  --output "results/dev-$(date +%Y%m%d-%H%M%S)"
```

检查器不加载大模型。运行程序默认自动加载已登记的主模型，依次运行六个开发宇宙，获取初始任务和第一束光、自动实验、生成 `law.py` 与 `claims.json`，再调用固定积分器和评分程序；汇总在新结果目录的 `summary.json`。可添加 `--worlds 1` 先测试一界，3轮限制保持不变。输出目录必须是新目录，避免覆盖旧结果。

基础框架把一次模型决策计为一轮；模型可在同一次回复中明确安排一组顺序操作，各项实验仍分别扣除配额，工具处理也计入本宇宙的时间限制。第一束光已包含在初始输入中，无需为读取相同数据单独消耗一次决策。候选检查或拟合成功后，程序会保存规律代码并返回 `candidate_id`，此时可以尚未填写声明；完整交卷仍须包含该规律和由模型生成的三项声明。后续可以按编号引用候选，或使用 `law_ref:"last_fit"` 引用最近成功拟合的结果，再附上声明交卷，无需重复输出整份代码。`law_ref:"saved"` 引用最近保存的规律，`law_ref:"draft"` 引用最后完整有效的草稿；程序不会把另一份规律的旧声明自动套用到新候选。拟合只处理模型自行提出的参数，不替模型选择实验、提出规律或填写声明。

本地vLLM服务支持结构化输出时，可在运行前设置 `export SHORE_STRUCTURED_OUTPUT=json_schema`，减少回复中的JSON格式错误；此选项不提供科学答案，也不会因格式错误额外调用模型。服务不支持时应在开发自测中确认并关闭该选项，不会自动切换模型或增加重试。

若已启动对应本地权重服务，可添加 `--base-url http://127.0.0.1:8000/v1` 复用；服务必须与本次清单中的模型、适配器和服务名称一致。训练或更换权重后须重新加载。首次实际加载耗时、显存和模型兼容性以本机运行结果为准。

## 可以编辑哪些文件

| 文件或目录 | 处理方式 |
| --- | --- |
| `agent/run.py`、`agent/models.py`、`agent/tools.py`、`agent/tool_worker.py` | 可改写主循环、模型调用、实验设计与工具策略；统一入口保留为 `agent/run.py` |
| `agent/strategy.json` | 填写本队身份、提交ID和版本 |
| `agent/prompts/`、`agent/memory/`、`agent/configs/`、`agent/agents/` 等 | 可新增；全部运行时自定义策略、提示词、合法记忆、配置和子智能体代码放入 `agent/` |
| 根目录 `models.json` 与本队权重目录 | 可更新；所有实际模型与适配器必须完整登记，测评期间保持不变 |
| `rift/`、`rift_numerics/`、`development_worlds/`、`scripts/`、`shore_agent/`、`docs/contracts/` | 官方固定文件，不得修改实验、积分、计分、轮数限制或启动/检查接口 |

六开发宇宙源码可阅读，并可据此生成训练数据；标准自测仍必须调用固定实验、积分与评测程序。训练程序和训练数据可另行保存，但正式运行需要读取的提示词、记忆、配置和自定义代码必须在 `agent/` 内。允许自行量化、SFT、RL、符号回归及提示/记忆策略，具体方法由各攻坚组确定。

## 自动交卷与正式提交

每个宇宙由智能体自动生成一对文件：

1. **`law.py`**：实现 `acceleration(t, bodies, params)`，返回二维加速度；附带本队已拟合确定的参数。指挥部用固定积分器生成验证轨迹，与无噪声真值比较，不直接采用队伍预计算轨迹。全局可拟合 `PARAMS` 最多5个有限标量，不得用数组冒充单个参数；候选可用math、NumPy、SciPy，不得有外部读写或联网。
2. **`claims.json`**：仅包含下面三个必填字段。必须由白名单LLM根据观测自行判断和生成，禁止人工填写、改写或预置答案。

| 字段 | 类型与含义 |
| --- | --- |
| `dynamics_order` | 整数 `1` 或 `2`：作用决定速度填1，决定加速度填2 |
| `hidden_matter` | 布尔值：是否存在位置不可见、但产生作用的源；可见天体源强未知不属于此项 |
| `time_dependence` | 布尔值：固定完整瞬时状态与属性后，作用是否显式随物理时间变化；普通轨道运动不属于此项 |

以下仅演示JSON格式，不代表任何宇宙的答案，不能作为预置声明：

```json
{"dynamics_order": 1, "hidden_matter": false, "time_dependence": true}
```

向指挥部提交的是**两类绝对路径**：完整 `agent/` 策略目录的路径，以及列全本次模型/LoRA的 `models.json` 文件路径。不是赛前手填九份规律答案，也不要求上传巨大的模型权重文件。提交前完成目录和权重检查，并确认所有训练后的权重已经保存完毕；测评期间禁止编辑本次代码、配置和权重，全程无人干预。

每组最多6次正式测评，六开发宇宙本地自测不限此次数。排名取最好的一次完整测评成绩，不拼接历史各宇宙最高分。赛中仅反馈加权总分与各宇宙总分，不反馈预言/定律/节约细项、声明正误或隐藏运行日志、观测、新记忆与更新权重。

## 计分摘要

每个宇宙满分100分：预言80分、定律15分、节约5分。对全部固定验证探针及采样时刻等权计算二维位置误差：

```text
MSE = mean(||预测位置 - 无噪声真值位置||²)
V   = mean(||真值位置 - 全部真值位置的均值||²)
e   = MSE / max(V, 1e-12)
P   = 80 / (1 + 10e)
```

预言分连续映射，不分档。三个定律声明分别与预存真值精确比较，每项正确5分，不使用LLM裁判；不另设自洽分。仅当 `P>=40` 时，节约分为 `5*(1-E/10)`，否则为0；`E` 是控制台记录的实际能量消耗，不计现实耗时或剩余免费观测次数。总分按三个难度档的30%/40%/30%加权；同分比较加权节约分，较低者优先，仍同则并列。

## 允许范围与禁止事项

- 推理只使用上述白名单模型及其合规量化、微调结果；禁止使用非白名单权重。
- 禁止外网与外部API；官方控制台通信及本机已登记权重服务除外。
- 禁止现成智能体框架和编码智能体：LangChain / LangGraph、AutoGen、CrewAI、AgentScope、Qwen-Agent、MetaGPT / OpenManus、CAMEL、smolagents，以及Codex、Claude Code、Qwen Code、Kimi CLI等。可使用通用推理、科学计算和符号回归库。
- 禁止读取或探测正式控制台内部实现与隐藏真值，禁止绕过能量、时间、迭代轮数或计分限制，禁止修改官方固定程序。
- 正式测评期间禁止人工干预；三项定律声明必须由LLM自动完成。保留本队训练配置、代码和数据生成脚本，供赛后核查。违反模型或框架使用边界将按赛题说明计为0分。

更多操作细节见 `docs/`，六开发宇宙的公开定义及数值边界见 `development_worlds/README.md`。
