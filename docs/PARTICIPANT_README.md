# 彼岸：物理规律发现挑战赛

任务编号：SHORE-PLM-001。本文件整合任务、环境、模型、文件职责、实验与规律接口、提交及计分说明；完整任务背景见[赛题说明](docs/彼岸物理规律发现挑战赛_赛题说明_更新版.pdf)。开发包解压后，本文件位于根目录 `README.md`。先按“开始一次完整自测”操作，需要改写工具时再查阅后面的折叠接口说明，无须逐份阅读重复手册。

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
| 包内自动启动路线 | Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、torchaudio 2.11.0+cu128、vLLM 0.24.0、Transformers 5.8.0 |
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

**两条模型接入路线要分清：** 包内自动加载和 `scripts/serve_model.py` 当前限定vLLM0.24.0与Torch CUDA12.8。已经独立部署、通过真实生成检查的CUDA13.0本机服务，可通过运行器的 `--base-url http://127.0.0.1:8000/v1` 连接；这不表示自动安装、自动加载也已支持cu130。`--model-python`或 `SHORE_MODEL_PYTHON`只是选择解释器，不能解除版本检查。文件夹名字也不能代替 `torch.version.cuda` 的实际结果。

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

若编译器不在上述目录，应将 `CC`、`CXX` 改为本实例实际GCC/G++路径。Python包从命令指定的内部镜像下载；镜像必须包含所需版本及对应CUDA 12.8的安装包，缺包或网络失败时脚本会停止，不会自动更换CUDA或模型。编译时还可能下载第三方源码及离线推理文件。完整工具链、安装网络与离线资源的注意事项见[下方可选构建说明](#cuda128)。

在H200运行实例使用相同环境目录，先执行生成的 `source "$ENVROOT/activate_h200_offline.sh"`，再用 `python3 scripts/verify_h200_cuda128.py --env-root "$ENVROOT"` 检查本地GPU与依赖。模型权重须已完整保存在可读的本地路径。

## 模型白名单与配置

以下列出白名单模型及已提供的预置精度版本；允许依既有规则自行量化。397B系列及其它系列模型不在当前白名单中：

| 模型 | 基础框架预设ID | 预置目录名 |
| --- | --- | --- |
| Qwen3.8-27B（BF16） | `27b` | `Qwen3.8-27B` |
| Qwen3.6-35B-A3B（BF16） | `35b` | `Qwen3.6-35B-A3B` |
| Qwen3.6-35B-A3B-FP8 | `35b-fp8` | `Qwen-3.6-35B-A3B-FP8`，兼容不带额外连字符的别名 |
| Qwen3.5-122B-A10B（BF16） | `122b` | `Qwen3.5-122B-A10B` |
| Qwen3.5-122B-A10B-FP8 | `122b-fp8` | `Qwen3.5-122B-A10B-FP8` |

为避免在模型选择上带来分歧和差距，故保留以上几个模型，同时考虑算力资源和微调所需算力，不再保留需要4卡才能加载的模型。本说明针对赛题的H200资源配置；实际卡数还取决于精度、上下文和推理/训练方式。基础程序默认使用4卡并行，并不表示每个保留模型都必须使用4卡才能加载。

公共模型目录为 `/inspire/hdd/project/26-the-other-shore/public/models`。根目录 `models.json` 默认使用其中的Qwen3.8-27B。选用其它模型、训练后的完整权重或LoRA时，在清单中逐项填写唯一 `id`、白名单模型名称 `base_model`、完整权重路径 `path`，以及需要时的LoRA目录 `adapter_path`，并用 `primary_model_id` 指定主模型。正式提交时，全部 `path` 与 `adapter_path` 必须填写指挥部可读取的共享盘绝对路径。示例见 `examples/models.json`、`examples/models.multi.json`，其中相对路径示例须改为本次实际绝对路径。更换权重后应重新加载模型，避免继续调用旧权重。

同一个白名单模型可以加载多个实例，分别承担规划、拟合、复核等任务。多个实例使用相同的一份权重时，可以复用同一个路径和清单ID，无需重复复制；使用不同的微调权重或LoRA时，必须在 `models.json` 中分别列出，并设置不同的ID。多个模型、多个实例的调用和显存安排均由本队 `agent/run.py` 统一协调，权重文件不必复制到策略目录。

## 开始一次完整自测

以下在开发包根目录执行，`PY`、`MPY` 使用前文准备好的两个解释器。先编辑 `agent/strategy.json` 填写本队身份与版本，并检查 `models.json` 的实际路径。六开发宇宙仍按下面的 `agent/` 和 `models.json` 路径运行，不向本地运行器传ZIP：

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

若已启动对应本地权重服务，可添加 `--base-url http://127.0.0.1:8000/v1` 复用；27B服务还可指定 `--model-id Qwen3.8-27B`，其它模型使用其实际白名单标识。先在同节点检查 `curl -fsS http://127.0.0.1:8000/v1/models`，再确认能真实生成，不能只凭服务名证明已加载正确权重。训练或更换权重后须重新加载；连接被拒绝通常说明服务未启动、已退出或端口错误。

既有服务分支不走cu128自动加载检查，因此可接入已验收的cu130本机服务；但当前基础加载器在该分支不接受同时指定 `adapter_path`。LoRA应采用受支持的自动加载路线，或本队实现并自测的加载流程，不能把“能连服务”当作“适配器已生效”。多个模型的加载顺序、显存和服务协调由本队策略负责；清单不会自动生成多智能体，也不要求所有权重同时装入显存。

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

向指挥部提交的是 **`submission.zip` 文件**。ZIP根目录直接包含完整 `agent/` 目录树和 `models.json` 文件，不再在外层套一个项目目录；只打包这两项，不打包公开实验环境、依赖环境、结果目录或模型权重。`agent/run.py`仍是统一入口，`agent/strategy.json`仍登记本队身份与版本。多智能体、提示词、工具、配置及运行时记忆均随 `agent/` 一并交付。

```text
submission.zip
  agent/
    __init__.py
    run.py
    strategy.json
    models.py、tools.py及本队其它策略文件
    prompts/、memory/等本队运行时资产
  models.json
```

模型权重及LoRA适配器保留在共享盘，不放入ZIP。`models.json`须逐项列全本次使用的模型ID、完整权重与适配器，所有 `path`、`adapter_path` 均为可读取的共享盘绝对路径；不同训练版本、不同LoRA分别登记。不是赛前手填九份规律答案；`law.py`与`claims.json`仍在正式运行中由智能体逐宇宙自动生成。

除根目录models.json外，其余文件均须位于agent/内；禁止符号链接或路径越界成员。组织端默认技术接收上限为解压后1GiB、10000个成员；若合法策略或记忆资产超过，应提前请指挥部调整接收上限。这不是实验配额或计分项，不要通过把权重、依赖环境或大训练数据放进agent来代替模型路径登记。

在开发包根目录执行以下命令。先通过原有路径检查，再用Python标准库打包和检查ZIP；检查器不加载模型，也不能代替完整自测：

```bash
"$PY" scripts/validate_strategy.py \
  --agent-dir "$PWD/agent" --models-file "$PWD/models.json" --check-files &&
test ! -e submission.zip &&
"$PY" -m zipfile -c submission.zip agent models.json &&
"$PY" -m zipfile -t submission.zip
```

已有 `submission.zip` 时上述命令停止，不覆盖旧版本；下一次改用新文件名，例如 `submission-v2.zip`，并相应修改命令中的文件名。打包过程中不要修改策略或清单；只有打包和ZIP检查都成功后才交给指挥部，不能提交仍在写入或中途失败的ZIP。原有路径检查允许开发时使用相对模型路径，正式打包前仍须逐项确认已改为共享盘绝对路径。提交前确认训练检查点均已保存完毕，测评期间禁止编辑本次代码、配置或改写所引用权重，全程无人干预。

**本README的ZIP要求替代旧版赛题PDF和旧说明中的“两条路径交付”要求。** 六开发宇宙的运行命令保持不变；指挥部受理ZIP后私有复制并解压，再按内部 `agent/`、`models.json` 路径启动。该正式ZIP受理入口只在指挥部侧提供，无需修改选手固定运行组件、清单字段结构或六开发命令，也无需把ZIP传给 `development_worlds.runner`。

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

下面汇总其余运行接口、错误处理、模型诊断及六开发宇宙资料。自然语言说明归并不改变固定程序、JSON类型、宇宙或评分规则。


## 文件组织、运行配置与正式输出

```text
参赛包/
  agent/                     全部可修改策略和运行时自定义资产
    __init__.py              包入口
    run.py                   统一入口，协调所有子智能体
    strategy.json            本队身份、本次提交ID和版本
    models.py                模型加载、调用和服务安排
    tools.py / tool_worker.py 本队拟合和工具逻辑
    prompts/ memory/ configs/ agents/  可自行新增
  models.json                所有完整权重及适配器的路径清单
  development_worlds/        六公开环境、控制台和固定评分
  rift/ / rift_numerics/     固定交互、候选规律接口与积分器
  scripts/                   固定启动、安装、检查及模型辅助
  docs/contracts/            数据类型、数值和运行规则
  examples/                  非正式宇宙的格式与配置示例
  requirements.txt           实验、拟合和计分依赖
```

正式运行需要的非权重自定义资产全部位于 `agent/`，包内导入使用 `agent`命名空间，例如 `from agent.agents import planner`；读取提示词、配置和记忆时以当前运行的 `agent/` 文件位置为基准，不写死本组原项目的绝对位置，也不能依赖根目录另一个自编模块或外置提示词。正式运行使用指挥部接收ZIP后复制的策略，原项目位置不作为运行依赖。训练代码、训练数据和六公开开发结果可放在其它位置，但不随正式ZIP打包。`shore_agent/`、旧提交脚本只作兼容，新策略仍从 `agent/run.py`启动。示例文件可复制后改写，不是正式宇宙参考解；两个Dockerfile只是可选构建示例，当前提交不要求镜像。

**提示词修改只能用于推理策略。** 指挥部在私有副本中运行接收的策略，不回写队伍原 `agent/`。正式运行仅允许写入本局指定输出及允许的临时目录；输出位置由 `RIFT_OUTPUT_DIR`提供。隐藏观测、对话、日志、新记忆和更新权重不得写回队伍策略目录、原权重目录或其它队员可取得的位置。公开自测可使用本队 `results/`，不能把该开发目录惯例套用于隐藏运行。训练新权重另存本队可写目录，不改指挥部公共权重。

### 两份清单的字段与路径

`agent/strategy.json`准确包含四个字段，例如：

```json
{"schema_version":"1.0.0","team_id":"team01","submission_id":"run01","version":"1.0.0"}
```

`models.json`包含 `schema_version="1.0.0"`、`primary_model_id`和 `models`数组。每项必填唯一 `id`、白名单 `base_model`和 `path`，可填 `role`、`adaptation`和 `adapter_path`。`adaptation`只接受 `base`、`quantized`、`sft`、`rl`、`other`，LoRA通过adapter_path表达，不填写lora作为adaptation。完整权重只填path；LoRA同时给出底座path和适配器adapter_path。**正式ZIP内全部path与adapter_path均须为指挥部可读取的共享盘绝对路径，不能填相对路径。** 六开发的原有路径解析仍保持不变；若开发时使用相对路径，它相对models.json所在目录解析，但正式打包前必须转换为实际绝对路径。示例见 `examples/models.json`、`examples/models.multi.json`，不要将其中相对路径占位值直接用于正式提交。

team_id、submission_id及模型id使用1—64个ASCII字母、数字、下划线、点或连字符，首字符须为字母或数字；同一清单内模型id唯一。version是本次方案版本字符串，不是模型选择键。

同一份权重的多个实例可共享ID与路径；不同训练结果或不同适配器使用不同ID及实际路径，不能同ID混用。ID和role只是分工名称，不编码宇宙类型。正式通过 `SHORE_MODEL_REGISTRY`取得清单，自定义策略可调用 `rift.strategy.load_model_registry`与 `model_binding`按ID选模型；基础框架默认primary，也可用 `--model-ref`选择已登记ID。路径检查覆盖全部模型和适配器，不因只加载主模型而漏收其它权重。

需要辅助生成清单时运行 `scripts/prepare_strategy.py --help`；已有同名文件不会自动覆盖，确需替换才使用 `--overwrite`。提交前所有检查点都应保存完毕，本次运行期间不再覆盖；下一版训练应写入另一版本目录。

### 一轮的含义与自动流程

基础框架中，一份主模型行动回复计一轮。一次回复可显式安排1至8项工具操作，顺序执行，不能嵌套批次，submit只能最后；工具失败后不继续执行该批剩余项。数值优化内部计算多次，不额外占模型轮数。格式或动作错误已占当前轮，重新生成回复再占一轮。成功交卷可提前结束，已有完整有效配对可由程序直接提交，不追加模型调用。

第一束光已在初始任务中提供；无须先请求同一档案才能使用。`--max-turns 0`虽仍存在于兼容命令行，参赛时不得用它取消三轮限制。当前实现没有统一统计全部子模型调用，不能把本地主循环计数描述成已经完成全模型总次数的技术强制检查。

队员提交含完整agent与模型清单的ZIP，智能体逐界自动交law/claims，是两层不同的提交。指挥部接收、私有复制并解压本次ZIP，按清单读取全部已声明权重；内部运行仍使用agent目录与models.json路径，不改成旧submission.json或容器入口。受理并冻结版本后，启动策略和模型，领取任务与首光，自动实验、拟合、交卷，最后固定积分评分并汇总。策略逐宇宙重新启动，不保证模型进程自动复用；复用服务须确认它属于本次权重。

每组最多6次正式测评，六公开本地自测不占次数。截止以成功受理时间判断，截止前受理可在截止后完成；没有成功受理任何有效版本则记0分。允许复用合法公开开发数据与模型成果，不授予读取历史私有运行目录或其他组数据的权限。榜单使用最佳一次完整成绩，不拼接逐宇宙历史最高分。

## 实验时间、读数与错误处理

| 操作 | 作用与限制 |
| --- | --- |
| mission | 取得合法范围、属性、可见背景初态、配额及首光；首次成功调用开始300秒，重复调用不重置 |
| observe | 指定时刻全部可见背景的位置；免费，每宇宙最多20次，每次最多50时刻 |
| launch | 每枚新探针1点，每次最多8枚，每枚最多50个位置/速度读数时刻 |
| archive | 只返回已取得记录，不生成新观测、不重抽噪声 |
| save_draft | 保存有效规律与声明配对，不结束探索、不扣实验能量 |
| submit | 成功后冻结配对并结束探索 |

第一束光为2枚标准探针各8时刻的位置、速度及同期背景位置，附已知投放配置，免费且缓存，不占observe次数。每枚50时刻分别计算，8枚各50时刻为400条探针状态记录；位置与速度合成一条状态，不按坐标另计。背景50时刻也不是所有背景对象合计50条。gauge不属于当前启用操作，探索采样上限不等于验证样本数上限。

每局共用物理t=0；t_launch是绝对投放时刻，readout是相对投放的非负、严格递增偏移，可包含0。两者分别严格对齐0.005网格，只容忍浮点表示尾差，不把非法时刻舍入到网格。按请求重放同一背景，查询顺序或模型思考不推进物理时间；源侧配置不同的背景不能混用。具体投放区、速度、属性与物理时域以mission为准。探针会自由运动，投放区只约束初始位置，不截断之后的轨迹。

位置读数加入标准差为公开投放域参考长度1%的独立高斯噪声，探针速度每坐标加入公开初速上限0.01%的独立高斯噪声。新实验在不同坐标、探针、时刻独立抽样；第一束光使用同样规则。重读首光或archive不重抽，已知投放初态、共同t=0背景初态和评分真值保持精确。推理、拟合、候选积分不扣实验能量，但占现实探索窗口；物理时间与现实时间都不计入节约分。

| 情况 | 处理 |
| --- | --- |
| 上传格式错误 | 不覆盖前一有效配对，也不结束探索，可以修正 |
| 成功submit | 冻结这次law.py与claims.json配对，结束探索 |
| 超时或策略提前退出 | 取最后及时有效配对；无有效配对则该宇宙总分0 |
| 已有合法配对，但预测运行失败、超时或输出无效 | 预言分0，定律分仍独立判断，不一概清空整个宇宙分数 |
| 指挥部配置、参考或基础设施故障 | 保留受理版本，修复重评，不当作队伍科学错误 |

所有固定验证样本统一纳入位置平方误差平均，不能删掉失败样本再计分，也不能先算各组预言分再平均；二维位置不额外除以维数。V来自固定无噪真值，不随队伍预测变化。节约门槛按未舍入P判断，与定律满分无关，不考虑免费observe余量。正式验证预先固定、与队伍探索历史无关，不统一承诺“两倍距离”，具体初态、范围和机制分组不公开。

正式反馈只含整次总分、各宇宙总分及必要启动/格式错误，不含分项、声明正误、节约资格、归一化误差、隐藏轨迹或内部同分指标。格式错误反馈不等于科学正误反馈。报告可选，不以自然语言报告质量加分。三个声明必须使用标准JSON类型，不额外增加periodic选项；声明不改变加速度签名或切换积分器。


<details>
<summary>完整控制台调用、档案导出与通信错误接口</summary>

用途与边界

rift.Console只传递指挥部允许的实验请求、观测和提交文件，不含真实宇宙实现。
六公开控制台由development_worlds.service在队伍本地运行；九隐藏控制台由指挥部私有服务运行。二者使用相同客户端与操作；候选推演使用公开rift.rollout。
同一个会话由指挥部配置的凭据绑定，客户端不能自行选择或重置宇宙和配额。

启动配置

    import os
    from rift import Console, ConsoleError
    console = Console()
    mission = console.mission()

启动器注入RIFT_CONSOLE_URL和RIFT_SESSION_TOKEN，不应把令牌写入日志或提交说明。
可通过Console(endpoint=..., token=..., timeout=...)显式配置，参数优先于环境变量。
直接项目运行使用指挥部设置的本机HTTP地址，例如http://127.0.0.1:8000/v1/console。
Unix socket保留兼容；两种传输使用相同的POST /v1/console协议，均属于官方通信通道。
HTTP客户端不使用环境代理、不跟随重定向；HTTPS使用系统证书校验。
不提供自行联网的权限，实际可访问范围由指挥部运行环境控制。

操作

    mission = console.mission()
    sky = console.observe(times=[0.0, 0.5, 1.0])
    experiment = console.launch(
        t_launch=0.0, readout=[0.0, 0.5, 1.0],
        source_properties={},
        probes=[{"pos":[1.0,0.0], "vel":[0.0,0.1], "properties":{}}],
    )
    data = console.archive()          # 仅返回已获得的公开记录
    console.archive(os.environ["RIFT_OUTPUT_DIR"])        # 同时在本地导出JSON和CSV
    console.save_draft(law="law.py", claims="claims.json")
    console.submit(law="law.py", claims="claims.json", report="report.md")

示例数值只展示格式，不保证属于某个正式任务的合法范围；以mission目录为准。
观测时刻、配额、合法区域、属性和只读数量仍由服务端按现行规则检查。
NumPy数值和数组可作为普通请求数据，发送时转换为JSON标量与列表。

archive(path)生成archive.json和archive.csv。JSON保留服务端返回的完整公开档案；
CSV每行是一条第一束光或观测记录，包含record_type、record_index、record_json。
JSON列保留数组、时间、初态和属性，不把嵌套数据错误展开或丢弃；可用json.loads解析。
读取档案不重新运行实验、不重新抽噪声；本地导出失败也不会产生新的实验。

save_draft/submit读取本地law.py和claims.json的原始字节，以base64传输；
submit可附report。服务器检查双方格式后配对冻结。坏提交不覆盖旧有效配对。
客户端不预判声明科学正误，不获取隐藏测试轨迹或分项评分。

传输与错误

请求为{protocol_version:"1.0.0",method:方法名,params:参数对象}，
Authorization使用Bearer会话令牌。
成功返回{ok:true,result:公开结果}；失败返回{ok:false,error:{code,path,message}}。
ConsoleError提供code、path、message与as_dict()，不把远端原始响应或堆栈直接展示。
默认timeout=None；整个探索窗口仍由指挥部独立墙钟管控。
如显式设置timeout，它仅是HTTP/Unix通信护栏，不改变物理时间或计分时间。
客户端不自动重试请求。通信超时抛出transport_error，不表示科学预测失败，
也不能据此断言服务端尚未处理请求；复杂重试计费政策仍以正式运行规则为准。

</details>

<details>
<summary>完整law.py输入输出、对象元数据与固定积分接口</summary>

### 由谁负责什么

控制台实验：攻坚组调用 observe/launch，由指挥部私有模拟器推演并返回允许观测。
本地检验：攻坚组提出候选规律后，自行调用公开 rift.rollout(law, case)，与已有观测比较并拟合参数。不查询真实宇宙，不扣实验额度，但消耗算力和探索墙钟。
最终评测：智能体产出 law.py 与 claims.json。指挥部加载 law.py，传入测试初态和时刻，由官方固定驱动积分、生成预测轨迹，再按既有80/15/5规则评分。
law.py 不需要提供 predict()，不自行选择积分器或返回预先计算的测试轨迹。JSON数学表达式不是正式提交格式。

### law.py 的固定入口

    import numpy as np
    LAW_API_VERSION = "1.0.0"
    PARAMS = {}
    BODY_PROPS = {}
    HIDDEN = []

    def acceleration(t, bodies, params):
        return np.zeros((len(bodies), 2), dtype=np.float64)

以上零加速度仅展示接口。可运行的独立接口例子位于 examples/law.py，不代表任何正式宇宙。

LAW_API_VERSION 是接口版本，不是待拟合常数。
PARAMS 是至多5个有限实标量的字面量字典；不得用数组或bool代替标量。
BODY_PROPS 可省略，按可见背景ID补充推断的未知数值属性。官方已知输入优先：重复同值允许，冲突不能覆盖。
HIDDEN 可省略，每项为 {"id":字符串,"pos":[x,y],"vel":[vx,vy],"properties":数值字典}；初态相对于物理t=0。ID无前缀要求，必须与其他对象唯一。
BODY_PROPS/HIDDEN不是额外claims；既有对象属性与几何表达豁免PARAMS五标量上限。
元数据使用Python字面量；普通固定数学常量和数学辅助函数可定义。允许math、NumPy、SciPy数学运算。程序不得读写文件、联网或依赖随机/外部可变状态，不在顶层启动其他工作流。

### 加速度函数拿到什么

t 为积分内部的绝对物理时间，内部子步不一定在0.005采样网格上；不要在函数内舍入它。
bodies 为只读tuple，每个对象有 id、kind、pos、vel、properties。kind为background、hidden或probe。
pos/vel 是该子步的二维候选状态，不是真实未来数据；params及properties也只读。拟合时以传入的params和对象属性为准。
返回与本次bodies顺序等长的(N_body,2)实数加速度。list、tuple和NumPy实数数组均可，统一转换为float64；形状错误、bool、复数和非有限值明确报错。
背景/候选隐藏对象先从t=0推演至投放时刻。为保证被动性，背景回调不含探针，每个探针回调只含背景/候选隐藏对象与它自身；不要假设bodies总长度恒定，用id/kind识别对象。

### 统一推演输入与输出

case只含schema_version、t_launch、readout、background、probes；schema_version为1.0.0。
background和probes均为对象列表，每条恰含id、pos、vel、properties。背景初态在物理0，探针初态在t_launch。
每次实验已知的源侧控制值已合入相应背景对象的properties；没有真实未知源强、真实隐藏对象或未来轨迹。内部测试名称、分组标签不传给law.py。
t_launch与readout分别严格校验0.005网格，readout非负且严格递增，可包含0。只容忍浮点表示尾差，不把非网格请求吸附到附近时刻。

    from rift import load_law, rollout
    law = load_law("law.py")
    prediction = rollout(law, case)

prediction含times、readout、probe_ids、positions、velocities、numerics_version。位置与速度形状为(T,N_probe,2)，按readout及case.probes顺序排列。
从控制台公开记录构造本地case，可用case_from_observation(mission, record)。若存在源侧控制，应从任务说明取得明确的source_property_target_id，或显式传入source_target_id；不由固定接口猜测真实力场。
固定数值协议为Yoshida4、步长0.005、float64及既有0.05幅值软化约定。两边调用同一实现；队伍不能在law.py中换成自己的积分器。

### 提交前检查

以下命令在远端项目根目录运行：

    "$PY" scripts/validate_law.py examples/law.py --claims docs/contracts/claims.example.json
    "$PY" scripts/validate_law.py examples/law.py --case examples/law_case.json --execute

替换为本组文件后，第一条只做静态格式检查，不执行候选或判物理正确性；第二条才运行独立候选worker做本地推演。--timeout是本地检查护栏，不是正式计分时限。
检查错误给出code/path/message；修复后可以再次检查。law.py与claims.json须作为同一配对通过检查；坏提交不覆盖上一份有效配对。
预检通过不保证科学正确。候选计算错误按既有无效预测规则处理；官方case/grid或基础设施错误必须先修复，不算作队伍物理预测错误。
UTF-8 BOM、LF/CRLF均接受；加速度返回值的正常list/tuple/实数dtype差异不会产生格式惩罚。时间输入仍须满足严格网格及其浮点容差。

机器规范见 docs/contracts/law_api.json 和 law_case.schema.json；格式实现以已发布版本的公开检查器为准。

</details>


## 模型检查、缓存与可选环境构建

```bash
"$PY" scripts/serve_model.py --list
"$PY" scripts/serve_model.py --model 27b --plan-only
"$PY" scripts/serve_model.py --model 27b --dry-run
```

list与plan-only不加载权重、不查询GPU，后者甚至不要求模型文件；dry-run只检查配置、tokenizer及权重分片头部，不是完整模型加载。自训完整检查点可加 `--model-path /实际检查点目录`，按真实精度选择配置。路径别名、精度与服务名不能互相替代，不能改名冒充FP8。

CUDA12.8自动服务可运行 `"$PY" scripts/serve_model.py --model 27b --model-python "$MPY" --tp-size 4 --port 8000`。默认TP4、32K上下文、单并发、0.90显存比例只是起步设置；可见卡数需与TP一致。通过基础容量检查不等于KV缓存、FP8、多卡或真实生成均已通过；遇到不兼容不会静默换模型或量化。只有思考、没有最终动作的回复不会当作实验指令执行。

可写HF、vLLM、Triton、FlashInfer和Torch编译缓存由默认加载器放在本局输出目录，预下载内核作为只读资源；这项默认路径安排不等于对任意自定义程序的系统权限隔离保证。无网H200不临时下载包、模型或内核，缺资源应在联网实例预先补齐。

<a id="cuda128"></a>
<details>
<summary>可选：CUDA12.8源码构建、完整工具链及离线验收注意事项</summary>

已有可用环境可跳过。本路线要求先有完整CUDA Toolkit12.8（nvcc、开发头文件和库）、GCC/G++11.3—14、git、curl及glibc≥2.28；bootstrap不包办完整工具链，也不更改GPU驱动。联网CPU构建使用前文 `--cpu-build-for h200`，不需要GPU；不带该选项的现场GPU安装模式另要求目标GPU可见。

环境目标为Python3.12、NumPy2.1.3、SciPy1.15.3、Torch2.11.0+cu128、torchvision0.26.0+cu128、torchaudio2.11.0+cu128、vLLM0.24.0和Transformers5.8.0。H200模式以SM90构建通用内核，专用Hopper内核按上游规则使用SM90a，并保留匹配ptxas。其它GPU架构应另行检查。

`scripts/bootstrap_cuda128.sh`需要同版 `pip_index_install.py`、`preflight_cuda128_build.py`、`prepare_vllm_dependencies.py`、`prepare_torch_wheel.py`，不能只复制一个脚本。先用bootstrap的 `--plan`可检查计划，不安装、不联网、不调用GPU。首次构建建议新环境目录，不覆盖旧结果；失败会留下日志并停止，不应当作安装完成。

指定内部镜像必须确实包含精确版本和cu128变体；缺包时不会自动换CUDA或回退索引。uv管理的Python、工具链、CMake第三方源码和部分cubin资源仍可能需要独立网络来源，不能承诺所有资源只从pip镜像获取，也不能把cu128安装步骤改几个版本字样就当作cu130安装方案。

安装后运行前文的 `prepare_h200_offline_cache.py`准备离线资源，可再以 `--env-root "$ENVROOT" --verify`只检查本地缓存。CPU无法为所有模型形状预热；H200首次运行仍可能用保留的编译器和头文件做本地JIT。必须保留完整解释器、环境、Toolkit、GCC/G++、构建清单和缓存，不能只复制venv或pip列表；Toolkit的stub库不能加入运行时LD_LIBRARY_PATH。

实例和镜像的保存由平台使用者处理；共享盘项目与模型需在运行实例另行挂载，不默认随镜像复制。环境部署后保持相同绝对路径。在H200激活 `"$ENVROOT/activate_h200_offline.sh"` 后，运行 `python3 scripts/verify_h200_cuda128.py --env-root "$ENVROOT"`，再做真实模型生成和完整实验。验证器不联网安装、不加载大模型；构建成功、GPU计算通过、FP8/多卡推理通过和完整测评通过，是不同层级。

</details>

## 六公开开发宇宙的详细资料

`development_worlds/worlds.py`提供无噪声真实模拟，`catalog.py`提供公开操作范围、首光、噪声与验证案例；`console.py`计配额并加噪，`service.py`提供认证会话和配对提交，`runner.py`驱动实验，`evaluation.py`调用候选固定积分评分。这些程序应保留。公开公式可用于无限生成训练数据，但标准会话仍按额度自测；直接代入已知公式的高分只验证流程，不代表未知规律发现能力。

<details>
<summary>六公开宇宙的公式、属性、验证范围及数值边界</summary>

令探针相对原点位置为x=(x,y)，r=||x||，rho=sqrt(r²+0.05²)，u=x/r，原点取零。全部二维、探针被动且自由运动，投放区仅限制初态。下面只公开六开发宇宙，不给出正式九宇宙的规律对应关系。

1、2、4号允许设置 `source_properties.p1` 和每枚探针的 `properties.p2`，范围均为[0.5,2]、默认1。这里 p1 为源强，p2 为惯性。3、5、6号仅有每枚探针 `mass`，范围[0.5,4]、默认1；响应与惯性同比例变化，因此质量在运动方程中抵消。所有宇宙均允许选择探针数、初始位置、初速度和合法采样时刻，不增加材料、物种或专用提交字段。

1、2、4号背景只有固定可见原点 N0。3、5、6号另有8枚可见、无源的自由示踪粒子，按同一力场运动，便于验证 `observe` 的使用；它们不产生力，也不反向影响中心。初态由代码完整公开。

### 公开规律

| 编号 | 等级 | 加速度规律 | 固定参数／可拟合全局标量数 |
|---|---|---|---|
| 1 | 一级 | `a = -G*p1/(4*pi*p2*rho) * u` | G=1；1个 |
| 2 | 一级 | `a = -k*(p1/p2)*x` | k=0.06；1个 |
| 3 | 二级 | `a = -R(theta) diag(k1,k2) R(theta)^T x` | k1=0.04，k2=0.12，theta=pi/6；3个 |
| 4 | 二级 | `a = -G*p1*exp(-rho/lambda)/(2*pi*p2*rho) * u` | G=2，lambda=4；2个 |
| 5 | 三级 | `ax=-a*x-c*x^3-d*x*y^2; ay=-b*y-c*y^3-d*y*x^2` | a=0.06，b=0.10，c=0.002，d=0.012；4个 |
| 6 | 三级 | `a = -R(theta(t)) diag(k1,k2) R(theta(t))^T x`，`theta(t)=pi*t/T+phi` | k1=0.06，k2=0.18，T=16，phi=0.3；4个 |

`R(theta)` 是二维旋转矩阵 `[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]`。以上每个宇宙都能在最多5个全局标量参数内精确表达；控制输入、数学常数和已知数值步长不属于待拟合参数。

- 1号用于熟悉中心作用与源强／惯性控制。
- 2号为线性回复作用，距离越远回复越强，适合作为完整实验、拟合、提交链条的起点。
- 3号需要辨别两个方向不同的回复强度及旋转方向；不能只沿一个方向实验。
- 4号需要覆盖屏蔽长度内外，联合辨别幅值与衰减尺度。G改变强度，lambda改变距离尺度，二者不必同比调整。等级判断还取决于测试覆盖，幅值变大本身不代表更难。
- 5号含 `xy²`、`yx²` 交叉作用；仅沿坐标轴观测会漏掉它。该场是正四次势的负梯度，没有隐藏源、时间变化或额外物种。
- 6号的作用主轴随时间转动；它并非只有径向强度在变化。主轴转过pi后矩阵相同，因此这里的力场周期为16，而不是32。参数选择使旋转系中的有效二次势正定，避免指数失稳。

公开声明真值：全部 `dynamics_order=2`、`hidden_matter=false`；只有6号 `time_dependence=true`。公开开发集的标签分布不保证适用于正式未知任务。

### 公开实验范围与评分

每个标准会话仍是10能量、最多20次 `observe`；每枚新探针1能量，单次 `launch` 最多8枚、每枚最多50时刻。第一束光为2枚标准探针各8时刻，免费；`archive`只返回已有数据。位置和探针速度的读数均加独立高斯噪声；首光／归档不重复抽噪。公开模拟器可直接生成无限数据，训练和调试也可无限重开会话；使用标准会话自测时，计费与限制仍按上述规则。

| 编号 | 连续投放半径 | 初速度上限 | 最大绝对物理时刻 | 单轨迹跨度上限 | 当前验证 |
|---|---|---|---|---|---|
| 1 | [1,8] | 5 | 40 | 30 | 6组×4探针×20时刻 |
| 2 | [0.5,6] | 3 | 32 | 24 | 6组×4探针×20时刻 |
| 3 | [0.5,6] | 3 | 32 | 24 | 6组×4探针×20时刻 |
| 4 | [1,10] | 5 | 40 | 30 | 6组×4探针×20时刻 |
| 5 | [0.5,5] | 3 | 32 | 20 | 6组×4探针×20时刻 |
| 6 | [0.5,6] | 3 | 48 | 24 | 8组×4探针×20时刻 |

空间采样覆盖不同距离、方向和初速度；可控属性取多个组合；时变场覆盖多个投放相位。每案例统一取20个等距正偏移时刻，所有条件在 `catalog.py` 中公开。第一束光与验证初态分开。位置噪声每坐标标准差为 `sigma_x=0.01*sqrt((r_min²+r_max²)/2)`；探针速度噪声每坐标标准差为 `sigma_v=1e-4*initial_speed_max`。速度噪声尺度由公开初速度上限确定，与本次实际投放速度无关；初速度为零时也按同一尺度测量。各坐标、探针、返回时刻及新实验独立抽样，第一束光同样处理。已知投放初态和共同t=0背景初态保持精确，缓存读取不重抽噪声，验证真值不加噪声。位置与速度使用独立随机数流，增加速度噪声不会改变同一种子的原位置噪声序列。

队伍提交候选 `law.py` 和三项 `claims.json`。候选通过现有 `rift.law` 接口检查，再由现有 `rift.prediction` 调用固定积分器生成轨迹，不能用真实模拟器替代候选预测。公开六宇宙与正式任务的区别在于真实规律和案例的可见性，不改变这条评测链。

对全部验证案例、探针和时刻统一汇总：

`MSE = mean(||predicted_position - clean_position||²)`

`V = mean(||clean_position - mean_all_clean_positions||²)`

`e = MSE/max(V,1e-12)`，`P = 80/(1+10e)`。

不是分别计算各组预言分后再平均。三项定律声明每项5分；仅当P≥40时，节约分 `C=5*(1-E/10)`，否则0；每宇宙总分 `P+L+C`。公开自测允许展示细项以便调试，不改变正式隐藏任务仅返回总分与各宇宙总分的约定。


1、4号的径向单位方向在穿越原点时不连续，部分合法初态的步长减半检查不通过。标准验证案例已避开该情况；直接使用模拟器生成其它初态时，应另查数值稳定性，不能把有限案例结果当作全域收敛保证。以上范围和公开声明只适用于开发集，不保证正式未知宇宙沿用相同标签分布。

仅诊断一个自编候选时，可在包根目录运行：

```bash
"$PY" -m development_worlds.evaluation --world 1   --law /本队候选/law.py --claims /本队候选/claims.json   --energy-used 1 --output /新的本地诊断结果.json
```

这里的能量参数只用于单候选开发诊断，合法范围[0,10]；完整运行器和正式评分读取实际会话账本，不接受自报消耗。公开调试可看细项，正式隐藏运行仍只返回规定成绩。旧的内部测试次数、简化规律分数和阶段验收记录不属于当前参赛规则或模型能力保证。

</details>

## 需要继续保留的程序与资料

`requirements.txt`、LICENSE、文件清单、模型配置和 `docs/contracts/`中的JSON，以及全部模型、控制台、积分、评分、安装、检查脚本和示例数据，仍是可运行开发包的一部分。合并自然语言文档不代表可以删除这些内容。赛题PDF继续作为任务通知保留，本文用于具体操作；正式隐藏实现与运行材料始终由指挥部保管。
