# 启智平台的已验证环境与路径

本文件供项目维护和指挥部运行参考，不属于参赛者开发包。凭证只由本地账号管理保存，不写入本仓库。

## 招生营专项空间／26-the-other-shore

用户指定的既有实例为 `yuanmeilu-test5`。开始运行前应重新查询实例状态、实际GPU数量、显存和现有进程；不据历史配置假定有四张GPU。实例、镜像的保存和复制由用户负责。

长期主项目路径仍为：

- 主项目：`/inspire/hdd/project/26-the-other-shore/fengkairui-25026/DiscoverPhysics`
- 测试资料：`/inspire/hdd/project/26-the-other-shore/fengkairui-25026/test`
- 公共权重：`/inspire/hdd/project/26-the-other-shore/public/models`

本实例另有独立工作目录 `/inspire/hdd/project/26-the-other-shore/yuanmeilu-253114050257`，用于本次实测及其日志，不替代上述主项目路径。

## 2026-11-06实测使用的现成依赖

- Python：`/opt/shore-test5-forge2/bin/python`，版本3.12.16。
- PyTorch：`2.11.0+cu130`，其实际CUDA构建为13.0。
- vLLM：0.24.0；Transformers：5.8.0；Triton：3.6.0。
- NumPy：2.1.3；SciPy：1.15.3。
- 权重：`/inspire/hdd/project/26-the-other-shore/public/models/Qwen3.5-122B-A10B-FP8`。

当日已实际加载该FP8权重并完成文本生成。驱动显示的CUDA最高支持版本与 `torch.version.cuda` 分开核实，不能据 `nvidia-smi` 推断PyTorch构建版本。

该环境没有可直接发现的 `nvcc`。本次服务使用 `--gdn-prefill-backend triton`、`VLLM_USE_FLASHINFER_SAMPLER=0`，避免默认FlashInfer路径在预热时要求额外CUDA编译器。其余服务参数、端口及结果属于单次运行记录，保存在该次测试目录，不将其视为全局默认。

基础框架的自动加载器仍有CUDA12.8检查。本次CUDA13.0实测通过基础框架已有的本机推理服务连接接口接入，未重装或降级上述环境，也未因此修改正式模型白名单和评分规则。

## 个人策略与SFT早期准备（2026-11-07，历史）

用户指定实例 `hwz-test1`，空间为招生营专项空间；当日实查两张H200。个人工作根为 `/inspire/hdd/project/26-the-other-shore/263140067`，其中已有starter-v15副本。新增数值模拟、数据与训练准备位于个人根下独立目录；训练后的模型成果必须放在个人根的 `models/` 下。

`/inspire/hdd/project/26-the-other-shore/public` 为严格只读共享范围，模型可从其 `models/` 读取，但所有日志、缓存、临时文件及更新权重都应写到个人根。可读取的本轮模型为 `Qwen3.5-122B-A10B-FP8` 教师和 `Qwen3.6-35B-A3B` BF16学生；不能因目录别名推断架构，实际配置均需核验。122B用于推理生成数据，不将其FP8目录当作可直接训练的35B底座。

实例已有 `/opt/shore-test5-forge2/bin/python`，当日复核Torch2.11.0+cu130、Transformers5.8.0和vLLM0.24.0。模型服务可采用TP2；未来两卡LoRA训练使用DDP，不能把服务的TP2参数当作训练设置。当前任务只准备数据、配置和入口，不执行训练或保存镜像；正式测评继续使用此前已成功的独立指挥部流程。

## 个人四卡训练与测评资产（2026-11-07后续授权）

用户随后将同一既有 `hwz-test1` 改为四卡H200，并授权实际训练和完整模拟。开始新的运行仍须实查设备，不根据此记录假定资源始终不变。本轮已完成122B BF16 LoRA，使用单进程四卡连续层划分；这与早期两卡35B/DDP准备方案不同，服务TP参数也不能直接当成训练数据并行参数。

- 只读BF16基座：`/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-122B-A10B`。
- 更新LoRA：`/inspire/hdd/project/26-the-other-shore/263140067/models/qwen122b-observation-sft-v1/adapter`。
- 完整合并BF16：同级 `merged-bf16`，实际39个分片。保留原始基座、LoRA和合并结果，不覆盖公共模型。
- 训练解释器：个人根 `.venv-sft35b/bin/python`，复用现成CUDA13.0/Torch2.11环境；目录名保留早期命名，不代表训练对象是35B。
- 本轮测评账号可访问的解释器：个人根 `.evaluation-runtime/bin/python`。它复制所需Python核心及头文件，并只读引用既有可访问的依赖；不是可脱离原实例依赖的独立安装包。其它实例应以实际已安装环境为准。
- 本次获准验收的原版包位于个人根 `test/shore-organizer-runtime-strategy-v7-prompt`，仅此次改变测评放置位置，不自动迁移指挥部长期主项目。原始ZIP和程序保持，用户的公共目录仍严格只读。

## 2026-11-06九界实测与最终选集

本节保留更新前的真实实测记录。当前正式配置已改为strategy-v5：Adam、Lilith、Sachiel、Matriel、Leliel、Arael六界，显式权重见AGENTS.md第7.109节；每界10点/300秒，未测三界仍保留。此后397B测试已由用户取消，没有新的宇宙模型成绩，不将本节旧分数改称新规则成绩。

在同一既有实例、同一模型和上述环境中，已运行候选保存修复后的基础框架，显式选择完整九宇宙；未重新安装、训练模型或复用历史候选。每界最多3次决策、600秒探索，实际七界有效交卷，两界未交卷；评分公式和所有科学真值保持。

本次远端工作目录为 `/inspire/hdd/project/26-the-other-shore/yuanmeilu-253114050257/shore-qwen122b-nine-20261106-59694cbc`。整体运行约19分49秒，含模型启动和正式积分评分；其中模型启动与预检约1分43秒，各界探索合计约11分27秒。不能把探索耗时当作完整运行耗时。

各档本次总分最高者为Tabris、Ramiel、Leliel，最终默认strategy-v4据此设置；最终交付包仅同步选集和接口契约，未重新调用模型。全部结果与候选在仓外私有审计目录保留。测试结束已停止本次自建推理服务，确认GPU占用回到0MiB，实例保持运行。调用次数与成功交卷以服务端记录复核，不把本地日志抢先终止时的旧状态当作现实超时。
