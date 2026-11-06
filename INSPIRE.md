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
