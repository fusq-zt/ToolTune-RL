# 运行说明

## 环境与入口

基础示例与逻辑测试只需 Python 3.11+。模型训练使用 Linux、Python 3.11/3.12、CUDA 环境，以及 Bubblewrap 提供的隔离 Python 执行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,dev]"
python scripts/install_trl_adapter.py
python scripts/install_trl_adapter.py --check
```

安装系统包 `bubblewrap`，或用 `TOOLTUNE_BWRAP` 指向已有可执行文件。系统需要允许非特权用户命名空间。执行工具前会检查文件与网络隔离；不具备该能力时停止运行。

训练环境固定 TRL 0.27.2 / vLLM 0.12.0 / Transformers 4.57.6 / PEFT 0.18.1。适配脚本让 TRL 的自定义 rollout 接口接收工具反馈掩码，不替换其损失函数。公共入口由实际训练实现整理；整理后完成本地逻辑检查，完整训练需在上述环境中执行。

## 1. 准备数据

基座模型的完整 ID 为 `Qwen/Qwen3-4B-Instruct-2507`，正文简写为 Qwen3-4B。模型及数据版本见 [sources.json](../configs/sources.json)。

```bash
python scripts/data/download_sources.py --root workspace --model-output models/base
python scripts/data/prepare_splits.py --root workspace
python scripts/data/import_toolstar.py --root workspace --model models/base
```

准备完成后，`workspace/data/prepared/` 包含：

| 文件 | 用途 |
| --- | --- |
| `sft_pool.jsonl` | 示范对齐与轨迹生成题池 |
| `rl.jsonl` | 在线训练问题 |
| `dev_config.jsonl` | 开发配置 |
| `dev_checkpoint.jsonl` | 检查点选择 |
| `test.jsonl` | 模型与配置固定后的最终评测 |

任务记录包含 `task_id`、`family`、`question`、`answer`、`verifier`、`documents` 和 `cluster_id`。[示例](../examples/task.json) 展示了字段。提示词仅接收问题；答案留给验证器，文档通过检索工具返回。

## 2. 构造 SFT 轨迹

公开示范导入后，对剩余问题进行有界生成。下面按两个固定分片依次执行，两轮合计每题最多四个候选：

```bash
for shard in 0 1; do
  python scripts/data/generate_sft.py --root workspace --model models/base --shard "$shard" --round 0
  python scripts/data/generate_sft.py --root workspace --model models/base --shard "$shard" --round 1
done

python scripts/data/assemble_sft.py \
  --inputs workspace/data/sft/public.jsonl \
    workspace/data/sft/generated-0.jsonl workspace/data/sft/generated-1.jsonl \
    workspace/data/sft/generated-0-r1.jsonl workspace/data/sft/generated-1-r1.jsonl \
  --output workspace/data/sft/train.jsonl
```

导入器校验 Tool-Star 格式、SFT 题池归属、最终答案与可重放的工具反馈。生成器保留首个通过验证的候选，`assemble_sft.py` 校验唯一 ID 和 token / 标签对齐；运行得到的样本数量以实际通过验证的轨迹为准。

## 3. LoRA SFT 与检查点选择

```bash
python scripts/train_sft.py --model models/base \
  --data workspace/data/sft/train.jsonl --output outputs/sft
```

将各个保存的适配器导出后，在 `dev_checkpoint.jsonl` 上评测。示例路径中的 `checkpoint-98` 是本次实验选定的 SFT 检查点；新的训练应选择自己的开发集检查点。

```bash
python scripts/export_model.py --base models/base \
  --adapter outputs/sft/checkpoint-98 --output models/sft
python scripts/evaluate.py --model models/sft \
  --data workspace/data/prepared/dev_checkpoint.jsonl --output outputs/sft-dev
```

检查点选择先比较开发集宏平均通过率；差异在 0.1 个百分点以内时比较调用数，再比较训练步数。最终测试集不用于选择检查点。

## 4. GRPO 与分支采样

先用选定的固定 SFT 模型，在 RL 训练题的固定子集上初始化熵历史，不使用答案奖励选择统计样本：

```bash
python scripts/calibrate_entropy.py --model models/sft \
  --data workspace/data/prepared/rl.jsonl --output workspace/entropy.json

python scripts/train_rl.py --model models/sft --method G3 \
  --data workspace/data/prepared/rl.jsonl \
  --dev workspace/data/prepared/dev_checkpoint.jsonl \
  --calibration workspace/entropy.json --output outputs/G3
```

`--method G0/G1/G2/G3/G4` 选择消融方案，含义见 [方法说明](method.md)。共同训练参数见 [experiment.json](../configs/experiment.json)。每个方案使用独立输出目录；恢复时显式传入 `--resume outputs/G3/checkpoint-N`，同时恢复优化器、调度器和采样历史。

## 5. 导出与评测

RL 会在开发集上选出检查点，将路径写入 `selected-checkpoint.json`。下面将 `checkpoint-N` 替换为该路径：

```bash
python scripts/export_model.py --base models/sft \
  --adapter outputs/G3/checkpoint-N --output models/tooltune
python scripts/evaluate.py --model models/tooltune \
  --data workspace/data/prepared/test.jsonl --output outputs/test
```

评测统一使用贪心单轨迹，输出逐题轨迹和简要指标；宏平均通过率先按四类任务分别计算，再取平均。
