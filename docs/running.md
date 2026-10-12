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

## 6. 正确低调用示范继续 SFT

这一步从**合并 SFT 底座上的 G3 LoRA**继续训练。`--model` 仍是 `models/sft`，`--adapter` 是已选定的 G3 检查点；不要用合并后的 `models/tooltune` 再叠加 G3 或效率适配器。

先从已有训练轨迹中完成来源、正确性、工具反馈与全文内容审核，准备两份配好的 JSONL：`low_call` 为正确低调用示范，`correct_random` 为同题固定随机正确示范。每份先放 181 条同题示范，再放完全相同的 85 条工具 replay，保持两份题目顺序一致。随机对照允许与低调用示范相同，不为扩大差距刻意排除。

每行的数据合同如下；`messages` 使用实际模型的原生消息与工具格式，下面只展示字段结构：

```json
{
  "task_id": "train-question-001",
  "kind": "matched",
  "training_split": "train",
  "reviewed_correct": true,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

工具 replay 的 `kind` 为 `replay`，工具观察以原消息保留。入口用底座 tokenizer 重建 assistant-only 标签，保留工具反馈的 attention；如果同时提供 `input_ids`、`attention_mask`、`labels`，必须与重建结果完全一致。布尔标志只表示上游审核结论，不能替代实际内容审核；不要将开发或测试轨迹用于该步骤。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_efficiency_sft.py \
  --model models/sft --adapter outputs/G3/checkpoint-N \
  --data workspace/efficiency/low_call.jsonl \
  --paired-data workspace/efficiency/correct_random.jsonl \
  --arm low_call --output outputs/efficiency-low_call

CUDA_VISIBLE_DEVICES=0 python scripts/train_efficiency_sft.py \
  --model models/sft --adapter outputs/G3/checkpoint-N \
  --data workspace/efficiency/correct_random.jsonl \
  --paired-data workspace/efficiency/low_call.jsonl \
  --arm correct_random --output outputs/efficiency-correct_random
```

共同配置见 [efficiency_sft.json](../configs/efficiency_sft.json)：学习率 `1e-5`、2 epoch、微批 1、累积 16，共 34 次更新，两个 epoch 的尾组均为 10。累积组的 CE 按实际有效目标 token 总数归一化，不把各微批的均值直接等权平均。两次运行都使用新优化器并从同一 G3 起点开始，必须使用独立的新输出目录。

上例顺序执行，每次只让一个设备可见。入口只支持从指定 G3 适配器开始的新训练，不会覆盖已有输出目录，也不自动恢复失败任务。默认保存 `checkpoint-17`、`checkpoint-34`，另保存 `last-adapter`。

在普通提示的 `dev_config.jsonl` 上比较两组，通过预先设定的质量与调用要求后冻结候选，再执行确认和最终测试。导出继续沿用第 5 节的脚本，`--base` 为 `models/sft`，`--adapter` 为新效率输出的 `checkpoint-34` 或 `last-adapter`。结果表在 [README](../README.md#实验结果) 中简述。

该公开入口由实际训练流程整理，配套检查覆盖数据合同与损失归一化等 CPU 逻辑；整理后的入口未重新执行完整 GPU 训练。原始示范内容、权重与运行日志不随代码仓库分发，需要自行准备符合数据合同的已审核输入。
