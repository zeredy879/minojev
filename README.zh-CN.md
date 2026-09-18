<p align="center"><img src="assets/banner.svg" alt="minojev — 决策，而非 token" width="960"></p>

<p align="center">
  <b>简体中文</b> · <a href="README.md">English</a> · <a href="https://zeredy879.github.io/minojev/">在线 Demo</a> · <a href="https://huggingface.co/zeredy879/minojev">模型权重</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-f0b06a">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-6ea8fe">
  <img alt="tests" src="https://img.shields.io/badge/tests-41%20passing-59d3a8">
  <img alt="decode steps" src="https://img.shields.io/badge/decode__steps-0-6ea8fe">
  <img alt="parameters" src="https://img.shields.io/badge/parameters-547k-8d9bb3">
</p>

> **决策不是句子。** 当模型读完你的问题，它心里已经有答案了。minojev 直接
> 读出这个答案，而不是让模型把它"写"出来。

## 30 秒理解核心思路

问聊天模型："这张工单该进哪个队列？"它会写一段话。你的程序再把这段话解析成
一个标签和一个置信度。那段话从来都不是答案本身，而是包装——你为它付出了延迟
（逐 token 生成）、成本（输出 token 更贵）和格式风险（生成的内容可能无法解析）。

minojev 去掉了这层包装：

```
状态 + 问题 + 候选  ──►  一次前向传播  ──►  {access: 0.79, deliverability: 0.14, billing: 0.07}
```

没有句子、没有解析、没有格式错误，只有软件可以直接分支判断的带类型概率分布。
返回的每条记录都写着 `decode_steps: 0`——因为它从未生成任何输出 token。

<img src="assets/explainer.svg" alt="聊天模型逐 token 生成再解析；minojev 一次前向传播直接读出分布" width="100%">

## minojev 的与众不同

| | 聊天模型 | minojev |
|---|---|---|
| 输出 | 一段待解析的文字 | 声明候选集上的带类型分布 |
| 答案来源 | 逐 token 采样 | 从隐藏状态读出，零输出 token |
| 成本主要来自 | 输出 token | 一次前向传播 |
| 格式错误 | 可能发生 | 不可能——输出空间提前声明 |
| 置信度 | 自我报告，常常过度自信 | dev 集校准，ECE 可测可查 |
| 同状态多问题 | 每个问题生成一次 | 一次前向，共享 KV 前缀 |
| 运行环境 | GPU 集群或 API | 笔记本 CPU；可选 Hugging Face 模型 |

相比其它决策模型实验，这个项目还有这些特点：

- **完全离线复现。** 内置 547k 参数模型在 CPU 上几分钟从零训练完成，不需要
  下载、API key 或 GPU。
- **每个结论都有产物。** 数据集、teacher 目标、逐题预测、指标和回放数据
  全部提交在仓库里。
- **校准是一等公民。** `--calibrate gold` 是默认行为——无法信任的概率比标签
  更糟糕。
- **双引擎。** 可训练决策头提供完全控制，原生 logits 路径支持预训练
  Hugging Face 模型。
- **双语 + 可视化。** 英文 / 简体中文文档、动态迷宫智能体和交互式决策控制台。

## 在线 Demo

| [迷宫智能体回放](https://zeredy879.github.io/minojev/maze.html) | [并行决策控制台](https://zeredy879.github.io/minojev/console.html) |
|---|---|
| 每一步都展示移动分布、四个并行安全布尔判断，以及代码约束后的最终移动。**6 局解出 5 局。** | 单个状态下多个运行时问题一起打分，并叠加 teacher 分布对照。 |

![迷宫回放预览](https://zeredy879.github.io/minojev/data/maze-preview.svg)

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
# http://127.0.0.1:8080
```

## 安装

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[test]"
# 可选：使用 Hugging Face 模型的原生 logits 引擎
uv pip install -e ".[hf]"
```

## 快速开始

```bash
# 1. 生成合成决策数据（属性查找、比较、支持度）
minojev synth --out data/train.jsonl --count 512 --seed 17 --split train
minojev synth --out data/dev.jsonl   --count 128 --seed 17 --split dev

# 2. 一条命令完成训练 + 校准
minojev train --train data/train.jsonl --dev data/dev.jsonl \
  --output-dir runs/synth --steps 800 --head-steps 30 --eval-every 100 \
  --calibrate gold --device cpu

# 3. 为任意请求打分
minojev score --checkpoint runs/synth/checkpoint --input examples/decisions.jsonl \
  --output results/example-scores.jsonl --mode reuse
```

训练迷宫智能体并导出回放数据：

```bash
minojev maze-data --out data/maze-train.jsonl --count 1024 --seed 17 --split train
minojev train --train data/maze-train.jsonl --dev data/maze-dev.jsonl \
  --output-dir runs/maze --steps 1500 --head-steps 40 --eval-every 100 --device cpu
minojev maze-rollout --checkpoint runs/maze/checkpoint \
  --output web/data/maze.json --count 6 --seed 23
```

## 概率校准：让置信度有意义

训练只最小化分布损失，这本身并不能保证置信度可信。`minojev calibrate` 在
dev 集上为每种原语拟合一个温度，写入检查点并在推理时自动应用。温度始终为正，
所以它永远不会改变模型的预测结果，只会改变模型的"确信程度"。

| 内置模型 | 准确率 | 校准前 ECE | 校准后 ECE | 平均置信度 |
|---|---:|---:|---:|---:|
| 属性决策 | 66.5% | 0.093 | **0.074** | 0.72 |
| 迷宫决策 | 83.8% | 0.090 | **0.016** | 0.84 |

```bash
minojev calibrate --checkpoint runs/synth/checkpoint --input data/dev.jsonl \
  --output runs/synth-calibrated --target gold
minojev evaluate  --checkpoint runs/synth-calibrated --input data/test.jsonl
```

## 结果

两个内置模型都是从零训练的 547k 参数 Transformer，全程在 CPU 上完成。
"Teacher top-set" 指预测命中 teacher 的最优候选集合——在网格移动中两个方向
经常并列，这一指标更公平。运行 `scripts/build_results.sh` 可复现全部数字。

| 任务 | 问题数 | 准确率 | Teacher 最优集 | 分布误差 | 解码步数 |
|---|---:|---:|---:|---:|---:|
| 属性决策 | 627 | 66.5% | 66.5% | 0.29 | 0 |
| 迷宫决策 | 1,536 | 83.8% | 89.5% | 0.13 | 0 |

迷宫分原语 top-set 准确率：**布尔安全判断 87.8%**、**距离评分 97.7%**、
**移动选择 87.9%**。

### 延迟与吞吐（笔记本 CPU，547k 参数）

| 任务 | 模式 | p50 | p95 | 决策数/秒 |
|---|---|---:|---:|---:|
| 属性 | fresh | 13.1 ms | 16.3 ms | 198 |
| 属性 | reuse | **11.8 ms** | **14.7 ms** | **233** |
| 迷宫 | fresh | 23.6 ms | 27.0 ms | 198 |
| 迷宫 | reuse | **17.1 ms** | **18.0 ms** | **366** |

一个决策对应一个问题；一个请求可以包含多个问题。使用 `minojev bench` 复现。

原始指标和逐题预测提交在 [`results/`](results) 下；回放数据在
[`web/data/`](web/data) 下。

## 模型权重

训练并校准后的检查点已发布到 Hugging Face：

```python
from huggingface_hub import snapshot_download
from minojev import DecisionModel

path = snapshot_download("zeredy879/minojev", allow_patterns=["maze/*"])
model = DecisionModel.load(f"{path}/maze", device="cpu")
```

## 请求格式

```json
{
  "id": "route-1",
  "state": "Customer cannot access an account after a password reset.",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which queue should handle this request?",
      "options": {"access": "Account access support.", "billing": "Billing support."}
    },
    "urgent": {
      "type": "boolean",
      "instructions": "Is the customer fully locked out?",
      "criteria": {"true": "No usable login path", "false": "Login path still works"}
    },
    "confidence": {
      "type": "score",
      "instructions": "How confident is this judgment?",
      "levels": ["very low", "low", "medium", "high", "very high"]
    }
  }
}
```

`state` 可以是文本或 JSON。扁平的单问题行
（`{"id", "state", "question", "options"}`）会按 choice 问题解析。问题 id
只用于标识返回结果，永远不会进入模型输入。可选的 `gold` 与 `teacher`
字段用于评估和有监督训练。

## Python API

```python
from minojev import DecisionModel, Request, make_choice_question, ScoreOptions

model = DecisionModel.load("runs/synth-calibrated", device="cpu")
request = Request(
    id="r1",
    state={"color": "blue", "shape": "round"},
    questions=[make_choice_question("q", "Which value belongs to 'shape'?",
                                    {"round": "round", "blue": "blue", "tiny": "tiny"})],
)
record = model.score([request], ScoreOptions(mode="reuse"))[0]
print(record["candidate_ids"], record["probabilities"], record["decode_steps"])
```

## 原生 logits 引擎（预训练模型）

对于预训练的 Hugging Face 因果语言模型，`minojev.logits` 直接读取固定答案槽
位置的 next-token logits 为声明的候选打分。当候选数超过 16 个字母槽时，
两阶段路径会逐个候选独立打分（yes/no log-odds 后归一化）：

```python
from minojev import load_hf_backbone
from minojev.data import read_requests
from minojev.logits import score_logits, score_logits_two_stage

backbone = load_hf_backbone("Qwen/Qwen2.5-0.5B-Instruct")
requests = read_requests("examples/decisions.jsonl")
records = score_logits(backbone, backbone.tokenizer, requests)
wide = score_logits_two_stage(backbone, backbone.tokenizer, requests)
```

## 仓库结构

```
src/minojev/
  types.py       请求校验与原语定义
  encoding.py    候选路径、状态/后缀切分
  backbone.py    TinyLM（RoPE + KV 缓存）与 HF 适配器
  heads.py       共享标量 + 集合注意力、各原语读出
  calibrate.py   每种原语的温度缩放校准
  model.py       推理模式与检查点
  train.py       预热、训练目标、dev 选点
  synth.py       属性决策合成任务
  maze.py        网格世界状态、teacher 与智能体回放
  logits.py      原生 logits 读出（单次与两阶段）
  bench.py       延迟与吞吐测量
  metrics.py     准确率、CE、ECE、按任务族拆分
  cli.py         synth / train / calibrate / bench / score / evaluate / demo / maze
tests/           41 个测试：契约、等价性、校准、训练、迷宫、CLI
web/             主页、决策控制台、迷宫回放
assets/          banner、logo、原理图
```

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

测试覆盖校验边界、路径编码、置换等变、布尔与评分读出、fresh 与 reuse
等价性、批量独立性、检查点往返、校准不变性与拟合、ECE、端到端训练收敛、
迷宫任务与回放机制、基准测量、CLI，以及两条原生 logits 路径。全部离线运行，
几分钟内完成。

## 许可证

MIT — 见 [LICENSE](LICENSE)。
