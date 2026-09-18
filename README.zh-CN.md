<p align="center"><img src="assets/banner.svg" alt="minojev — 决策，而非 token" width="960"></p>

<p align="center">
  <b>简体中文</b> · <a href="README.md">English</a>
</p>

<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-f0b06a">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-6ea8fe">
  <img alt="tests" src="https://img.shields.io/badge/tests-41%20passing-59d3a8">
  <img alt="decode steps" src="https://img.shields.io/badge/decode__steps-0-6ea8fe">
  <img alt="parameters" src="https://img.shields.io/badge/parameters-547k-8d9bb3">
</p>

**minojev 在一次前向传播中，把运行时定义的判断问题变成带类型的概率分布。**
状态与问题输入，完整分布输出，全程不生成任何输出 token。数据生成、训练、
校准、推理与评估的完整流水线都可以在笔记本 CPU 上离线运行。

## 为什么是"决策，而非 token"？

聊天模型回答一个路由问题时，需要生成一句话，软件再把它解析回一个 `if`
语句：既消耗输出 token、增加延迟，也可能产生幻觉。minojev 直接把决策交给
软件——针对声明的候选集，从隐藏状态中读出经过校准的概率分布，永不采样。

|  |  |
|---|---|
| **三种原语** | `choice`（2–255 个候选）、`boolean`（单个命题）、`score`（2–10 个有序等级 + 期望值） |
| **零解码** | 不产生任何输出 token，每条记录都报告 `decode_steps: 0` |
| **并行判断** | 多个独立问题在一次前向中完成，问题之间没有交叉注意力 |
| **状态复用** | 每个不同状态只做一次 KV 缓存预填充，再按问题与候选分支 |
| **概率校准** | 在 dev 集上拟合温度，让置信度与准确率对齐，并报告 ECE |
| **可审计** | teacher 目标由构造精确给出，按 dev 选择检查点，带置换等变测试 |

## 在线 Demo

| [迷宫智能体回放](https://zeredy879.github.io/minojev/maze.html) | [并行决策控制台](https://zeredy879.github.io/minojev/console.html) |
|---|---|
| 每一步都展示移动分布、四个并行安全布尔判断，以及代码约束后的最终移动。**6 局解出 5 局。** | 单个状态下多个运行时问题一起打分，并叠加 teacher 分布进行对照。 |

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

## 概率校准

训练只最小化分布损失，这本身并不能保证置信度有意义。`minojev calibrate`
在 dev 集上为每种原语拟合一个温度（最小化负对数似然），写入检查点并在推理时
自动应用。温度缩放不会改变任何预测结果。

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
"Teacher top-set" 指预测命中 teacher 的最优候选集合（在网格移动中两个方向
经常并列，这一指标更公平）。运行 `scripts/build_results.sh` 可复现全部数字。

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

一个决策对应一个问题；一个请求可以包含多个问题。使用 `minojev bench`
复现。

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
位置的 next-token logits，为声明的候选打分。当候选数超过 16 个字母槽时，
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
assets/          banner 与 logo
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
