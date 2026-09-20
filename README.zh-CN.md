<p align="center"><img src="assets/banner.svg" alt="minojev — 决策，而非 token" width="960"></p>

<p align="center">
  <b>简体中文</b> · <a href="README.md">English</a> · <a href="https://zeredy879.github.io/minojev/zh.html">在线 Demo</a> · <a href="https://zeredy879.github.io/minojev/zh-benchmark.html">基准对比</a> · <a href="https://huggingface.co/zeredy879/minojev">模型</a> · <a href="https://huggingface.co/datasets/zeredy879/minojev-data">数据</a>
</p>

<p align="center">
  <a href="https://huggingface.co/zeredy879/minojev"><img alt="Hugging Face models" src="https://img.shields.io/badge/Hugging%20Face-models-f0b06a"></a>
  <a href="https://huggingface.co/datasets/zeredy879/minojev-data"><img alt="Hugging Face datasets" src="https://img.shields.io/badge/Hugging%20Face-datasets-59d3a8"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT-f0b06a">
  <img alt="decode steps" src="https://img.shields.io/badge/decode__steps-0-6ea8fe">
  <img alt="head params" src="https://img.shields.io/badge/decision__head-0.8M-8d9bb3">
</p>

> **一句话版本：** 聊天模型靠"写"文字回答，minojev 靠"读"概率分布回答——而
> **头训练（head training）**能在笔记本上几分钟把一个冻结的语言模型变成这样的决策层，
> 零输出 token，也不需要微调骨干。

## 什么是头训练？

大部分 "Jev 式"项目要么调用托管 API，要么微调整个模型。minojev 的核心比两者都小：

1. **冻结骨干。** 发布版本中 Qwen3-1.7B 全程不更新。
2. **一次性缓存候选特征。** 每条候选路径（状态 + 问题 + 选项）只前向一次，末位隐藏状态被存下来。
3. **只训练决策头。** 一个共享打分器加集合注意力——**约 0.8M 参数**——在缓存向量上用分布损失训练。
4. **在 dev 上校准。** 每种原语拟合一个温度，使用留出结果。

整个训练在 **Apple Silicon 笔记本上约 37 分钟、峰值 4GB 内存**。骨干没有被改动，因此没有
灾难性遗忘、不需要 GPU，而且每一步都可见（见 [内存安全的训练](#内存安全的训练)）。

## 基准：决策读出 vs 逐 token 生成

同一底座（Qwen3-1.7B）、同一批问题、同样的提示。基线必须逐 token **生成**答案；minojev
直接从隐藏状态读出带类型的分布。测试集：120 条平衡决策，覆盖 banking77、CLINC150、
Amazon Polarity、GSM8K 验证。生成式基线使用 chat 模板、关闭 thinking，并要求只答一个字母。

| 指标 | minojev（头训练） | 零训练 logits 读出 | 生成式基线 |
|---|---:|---:|---:|
| 准确率 | **95.8%** | 88.3% | 80.0% |
| ECE | **0.024** | 0.097 | 无法给出 |
| 输出 token / 决策 | **0** | **0** | 3.48 |
| 首 token 延迟（p50） | **0 ms** | **0 ms** | ~100 ms |
| 延迟 p95 | **~1.1 s** | ~1.1 s | ~5.3 s |
| 格式 / 解析失败 | **0** | 0 | 0 |

完整测试集（200 条平衡请求，校准版）：**准确率 97.5%、ECE 0.014**，平均置信度 0.963；
置信度 ≥ 0.9 时覆盖 **89.5% 的请求、准确率 98.9%**——可以直接用于"接受/升级"策略。

**诚实边界：** 在一个刻意的分布外工作负载（从未训练过的客服领域）上，头训练模型只有
31.7%，而零样本生成式基线是 40.0%。头训练是"专精"，不能替代数据广度。因此流水线会把
OOD 来源单独隔离并单独报告。

<img src="assets/explainer.svg" alt="聊天模型逐 token 生成再解析；minojev 一次前向直接读出分布" width="100%">

## 在线试玩

- **[基准对比页](https://zeredy879.github.io/minojev/zh-benchmark.html)**——准确率、token 经济、响应速度并排展示；
- **[多领域体验场](https://zeredy879.github.io/minojev/zh-playground.html)**——六个业务场景的带类型分布；
- **[迷宫智能体](https://zeredy879.github.io/minojev/zh-maze.html)**——逐步回放学到的策略与概率；
- **[决策控制台](https://zeredy879.github.io/minojev/zh-console.html)**——一个状态、多个运行时问题。

## 快速开始

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e ".[test,monitor]"          # 需要 Hugging Face 骨干再加 .[hf]
```

构建通用数据集并运行头训练（带实时监控）：

```bash
# 1. 把宽松许可的公开数据集转换成决策请求（train/dev/test/OOD）
minojev build-data --per-source 2500 --per-ood 800

# 2. 头训练：冻结骨干、缓存特征、只训决策头
minojev posttrain --backbone Qwen/Qwen3-1.7B --mode head \
  --train data/general-train.jsonl --dev data/general-dev.jsonl \
  --output-dir runs/general-head --steps 3000 \
  --inference-dtype bfloat16 --max-memory-gb 12

# 3. 用 dev 结果做温度校准
minojev calibrate --checkpoint runs/general-head/checkpoint \
  --input data/general-dev.jsonl --output runs/general-cal --target gold --from-records

# 4. 与同底座的逐 token 生成对比
minojev compare --checkpoint runs/general-cal \
  --input data/general-test.jsonl --limit 120 --chat-template
```

另开一个终端实时看训练面板：

```bash
minojev watch --run runs/general-head --port 8010   # http://127.0.0.1:8010/
```

## 内存安全的训练

监控面板不是装饰：每个训练阶段都会把 loss、梯度范数、tokens/s、ETA、MPS 显存、进程 RSS、
CPU 负载和系统内存压力写入 `status.json` 与 `metrics.jsonl`。硬预算（`--max-memory-gb`）
会在机器濒临危险前终止该步；校准前向分块执行；每个阶段先释放上一个模型再加载下一个。
1.7B 头训练峰值 **4.0GB**，预算 12GB。

## 不用 loss 的评测

loss 对决策模型是个糟糕的进度指标——teacher 分布有熵下界，异构批次又让它噪声很大。流水线报告：

| 维度 | 指标 |
|---|---|
| 决策质量 | 准确率、teacher 最优集、按来源与候选数的拆分 |
| 概率质量 | ECE、Brier、gold NLL、选择性准确率（覆盖率 @ 置信度） |
| Token 经济 | 每决策输入/输出 token、解码步数（0） |
| 响应速度 | 首 token 延迟、每请求 p50/p95、每秒决策数、解析失败率 |

`minojev compare` 会从提交在仓库里的产物生成 JSON + Markdown 报告，
`scripts/build_benchmark_bundle.py` 为在线基准页提供数据。

## Hugging Face 模型与数据集

- 检查点：[`zeredy879/minojev`](https://huggingface.co/zeredy879/minojev)——
  `general/`（Qwen3-1.7B + 决策头），以及从零训练的微型 `synth/`、`maze/`；
- 数据：[`zeredy879/minojev-data`](https://huggingface.co/datasets/zeredy879/minojev-data)——
  `general/{train,dev,test,ood}.jsonl`，每行都带来源与许可证。

```python
from huggingface_hub import snapshot_download
from minojev import DecisionModel

path = snapshot_download("zeredy879/minojev", allow_patterns=["general/*"])
model = DecisionModel.load(f"{path}/general", device="cpu")
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

`choice` 支持 2–255 个候选，`score` 支持 2–10 个有序等级。扁平的单问题行会按 choice 解析。
问题 id 永远不会进入模型输入。

## 本地服务

```bash
minojev serve --checkpoint runs/general-cal --port 8000
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"id":"r1","state":"Where is my card?","question":"Which category?","options":{"card_arrival":"Card arrival","atm":"ATM"}}'
```

每条记录都会报告 `decode_steps: 0` 并携带完整候选分布。

## 原生 logits 引擎（零训练）

对预训练模型，`minojev.logits` 直接读取字母槽位置的选项 logits——基准里的零训练路线：

```python
from minojev import load_hf_backbone
from minojev.logits import score_logits, score_logits_two_stage

backbone = load_hf_backbone("Qwen/Qwen3-1.7B")
records = score_logits(backbone, backbone.tokenizer, requests)
wide = score_logits_two_stage(backbone, backbone.tokenizer, requests)  # 超过 16 个候选
```

## 仓库结构

```
src/minojev/
  types.py        请求校验与原语
  encoding.py     候选路径、状态/后缀切分
  backbone.py     TinyLM（RoPE + KV 缓存）与 HF 适配器
  heads.py        共享打分器 + 集合注意力、各原语读出
  posttrain.py    头训练 / LoRA（缓存特征）
  monitor.py      实时状态、内存预算、CPU/内存压力
  watch.py        训练面板服务
  calibrate.py    温度校准（前向式与记录式）
  generative.py   生成式基线（含 TTFT 测量）
  compare.py      决策引擎 vs 生成基准
  domains.py      六领域合成决策
  dataset_build.py 公开数据集转换与来源隔离
  logits.py       原生 logits 读出（单次与两阶段）
  serve.py        本地 HTTP 决策 API
  metrics.py      准确率、ECE、Brier、选择性准确率、分来源
  maze.py         网格世界任务与智能体回放
  synth.py        属性决策任务
tests/            离线测试套件（单元 + 端到端 + 面板）
web/              主页、基准、体验场、迷宫、控制台（中英双语）
```

## 开发说明

本项目在开发过程中使用了 AI 辅助。正确性由离线测试套件、随仓库提交的逐题预测与指标，
以及 [`scripts/`](scripts) 下的可复现脚本共同保证。训练数据只用宽松许可（MIT、Apache-2.0、
CC-BY）；NC 许可的数据仅用于评测。

## 许可证

MIT — 见 [LICENSE](LICENSE)。`general/` 检查点是 Qwen3-1.7B（Apache-2.0）的衍生作品，
继承其许可证。
