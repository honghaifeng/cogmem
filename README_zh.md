# CogMem: 认知记忆网络

[English](README.md) | **中文**

**一个面向 LLM 智能体的混合记忆系统，融合符号检索、神经检索与图谱扩散激活。**

CogMem 维护三种并行记忆结构 —— FTS5 索引的扁平文本片段、带扩散激活的实体-关系知识网络、以及稠密向量嵌入 —— 通过加权融合实现稳健的长期对话记忆。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    CogMem 架构                              │
├─────────────────────────────────────────────────────────┤
│                                                            │
│  记忆存储                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ 扁平文本片段  │  │ 实体-关系     │  │  向量嵌入     │    │
│  │ (FTS5 索引)   │  │ 网络         │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐    │
│  │ 路径 1:       │  │ 路径 3:       │  │ 路径 2:       │    │
│  │ FTS5 关键词   │  │ 扩散激活      │  │ 向量相似度    │    │
│  │ 检索         │  │              │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                  ┌────────▼───────┐                       │
│                  │ 加权融合       │                       │
│                  │ & 去重         │                       │
│                  └────────────────┘                       │
│                                                            │
└─────────────────────────────────────────────────────────┘
```

**三条检索路径：**

| 路径 | 方法 | 优势 |
|------|------|------|
| 符号检索 | FTS5 全文检索 + 中文 LIKE 匹配 | 精确关键词/人名/日期匹配 |
| 神经检索 | 稠密向量余弦相似度（SentenceTransformer） | 语义改写匹配 |
| 结构化检索 | 实体-关系图 + 一跳扩散激活 | 多跳关联回忆 |

## 快速开始

### 安装

```bash
pip install -e .
```

### 基本用法

```python
from cogmem import CogMem

# 使用默认配置初始化（从环境变量读取 DeepSeek API）
mem = CogMem(db_path="my_memory.db")

# 添加对话内容
mem.add("I had lunch with Alice at the Italian restaurant on 5th Avenue. She mentioned she's moving to Tokyo next month.",
        user_id="user1")

# 搜索记忆
results = mem.search("Where is Alice moving?", user_id="user1")
for r in results:
    print(f"[{r['type']}] {r['memory']} (score: {r['score']:.3f})")

mem.close()
```

### 中文对话支持

```python
from cogmem import CogMem

# 使用中文专用嵌入模型以获得更好的中文性能
mem = CogMem(
    db_path="chinese_memory.db",
    embed_model="BAAI/bge-small-zh-v1.5",  # 512维，中文优化
)

mem.add("今天和小王在星巴克喝了咖啡，他说下周要去北京出差。", user_id="user1")
results = mem.search("小王要去哪里出差？", user_id="user1")
```

### 使用基线系统（仅 FTS5）

```python
from cogmem import BaselineMemory

baseline = BaselineMemory(db_path="baseline.db")
baseline.add("I met Bob at the conference yesterday.", user_id="user1")
results = baseline.search("Who did I meet?", user_id="user1")
```

### 使用认知记忆（FTS + 扩散激活，无向量）

```python
from cogmem import CognitiveMemory

# FTS5 + 实体-关系扩散激活，不含向量嵌入
# 适用于无 SentenceTransformer 的环境，或用于消融实验评估向量检索的贡献
cognitive = CognitiveMemory(db_path="cognitive.db")
cognitive.add("I met Bob at the conference yesterday.", user_id="user1")
results = cognitive.search("Who did I meet?", user_id="user1")
```

### 多通道 LLM 故障转移

```python
from cogmem import CogMem, LLMClient

client = LLMClient([
    {"name": "deepseek", "api_key": "sk-xxx", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    {"name": "qwen", "api_key": "sk-yyy", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max"},
])

mem = CogMem(db_path="memory.db", llm_client=client)
```

## 配置

通过环境变量配置（复制 `.env.example` 为 `.env`）：

```bash
COGMEM_API_KEY=sk-your-key
COGMEM_BASE_URL=https://api.deepseek.com/v1
COGMEM_MODEL=deepseek-chat
```

支持任何 OpenAI 兼容的 API 端点：DeepSeek、OpenAI、Qwen/DashScope 等。

## 基准测试结果

在两个双语长对话数据集上进行评估：

- **CLongEval**（中文）：70 组对话，358 个问题
- **LoCoMo**（英文）：10 组对话，1,542 个问题

LLM 后端：DeepSeek-V3（所有系统使用相同 LLM 以确保公平比较）。

### 主要结果

| 系统 | CLongEval (中文) | LoCoMo (英文) | 平均 |
|------|:----------------:|:------------:|:---:|
| 基线 (FTS5) | 90.50% | 64.85% | 77.68% |
| 认知检索 (FTS+SA) | 89.39% | 68.74% | 79.07% |
| 完整 CogMem | 89.11% | **75.16%** | **82.14%** |
| **CogMem (bge-small-zh)** | **92.18%** | --- | --- |

**关键发现：**

1. **英文**：CogMem 达到 75.16%（较 FTS5 基线提升 10.31pp），验证了混合检索的价值。
2. **中文**：使用英文嵌入模型时，FTS5 基线最高（90.50%）。但切换到 `bge-small-zh-v1.5` 后，CogMem 提升至 **92.18%**，超过 FTS5 基线 1.68pp。这证明中文场景下向量检索的负面影响源于嵌入语言不匹配，而非架构问题。
3. **消融实验**：向量检索在英文上提升 6.42pp，但在中文上（使用英文嵌入）下降 0.28pp —— 11.70pp 的差异表明最优架构具有语言依赖性。

### 消融实验

| 配置 | 组件 | CLongEval | LoCoMo |
|------|------|:---------:|:------:|
| 基线 | 仅 FTS | 90.50% | 64.85% |
| +扩散激活 | FTS + SA | 89.39% (-1.11pp) | 68.74% (+3.89pp) |
| +向量 | FTS + Vec + SA | 89.11% (-1.39pp) | 75.16% (+10.31pp) |
| +bge-small-zh | FTS + Vec(zh) + SA | **92.18% (+1.68pp)** | --- |

### 分类别分析（LoCoMo，英文）

| 类别 | 问题数 | 基线 | 认知检索 | CogMem |
|------|:------:|:----:|:--------:|:------:|
| 单跳 | 282 | 64.9% (183) | 61.0% (172) | **71.3% (201)** |
| 多跳 | 321 | 63.2% (203) | 76.0% (244) | **82.2% (264)** |
| 时序 | 96 | 54.2% (52) | 44.8% (43) | **62.5% (60)** |
| 对抗性 | 841 | 66.6% (560) | 71.2% (599) | **75.1% (632)** |

CogMem 在多跳推理上取得最佳成绩（82.2%，较基线提升 19.0pp），验证了扩散激活在复杂推理中的价值。

### 多 LLM 后端评估（CLongEval）

同一 CogMem 系统使用五种不同 LLM 后端进行评估：

| LLM 后端 | 提供商 | 准确率 | 正确/总数 | 错误模式 |
|----------|--------|:------:|:---------:|:--------:|
| DeepSeek-V3 | DeepSeek | **89.1%** | 319/358 | 诚实（79.5% "未记录"） |
| GPT-5.6-sol | TokenSpace | 81.3% | 291/358 | 自信（55.2% 编造） |
| DS-V4-Flash | 火山引擎 | 80.7% | 289/358 | 均衡 |
| Qwen3.8-Flash | DashScope | 76.0% | 272/358 | 自信 |
| Qwen-Max | DashScope | 72.3% | 224/310* | 幻觉（41.9% 编造） |

*Qwen-Max 因 API 限流仅完成 58/70 组对话。

**关键洞察**：模型的"诚实校准"（愿意承认"我没有该记录"）与准确率的相关性比模型等级更强。免费模型（DS-V4-Flash，80.7%）可以超越付费旗舰模型（Qwen-Max，72.3%）。

#### 错误模式分析

| LLM 后端 | 错误数 | 未记录 | 编造 | 诚实率 | 编造率 |
|----------|:------:|:------:|:----:|:------:|:------:|
| DeepSeek-V3 | 39 | 31 | 3 | 79.5% | 7.7% |
| ARK V4-Flash | 69 | 29 | 14 | 42.0% | 20.3% |
| Qwen3.8-Flash | 86 | 23 | 38 | 26.7% | 44.2% |
| Qwen-Max | 86 | 31 | 36 | 36.0% | 41.9% |
| GPT-5.6-sol | 67 | 9 | 37 | 13.4% | 55.2% |

#### 回答风格对比

| LLM 后端 | 平均长度 | 不确定语气 | 直接回答 | 风格 |
|----------|:--------:|:---------:|:--------:|------|
| DeepSeek-V3 | 89 字符 | 12.9% | 87.1% | 详细-引用式 |
| Qwen-Max | 70 字符 | 15.6% | 84.4% | 谨慎式 |
| ARK V4-Flash | 57 字符 | 7.6% | 92.4% | 简洁式 |
| GPT-5.6-sol | 28 字符 | 2.7% | 97.3% | 极简-自信式 |
| Qwen3.8-Flash | 22 字符 | 2.9% | 97.1% | 极简-自信式 |

### 效率对比（CLongEval）

| 系统 | 写入时间 | 回答时间 | 写入 Token | 回答 Token | 记忆条目 | 数据库大小 |
|------|:--------:|:--------:|:----------:|:----------:|:--------:|:---------:|
| 基线 (FTS5) | 51 分钟 | 14 分钟 | 1.07M | 0.41M | 9,754 | 0.4 MB |
| 认知检索 | 148 分钟 | 13 分钟 | 2.89M | 0.66M | 36,025 | 13.1 MB |
| 完整 CogMem | 155 分钟 | 13 分钟 | 2.88M | 0.57M | 35,796 | 83.5 MB |

**观察：**
- CogMem 的向量检索将回答 Token 减少 13%（0.57M vs 0.66M）。
- 基线最高效（51 分钟，0.4 MB），在使用英文嵌入时中文准确率最高。
- CogMem 使用 `bge-small-zh-v1.5` 后准确率超过基线，达到 92.18%。

## 工作原理

### 记忆写入流程

1. **LLM 提取**：对话文本发送给 LLM，使用结构化提取提示（双语：根据文本内容自动选择英文或中文）。
2. **实体-关系网络**：提取的实体（人物、地点、组织、事件、事物）存储属性和摘要。关系形成有向图。
3. **扁平片段**：提取的事实、实体属性和关系三元组存储为带 FTS5 索引的扁平文本片段。
4. **向量嵌入**：每个片段使用 SentenceTransformer 编码，同时存储在 numpy 矩阵（内存）和 SQLite BLOB（持久化）中。

### 记忆检索流程

三条并行检索路径被激活并融合：

1. **FTS5/LIKE 路径**（权重：0.4）：英文使用 FTS5 BM25 排序。中文使用多关键词 LIKE 搜索，支持日期感知的时间戳加权。
2. **向量路径**（权重：0.6）：计算查询嵌入，通过 numpy 点积计算与所有存储嵌入的余弦相似度（嵌入已预归一化）。
3. **扩散激活路径**：通过 FTS5 将查询关键词匹配到实体，然后沿关系边以衰减因子 0.5 扩散激活，实现多跳关联回忆。

所有路径的结果经归一化、去重后通过加权组合合并。最终排序优先出现在多个路径中的片段。

## API 参考

### `CogMem`

```python
CogMem(
    db_path="cogmem.db",
    llm_client=None,           # LLMClient 实例（从环境变量自动创建）
    embed_model="all-MiniLM-L6-v2",  # SentenceTransformer 模型名称
    fts_weight=0.4,            # FTS5 路径融合权重
    vector_weight=0.6,         # 向量路径融合权重
)
```

**方法：**
- `add(conversation_text, user_id="", timestamp=None) -> dict` — 添加对话到记忆
- `search(query, user_id="", top_k=30) -> list[dict]` — 搜索记忆（返回 `{"memory", "score", "type"}` 列表）
- `count_memories(user_id="") -> int` — 统计存储的记忆数量
- `close()` — 关闭数据库连接

### `BaselineMemory`

```python
BaselineMemory(
    db_path="baseline_memory.db",
    llm_client=None,
)
```

与 `CogMem` 相同的 API，但仅使用 FTS5 检索（无向量或扩散激活）。

### `CognitiveMemory`

```python
CognitiveMemory(
    db_path="cognitive.db",
    llm_client=None,
)
```

与 `CogMem` 相同的 API 和实体-关系提取逻辑，但**不含向量嵌入**。仅使用 FTS5 + 扩散激活。这是 `BaselineMemory`（仅 FTS5）和完整 `CogMem`（FTS5 + 向量 + 扩散激活）之间的中间消融配置。

## 项目结构

```
cogmem/
├── cogmem/
│   ├── __init__.py        # 包导出
│   ├── memory.py          # CogMem: 三路径混合检索
│   ├── cognitive.py       # CognitiveMemory: FTS + 扩散激活（无向量）
│   ├── baseline.py        # BaselineMemory: 仅 FTS5 基线
│   ├── llm_client.py      # LLMClient: OpenAI 兼容多通道客户端
│   ├── search_utils.py    # 中文关键词提取 & LIKE 搜索
│   └── prompts.py         # 双语提取提示
├── data/
│   ├── clongeval_zh.jsonl # 中文评估数据集（70 组对话，358 个问题）
│   └── locomo10_en.json   # 英文评估数据集（10 组对话，1,542 个问题）
├── examples/
│   ├── basic_usage.py     # 快速开始示例
│   └── chinese_demo.py    # 中文对话演示
├── tests/
│   └── test_memory.py     # 单元测试
├── README.md
├── README_zh.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## 评估数据集

CogMem 在两个双语长对话基准测试上进行评估。两个数据集均包含在本仓库的 `data/` 目录下，确保可独立复现。

### 中文：CLongEval

| 项目 | 详情 |
|------|------|
| 文件 | `data/clongeval_zh.jsonl` |
| 格式 | JSONL（每行一组对话） |
| 对话数 | 70 |
| 问题数 | 358 |
| 类别 | 单跳、多跳、时序推理、对话理解 |
| 大小 | ~15 MB |
| 下载 | `wget https://github.com/honghaifeng/cogmem/raw/main/data/clongeval_zh.jsonl` |

**描述**：CLongEval 是一个中文长对话基准测试，用于评估 LLM 的中文长上下文能力。每组对话模拟用户与 AI 助手之间的多轮对话，问题测试事实回忆、多跳推理、时序推理和对话理解。数据集使用 `small.jsonl`（精简子集），包含 70 组对话和 358 个标注问题。

**数据结构**（每行 JSONL）：
```json
{
  "idx": 0,
  "conversation": [...],
  "qa_pairs": [
    {
      "question": "用户在4月25号这天和你分享过什么？",
      "answer": "...",
      "type": "time.query"  // 单跳、多跳、时序等
    }
  ]
}
```

### 英文：LoCoMo

| 项目 | 详情 |
|------|------|
| 文件 | `data/locomo10_en.json` |
| 格式 | JSON（对话对象数组） |
| 对话数 | 10 |
| 问题数 | 1,542 |
| 类别 | 单跳、多跳、时序、对抗性 |
| 大小 | ~2.7 MB |
| 下载 | `wget https://github.com/honghaifeng/cogmem/raw/main/data/locomo10_en.json` |

**描述**：LoCoMo（Long-Context Multi-Modal Output）是一个英文长对话基准测试，用于评估长期记忆系统。每组对话包含 600+ 轮自然对话，标注问题涵盖四个类别：单跳（直接事实回忆）、多跳（交叉引用多条记忆）、时序（时间依赖推理）和对抗性（设计用于触发虚假记忆的问题）。

**数据结构**：
```json
[
  {
    "conversation_id": 0,
    "session": [...],
    "qa_pairs": [
      {
        "question": "What did Alice say about her new job?",
        "answer": "...",
        "category": "single-hop"  // multi-hop, temporal, adversarial
      }
    ]
  }
]
```

### 使用数据集

```python
import json

# 加载 CLongEval（中文，JSONL）
clongeval = []
with open("data/clongeval_zh.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        clongeval.append(json.loads(line))
print(f"CLongEval: {len(clongeval)} 组对话, {sum(len(c['qa_pairs']) for c in clongeval)} 个问题")

# 加载 LoCoMo（英文，JSON）
with open("data/locomo10_en.json", "r", encoding="utf-8") as f:
    locomo = json.load(f)
print(f"LoCoMo: {len(locomo)} 组对话, {sum(len(c['qa_pairs']) for c in locomo)} 个问题")
```

### 运行评估

```bash
# CLongEval（中文，70 组对话）
python -m cogmem.eval --dataset clongeval --data data/clongeval_zh.jsonl

# LoCoMo（英文，10 组对话）
python -m cogmem.eval --dataset locomo --data data/locomo10_en.json
```

### 数据来源

| 数据集 | 原始来源 | 论文 |
|--------|---------|------|
| CLongEval | [CLongEval GitHub](https://github.com/zqx-star/CLongEval) | CLongEval: A Chinese Long-Context Evaluation Benchmark for LLMs |
| LoCoMo | [LoCoMo GitHub](https://github.com/snap-research/locomo) | LoCoMo: Long-Context Memory Evaluation (Snap Research) |

## 引用

```bibtex
@misc{hong2026cogmem,
  title={Cognitive Memory Network: Bridging Symbolic and Neural Retrieval for Long-Term Conversation Memory},
  author={Hong, Haifeng},
  year={2026},
  url={https://github.com/honghaifeng/cogmem}
}
```

## 许可证

MIT 许可证 —— 详见 [LICENSE](LICENSE)。
