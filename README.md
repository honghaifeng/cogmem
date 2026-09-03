# CogMem: Cognitive Memory Network

**A hybrid memory system for LLM agents that bridges symbolic retrieval, neural retrieval, and graph-based spreading activation.**

CogMem maintains three parallel memory structures — flat text fragments with FTS5 indexing, an entity-relation knowledge network with spreading activation, and dense vector embeddings — and fuses them through weighted combination for robust long-term conversation memory.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CogMem Architecture                      │
├─────────────────────────────────────────────────────────┤
│                                                            │
│  Memory Storage                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Flat Fragments│  │ Entity-Rel   │  │  Vector      │    │
│  │ (FTS5 index)  │  │ Network      │  │  Embeddings  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐    │
│  │ Path 1:      │  │ Path 3:      │  │ Path 2:      │    │
│  │ FTS5 Keyword │  │ Spreading    │  │ Vector       │    │
│  │ Search       │  │ Activation   │  │ Similarity   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                  ┌────────▼───────┐                       │
│                  │ Weighted Fusion │                       │
│                  │ & Deduplication │                       │
│                  └────────────────┘                       │
│                                                            │
└─────────────────────────────────────────────────────────┘
```

**Three retrieval paths:**

| Path | Method | Strength |
|------|--------|----------|
| Symbolic | FTS5 full-text search + Chinese LIKE matching | Precise keyword/names/dates matching |
| Neural | Dense vector cosine similarity (SentenceTransformer) | Semantic paraphrase matching |
| Structured | Entity-relation graph + 1-hop spreading activation | Multi-hop associative recall |

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from cogmem import CogMem

# Initialize with default settings (DeepSeek API from env)
mem = CogMem(db_path="my_memory.db")

# Add conversation content
mem.add("I had lunch with Alice at the Italian restaurant on 5th Avenue. She mentioned she's moving to Tokyo next month.",
        user_id="user1")

# Search memories
results = mem.search("Where is Alice moving?", user_id="user1")
for r in results:
    print(f"[{r['type']}] {r['memory']} (score: {r['score']:.3f})")

mem.close()
```

### Chinese Conversation Support

```python
from cogmem import CogMem

# Use Chinese-specific embedding model for better Chinese performance
mem = CogMem(
    db_path="chinese_memory.db",
    embed_model="BAAI/bge-small-zh-v1.5",  # 512-dim, Chinese-optimized
)

mem.add("今天和小王在星巴克喝了咖啡，他说下周要去北京出差。", user_id="user1")
results = mem.search("小王要去哪里出差？", user_id="user1")
```

### Using Baseline (FTS5-only)

```python
from cogmem import BaselineMemory

baseline = BaselineMemory(db_path="baseline.db")
baseline.add("I met Bob at the conference yesterday.", user_id="user1")
results = baseline.search("Who did I meet?", user_id="user1")
```

### Multi-Channel LLM Failover

```python
from cogmem import CogMem, LLMClient

client = LLMClient([
    {"name": "deepseek", "api_key": "sk-xxx", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    {"name": "qwen", "api_key": "sk-yyy", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-max"},
])

mem = CogMem(db_path="memory.db", llm_client=client)
```

## Configuration

Configure via environment variables (copy `.env.example` to `.env`):

```bash
COGMEM_API_KEY=sk-your-key
COGMEM_BASE_URL=https://api.deepseek.com/v1
COGMEM_MODEL=deepseek-chat
```

Any OpenAI-compatible API endpoint works: DeepSeek, OpenAI, Qwen/DashScope, etc.

## Benchmark Results

Evaluated on two bilingual long-conversation datasets:

- **CLongEval** (Chinese): 70 conversations, 358 questions
- **LoCoMo** (English): 10 conversations, 1,542 questions

LLM backend: DeepSeek-V3 (all systems use the same LLM for fair comparison).

### Main Results

| System | CLongEval (ZH) | LoCoMo (EN) | Average |
|--------|:--------------:|:-----------:|:-------:|
| Baseline (FTS5) | **90.50%** | 64.85% | 77.68% |
| Cognitive (FTS+SA) | 89.39% | 68.74% | 79.07% |
| Full CogMem | 89.11% | 75.16% | 82.14% |
| **CogMem (bge-small-zh)** | **92.18%** | --- | --- |
| Mem0 (official) | 85.47% | **81.98%** | **83.73%** |
| A-Mem (official) | 87.43% | 75.94% | 81.69% |

**Key findings:**

1. **English**: CogMem achieves 75.16% (+10.31pp over FTS5 baseline). Mem0 is highest at 81.98%.
2. **Chinese**: With English-focused embeddings, FTS5 baseline is highest (90.50%). But switching to `bge-small-zh-v1.5` raises CogMem to **92.18%**, surpassing FTS5 by +1.68pp. This proves the negative effect of vector retrieval on Chinese stems from embedding-language mismatch, not the architecture.
3. **Ablation**: Vector retrieval provides +6.42pp on English but -0.28pp on Chinese (with English embeddings) — a 11.70pp swing showing language-dependent optimal architecture.

### Ablation Study

| Configuration | Components | CLongEval | LoCoMo |
|--------------|------------|:---------:|:------:|
| Baseline | FTS only | 90.50% | 64.85% |
| +Spreading Activation | FTS + SA | 89.39% (-1.11pp) | 68.74% (+3.89pp) |
| +Vector | FTS + Vec + SA | 89.11% (-1.39pp) | 75.16% (+10.31pp) |
| +bge-small-zh | FTS + Vec(zh) + SA | **92.18% (+1.68pp)** | --- |

### Category-wise Analysis (LoCoMo, English)

| Category | # Q | Baseline | Cognitive | CogMem | A-Mem |
|----------|:---:|:--------:|:---------:|:------:|:-----:|
| Single-hop | 282 | 64.9% (183) | 61.0% (172) | 71.3% (201) | **80.5% (227)** |
| Multi-hop | 321 | 63.2% (203) | 76.0% (244) | **82.2% (264)** | 65.7% (211) |
| Temporal | 96 | 54.2% (52) | 44.8% (43) | **62.5% (60)** | 58.3% (56) |
| Adversarial | 841 | 66.6% (560) | 71.2% (599) | 75.1% (632) | **80.3% (675)** |

CogMem achieves the best multi-hop reasoning (82.2%, +19.0pp over baseline, +16.5pp over A-Mem), validating the value of spreading activation for complex reasoning.

### Multi-LLM Backend Evaluation (CLongEval)

The same CogMem system evaluated with five different LLM backends:

| LLM Backend | Provider | Accuracy | Correct/Total | Error Pattern |
|-------------|----------|:--------:|:-------------:|:------------:|
| DeepSeek-V3 | DeepSeek | **89.1%** | 319/358 | Honest (79.5% "not recorded") |
| GPT-5.6-sol | TokenSpace | 81.3% | 291/358 | Confident (55.2% fabrication) |
| DS-V4-Flash | Volcengine | 80.7% | 289/358 | Balanced |
| Qwen3.8-Flash | DashScope | 76.0% | 272/358 | Confident |
| Qwen-Max | DashScope | 72.3% | 224/310* | Hallucination (41.9% fab.) |

*Qwen-Max completed 58/70 conversations due to API rate limits.

**Key insight**: A model's "honesty calibration" (willingness to admit "I don't have that record") correlates more strongly with accuracy than model tier. Free models (DS-V4-Flash, 80.7%) can outperform paid flagships (Qwen-Max, 72.3%).

#### Error Pattern Analysis

| LLM Backend | # Errors | Not Recorded | Fabrication | Honesty Rate | Fab. Rate |
|-------------|:--------:|:------------:|:-----------:|:-----------:|:---------:|
| DeepSeek-V3 | 39 | 31 | 3 | 79.5% | 7.7% |
| ARK V4-Flash | 69 | 29 | 14 | 42.0% | 20.3% |
| Qwen3.8-Flash | 86 | 23 | 38 | 26.7% | 44.2% |
| Qwen-Max | 86 | 31 | 36 | 36.0% | 41.9% |
| GPT-5.6-sol | 67 | 9 | 37 | 13.4% | 55.2% |

#### Answer Style Comparison

| LLM Backend | Avg Length | Uncertainty | Direct | Style |
|-------------|:----------:|:-----------:|:------:|-------|
| DeepSeek-V3 | 89 chars | 12.9% | 87.1% | Detailed-cited |
| Qwen-Max | 70 chars | 15.6% | 84.4% | Hedged |
| ARK V4-Flash | 57 chars | 7.6% | 92.4% | Concise |
| GPT-5.6-sol | 28 chars | 2.7% | 97.3% | Minimal-confident |
| Qwen3.8-Flash | 22 chars | 2.9% | 97.1% | Minimal-confident |

### Efficiency Comparison (CLongEval)

| System | Write Time | Answer Time | Write Tokens | Answer Tokens | Memory Entries | DB Size |
|--------|:----------:|:-----------:|:------------:|:-------------:|:--------------:|:-------:|
| Baseline (FTS5) | 51 min | 14 min | 1.07M | 0.41M | 9,754 | 0.4 MB |
| Cognitive | 148 min | 13 min | 2.89M | 0.66M | 36,025 | 13.1 MB |
| Full CogMem | 155 min | 13 min | 2.88M | 0.57M | 35,796 | 83.5 MB |
| A-Mem | 408 min | 15 min | N/A | 1.92M | 2,734 | N/A |
| Mem0 | 87 min | N/A | N/A | N/A | 6,442 | N/A |

**Observations:**
- A-Mem is 2.6x slower than CogMem on write (408 min vs 155 min) due to 3 LLM calls per memory.
- CogMem's vector retrieval reduces answer tokens by 13% vs Cognitive-only (0.57M vs 0.66M).
- Baseline is most efficient (51 min, 0.4 MB) and achieves highest Chinese accuracy with English embeddings.
- CogMem with `bge-small-zh-v1.5` surpasses baseline accuracy at 92.18%.

## How It Works

### Memory Write Flow

1. **LLM Extraction**: Conversation text is sent to the LLM with a structured extraction prompt (bilingual: English or Chinese based on text content).
2. **Entity-Relation Network**: Extracted entities (persons, locations, organizations, events, things) are stored with properties and summaries. Relations form a directed graph.
3. **Flat Fragments**: Extracted facts, entity properties, and relation triples are stored as flat text fragments with FTS5 indexing.
4. **Vector Embeddings**: Each fragment is encoded using SentenceTransformer and stored in both a numpy matrix (in-memory) and SQLite BLOB (persistent).

### Memory Search Flow

Three parallel retrieval paths are activated and fused:

1. **FTS5/LIKE Path** (weight: 0.4): For English, uses FTS5 BM25 ranking. For Chinese, uses multi-keyword LIKE search with date-aware timestamp boosting.
2. **Vector Path** (weight: 0.6): Computes query embedding, calculates cosine similarity against all stored embeddings via numpy dot product (embeddings are pre-normalized).
3. **Spreading Activation Path**: Matches query keywords to entities via FTS5, then spreads activation along relation edges with decay factor 0.5, enabling multi-hop associative recall.

Results from all paths are normalized, deduplicated, and merged through weighted combination. Final ranking prioritizes fragments that appear in multiple paths.

## API Reference

### `CogMem`

```python
CogMem(
    db_path="cogmem.db",
    llm_client=None,           # LLMClient instance (auto-created from env)
    embed_model="all-MiniLM-L6-v2",  # SentenceTransformer model name
    fts_weight=0.4,            # FTS5 path fusion weight
    vector_weight=0.6,         # Vector path fusion weight
)
```

**Methods:**
- `add(conversation_text, user_id="", timestamp=None) -> dict` — Add conversation to memory
- `search(query, user_id="", top_k=30) -> list[dict]` — Search memories (returns list of `{"memory", "score", "type"}`)
- `count_memories(user_id="") -> int` — Count stored memories
- `close()` — Close database connection

### `BaselineMemory`

```python
BaselineMemory(
    db_path="baseline_memory.db",
    llm_client=None,
)
```

Same API as `CogMem` but with FTS5-only retrieval (no vector or spreading activation).

## Project Structure

```
cogmem/
├── cogmem/
│   ├── __init__.py        # Package exports
│   ├── memory.py          # CogMem: three-path hybrid retrieval
│   ├── baseline.py        # BaselineMemory: FTS5-only baseline
│   ├── llm_client.py      # LLMClient: OpenAI-compatible multi-channel client
│   ├── search_utils.py    # Chinese keyword extraction & LIKE search
│   └── prompts.py         # Bilingual extraction prompts
├── data/
│   ├── clongeval_zh.jsonl # Chinese evaluation dataset (70 conv, 358 Q)
│   └── locomo10_en.json   # English evaluation dataset (10 conv, 1,542 Q)
├── examples/
│   ├── basic_usage.py     # Quick start example
│   └── chinese_demo.py     # Chinese conversation demo
├── tests/
│   └── test_memory.py     # Unit tests
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Evaluation Datasets

CogMem is evaluated on two bilingual long-conversation benchmarks. Both datasets are included in this repository under `data/` for self-contained reproducibility.

### Chinese: CLongEval

| Item | Detail |
|------|--------|
| File | `data/clongeval_zh.jsonl` |
| Format | JSONL (one conversation per line) |
| Conversations | 70 |
| Questions | 358 |
| Categories | Single-hop, Multi-hop, Temporal Reasoning, Conversation Understanding |
| Size | ~15 MB |
| Download | `wget https://github.com/honghaifeng/cogmem/raw/main/data/clongeval_zh.jsonl` |

**Description**: CLongEval is a Chinese long-conversation benchmark designed to evaluate LLM long-context capabilities in Chinese. Each conversation simulates multi-turn dialogue between a user and an AI assistant, with questions testing factual recall, multi-hop reasoning, temporal reasoning, and dialogue comprehension. The dataset uses `small.jsonl` (the compact subset) which contains 70 conversations with 358 annotated questions.

**Data structure** (each JSONL line):
```json
{
  "idx": 0,
  "conversation": [...],
  "qa_pairs": [
    {
      "question": "用户在4月25号这天和你分享过什么？",
      "answer": "...",
      "type": "time.query"  // single-hop, multi-hop, temporal, etc.
    }
  ]
}
```

### English: LoCoMo

| Item | Detail |
|------|--------|
| File | `data/locomo10_en.json` |
| Format | JSON (array of conversation objects) |
| Conversations | 10 |
| Questions | 1,542 |
| Categories | Single-hop, Multi-hop, Temporal, Adversarial |
| Size | ~2.7 MB |
| Download | `wget https://github.com/honghaifeng/cogmem/raw/main/data/locomo10_en.json` |

**Description**: LoCoMo (Long-Context Multi-Modal Output) is an English long-conversation benchmark for evaluating long-term memory systems. Each conversation contains 600+ turns of natural dialogue between two speakers, with annotated questions across four categories: single-hop (direct fact recall), multi-hop (cross-reference multiple memories), temporal (time-dependent reasoning), and adversarial (questions designed to trigger false memories).

**Data structure**:
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

### Using the Datasets

```python
import json

# Load CLongEval (Chinese, JSONL)
clongeval = []
with open("data/clongeval_zh.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        clongeval.append(json.loads(line))
print(f"CLongEval: {len(clongeval)} conversations, {sum(len(c['qa_pairs']) for c in clongeval)} questions")

# Load LoCoMo (English, JSON)
with open("data/locomo10_en.json", "r", encoding="utf-8") as f:
    locomo = json.load(f)
print(f"LoCoMo: {len(locomo)} conversations, {sum(len(c['qa_pairs']) for c in locomo)} questions")
```

### Running Evaluations

```bash
# CLongEval (Chinese, 70 conversations)
python -m cogmem.eval --dataset clongeval --data data/clongeval_zh.jsonl

# LoCoMo (English, 10 conversations)
python -m cogmem.eval --dataset locomo --data data/locomo10_en.json
```

### Dataset Sources

| Dataset | Original Source | Paper |
|---------|----------------|-------|
| CLongEval | [CLongEval GitHub](https://github.com/zqx-star/CLongEval) | CLongEval: A Chinese Long-Context Evaluation Benchmark for LLMs |
| LoCoMo | [LoCoMo GitHub](https://github.com/snap-research/locomo) | LoCoMo: Long-Context Memory Evaluation (Snap Research) |

## Citation

```bibtex
@misc{hong2026cogmem,
  title={Cognitive Memory Network: Bridging Symbolic and Neural Retrieval for Long-Term Conversation Memory},
  author={Hong, Haifeng},
  year={2026},
  url={https://github.com/honghaifeng/cogmem}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
