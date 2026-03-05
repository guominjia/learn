# Table Understanding with RAG

A technical overview of approaches for table understanding and retrieval-augmented generation (RAG) applied to structured/tabular data.

---

## Background

Traditional RAG pipelines are optimized for **unstructured text**. Tables introduce unique challenges:

- Cells have **implicit relational meaning** that depends on row/column headers
- Queries may require **aggregation, filtering, or multi-hop reasoning** across rows and columns
- Large tables exceed LLM context windows easily
- Schema heterogeneity makes generalization difficult

Several specialized methods have been developed to address these challenges.

---

## RAPTOR

> **Recursive Abstractive Processing for Tree-Organized Retrieval**

RAPTOR is a general-purpose RAG improvement technique. It is **not table-specific**, but is relevant as a contrast to table-focused methods.

### How It Works

RAPTOR builds a hierarchical tree of document summaries through recursive clustering and summarization:

```
Raw document chunks (Leaf Nodes)
        ↓  cluster + summarize
    Mid-level summary nodes
        ↓  cluster + summarize
    Top-level summary node (Root)
```

**Steps:**
1. **Chunk** raw documents into small text segments
2. **Embed** each chunk using a text embedding model
3. **Cluster** chunks using UMAP dimensionality reduction + Gaussian Mixture Models
4. **Summarize** each cluster with an LLM
5. **Recurse** — treat summaries as new chunks and repeat until convergence
6. **Retrieve** — at query time, search across all tree levels simultaneously

### Strengths vs. Standard RAG

| Feature | Standard RAG | RAPTOR |
|--------|--------------|--------|
| Query type | Local/factual | Global/cross-segment |
| Long doc handling | Weak | Strong |
| Context granularity | Chunk-level | Multi-level |

### Limitation for Tables

RAPTOR operates on **plain text summaries**, so structural information in tables (row-column relationships) is lost during summarization.

- Paper: [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- GitHub: [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)

---

## TableRAG

> **Million-Token Table Understanding with Language Models**

TableRAG is designed specifically for large-scale table understanding. It addresses the problem of tables that are **too large to fit in a single LLM context window**.

### Core Idea

Instead of feeding the entire table to the LLM, TableRAG decomposes the table into two retrieval indexes:

| Index | Content | Purpose |
|-------|---------|---------|
| **Schema Index** | Column names + descriptions | Retrieve relevant columns |
| **Cell Index** | Individual cell values | Retrieve relevant rows/values |

Retrieval is query-driven: only the cells and columns relevant to the question are extracted and passed to the LLM.

### Architecture

```
User Query
    ↓
┌─────────────────────────────┐
│  Schema Retriever           │  → Relevant columns
│  Cell Retriever             │  → Relevant cell values
└─────────────────────────────┘
    ↓
Compact Table Context → LLM → Answer
```

### Strengths

- Handles **million-token scale** tables
- Reduces hallucination by providing focused context
- Works with standard LLMs (no fine-tuning required)

### Links

- Paper: [TableRAG: Million-Token Table Understanding with Language Models](https://arxiv.org/abs/2410.04739)
- GitHub: [google-research/google-research/table_rag](https://github.com/google-research/google-research/tree/master/table_rag)

---

## TAPAS

> **Table Parsing via Pre-training**

TAPAS is a BERT-based model fine-tuned on table-question answering tasks. It reasons directly over HTML/structured tables without converting them to text.

### Key Design

TAPAS extends BERT with **additional positional embeddings** to encode table structure:

| Embedding Type | Encodes |
|----------------|---------|
| Token position | Standard sequential position |
| Row ID | Which row the token belongs to |
| Column ID | Which column the token belongs to |
| Rank | Numerical rank of cell value within its column |
| Previous answer | Whether the cell was part of a previous answer |

This allows the model to attend to **cell relationships** rather than treating the table as flat text.

### Task Types

TAPAS supports:
- **Table QA** (e.g., WikiTableQuestions, SQA)
- **Cell selection** — identify which cells answer the question
- **Scalar aggregation** — SUM, COUNT, AVERAGE, etc.

### Example

```
Table:
| Player | Goals | Team    |
|--------|-------|---------|
| Alice  | 10    | Red     |
| Bob    | 7     | Blue    |
| Carol  | 15    | Red     |

Query: "How many total goals did Red team score?"
TAPAS → selects {Alice:10, Carol:15} → aggregation: SUM → 25
```

### Links

- Paper: [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/abs/2004.02349)
- GitHub: [google-research/tapas](https://github.com/google-research/tapas)
- HuggingFace: [google/tapas-base](https://huggingface.co/google/tapas-base)

---

## OmniTab

> **Pretraining with Natural and Synthetic Data for Few-shot Table-based QA**

OmniTab is a pre-trained model that jointly learns from **natural language** and **synthetic SQL-generated** table-question pairs to improve few-shot generalization.

### Motivation

Supervised table QA models require large labeled datasets. OmniTab addresses this by pre-training on:

1. **Natural data** — real NL questions paired with tables (e.g., NQ-Tables)
2. **Synthetic data** — SQL queries executed on tables, automatically converted to NL question-answer pairs

### Training Signal

```
Natural:   (Table, "Who scored the most?") → "Carol"
Synthetic: (Table, SQL: SELECT MAX(Goals)) → auto-generated NL → "Carol"
```

This dual-source pre-training teaches the model to understand both **natural language intent** and **structured table semantics**.

### Strengths vs. TAPAS

| Feature | TAPAS | OmniTab |
|---------|-------|---------|
| Training data | Supervised only | Natural + Synthetic |
| Few-shot ability | Limited | Strong |
| Base model | BERT | BART (seq2seq) |
| Output format | Cell selection | Free-form text |

### Links

- Paper: [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/abs/2207.03637)
- HuggingFace: [neulab/omnitab-large](https://huggingface.co/neulab/omnitab-large)

---

## Comparison Summary

| Method | Table-Specific | Approach | Scale | Fine-tuning Needed |
|--------|---------------|----------|-------|--------------------|
| RAPTOR | No | Hierarchical text summarization | Document-level | No |
| TableRAG | Yes | Schema + Cell retrieval | Million-token tables | No |
| TAPAS | Yes | Structured BERT encoding | Single table | Yes |
| OmniTab | Yes | Seq2seq + dual pre-training | Single table | Few-shot |

---

## When to Use What

- **Large tables, no fine-tuning** → **TableRAG**
- **Structured QA with aggregation** → **TAPAS**
- **Few-shot / low-resource table QA** → **OmniTab**
- **Long unstructured documents with some tables** → **RAPTOR** (text portions only)

---

## References

1. Sarthi et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)
2. Chen et al. (2024). *TableRAG: Million-Token Table Understanding with Language Models*. [arXiv:2410.04739](https://arxiv.org/abs/2410.04739)
3. Herzig et al. (2020). *TAPAS: Weakly Supervised Table Parsing via Pre-training*. [arXiv:2004.02349](https://arxiv.org/abs/2004.02349)
4. Jiang et al. (2022). *OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering*. [arXiv:2207.03637](https://arxiv.org/abs/2207.03637)
