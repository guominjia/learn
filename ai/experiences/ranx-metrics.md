# Understanding NDCG Evaluation with ranx

## Background

When building a Retrieval-Augmented Generation (RAG) system, evaluating retrieval quality is critical. **ranx** is a fast, lightweight Python library designed for information retrieval evaluation. At its core, it separates two concerns:

- **Qrels** (Query Relevance Judgments): the ground-truth relevance labels annotated by humans
- **Run**: the ranked list of documents your system returns, along with their scores

Understanding why both are needed — and exactly how metrics like NDCG are computed — is essential for interpreting evaluation results correctly.

---

## Setup

```python
from ranx import Qrels, Run, evaluate

qrels_dict = {
    "q_1": { "d_12": 5, "d_25": 3 },
    "q_2": { "d_11": 6, "d_22": 1 }
}

run_dict = {
    "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.7,
             "d_36": 0.6, "d_32": 0.5, "d_35": 0.4 },
    "q_2": { "d_12": 0.9, "d_11": 0.8, "d_25": 0.7,
             "d_36": 0.6, "d_22": 0.5, "d_35": 0.4 }
}

qrels = Qrels(qrels_dict)
run   = Run(run_dict)

evaluate(qrels, run, "ndcg@5")
# >>> 0.7861

evaluate(qrels, run, ["map@5", "mrr"])
# >>> {"map@5": 0.6416, "mrr": 0.75}
```

---

## Why Do We Need Both Qrels and Run?

A common question: *qrels already contains relevance scores — why can't we compute NDCG from qrels alone?*

The key distinction is:

| Object | Role | Data |
|--------|------|------|
| **Qrels** | Ground truth | Relevance labels per (query, doc) pair |
| **Run** | System output | Ranking scores your system assigned |

**NDCG = DCG(run) / IDCG(qrels)**

- **IDCG** (Ideal DCG) is computed from qrels — it represents the best possible ranking, where the most relevant documents come first.
- **DCG** is computed from the run — it measures how well your system actually ranked the documents.

Without the run, there is no ranking to evaluate. Without qrels, there is no ideal to compare against.

---

## The Role of System Scores (0.9, 0.8, ...)

The scores in `run_dict` (e.g. 0.9, 0.8, 0.7...) are the retrieval confidence scores produced by your system (e.g. cosine similarity, BM25 score, or a neural reranker score).

**These scores do NOT enter the DCG formula directly.** Their only job is to determine the rank order of documents for each query. Once sorted, ranx looks up each document's relevance label from qrels and places it at the corresponding rank position.

```
run ranking for q_1 (by score descending):
rank 1 → d_12 (score=0.9)  → rel=5   (from qrels)
rank 2 → d_23 (score=0.8)  → rel=0   (not in qrels)
rank 3 → d_25 (score=0.7)  → rel=3   (from qrels)
rank 4 → d_36 (score=0.6)  → rel=0
rank 5 → d_32 (score=0.5)  → rel=0
```

Only the relevance labels at each rank position matter for the DCG calculation.

---

## Step-by-Step NDCG@5 Calculation

ranx uses the classic DCG formula:

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

### Query q_1

**Run ranking (top 5) with relevance labels:**

| Rank | Doc  | Score | Relevance |
|------|------|-------|-----------|
| 1    | d_12 | 0.9   | 5         |
| 2    | d_23 | 0.8   | 0         |
| 3    | d_25 | 0.7   | 3         |
| 4    | d_36 | 0.6   | 0         |
| 5    | d_32 | 0.5   | 0         |

$$\text{DCG@5} = \frac{5}{\log_2 2} + \frac{0}{\log_2 3} + \frac{3}{\log_2 4} = 5 + 0 + 1.5 = 6.5$$

**Ideal ranking (from qrels):** d_12(5), d_25(3)

$$\text{IDCG@5} = \frac{5}{\log_2 2} + \frac{3}{\log_2 3} = 5 + \frac{3}{1.585} \approx 5 + 1.893 = 6.893$$

$$\text{NDCG@5}_{q_1} = \frac{6.5}{6.893} \approx 0.9430$$

---

### Query q_2

**Run ranking (top 5) with relevance labels:**

| Rank | Doc  | Score | Relevance |
|------|------|-------|-----------|
| 1    | d_12 | 0.9   | 0         |
| 2    | d_11 | 0.8   | 6         |
| 3    | d_25 | 0.7   | 0         |
| 4    | d_36 | 0.6   | 0         |
| 5    | d_22 | 0.5   | 1         |

$$\text{DCG@5} = \frac{0}{\log_2 2} + \frac{6}{\log_2 3} + 0 + 0 + \frac{1}{\log_2 6} = \frac{6}{1.585} + \frac{1}{2.585} \approx 3.786 + 0.387 = 4.173$$

**Ideal ranking (from qrels):** d_11(6), d_22(1)

$$\text{IDCG@5} = \frac{6}{\log_2 2} + \frac{1}{\log_2 3} = 6 + \frac{1}{1.585} \approx 6 + 0.631 = 6.631$$

$$\text{NDCG@5}_{q_2} = \frac{4.173}{6.631} \approx 0.6293$$

---

### Final Score (averaged over queries)

$$\text{NDCG@5} = \frac{0.9430 + 0.6293}{2} \approx 0.7861$$

This matches the ranx output exactly.

---

## Key Takeaways

1. **Qrels = ground truth labels**; Run = system output rankings. Both are required for NDCG.
2. **System scores only determine rank order** — their absolute values do not affect DCG.
3. **NDCG normalizes DCG against the ideal**, so it always falls in [0, 1].
4. A low NDCG often means highly relevant documents are being ranked too low — not necessarily that irrelevant documents are being returned.
5. When the ideal document appears at rank 1 in both qrels and run (like d_12 for q_1), that strongly boosts NDCG since it contributes the most to DCG (divided by log2(2) = 1).
