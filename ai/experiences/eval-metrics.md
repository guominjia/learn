# Understanding Information Retrieval Evaluation Metrics: NDCG, MAP, and MRR

When evaluating a retrieval or ranking system — such as a search engine, a RAG pipeline, or a recommendation recommender — you need rigorous metrics to know how well the system is actually performing. Three of the most widely used metrics are **NDCG@K**, **MAP@K**, and **MRR@K**. This post explains what they measure, how to compute them, and walks through a concrete worked example.

---

## Context: Where Do These Metrics Appear?

A typical evaluation call with a library like `ranx` looks like this:

```python
from ranx import Qrels, Run, evaluate

print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
```

- `Qrels` — the ground-truth relevance judgments (query → document → relevance grade).
- `Run` — the system's ranked output (query → document → score).
- The third argument is a list of metric strings with cutoff values.

---

## The Three Metrics

### 1. NDCG@K — Normalized Discounted Cumulative Gain

**What it measures:** Ranking quality when documents have *graded* relevance (e.g., 0 = irrelevant, 1 = partially relevant, 3 = highly relevant). Documents placed higher in the ranking contribute more to the score because the gain is *discounted* by their position.

**Formula:**

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

where IDCG@K is the DCG of the ideal (perfectly sorted) ranking.

**Range:** [0, 1] — higher is better.

---

### 2. MAP@K — Mean Average Precision

**What it measures:** For each query, it computes the *Average Precision* (AP) across the top-K results — i.e., at every position where a relevant document appears, it records the precision at that position and then averages over all relevant documents found. MAP is then the mean AP across all queries.

**Formula (binary relevance):**

$$
\text{AP@K} = \frac{1}{|\text{relevant}|} \sum_{i=1}^{K} \text{Precision@i} \cdot \mathbb{1}[rel_i \geq 1]
$$

$$
\text{MAP@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@K}_q
$$

**Range:** [0, 1] — higher is better.

---

### 3. MRR@K — Mean Reciprocal Rank

**What it measures:** For each query, it finds the rank position *r* of the **first** relevant document (within the top K). The score for that query is 1/r. MRR averages this over all queries. It is most useful when users care primarily about the single best answer.

**Formula:**

$$
\text{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q} \quad \text{(where } r_q \leq K \text{, else 0)}
$$

**Range:** [0, 1] — higher is better.

---

## Worked Example

Consider two queries with ranked result lists. Each number is the relevance grade of the document at that position:

| Position | q1 grades | q2 grades |
|----------|-----------|-----------|
| 1        | 3         | 0         |
| 2        | 2         | 1         |
| 3        | 0         | 0         |
| 4        | 1         | 0         |
| 5        | 0         | 1         |

### Computing NDCG@10

```python
import math

def dcg(rels, k=10):
    return sum((2**rels[i] - 1) / math.log2(i + 2) for i in range(min(len(rels), k)))

def idcg(rels, k=10):
    return dcg(sorted(rels, reverse=True), k)

def ndcg(rels, k=10):
    ideal = idcg(rels, k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0

q1 = [3, 2, 0, 1, 0]
q2 = [0, 1, 0, 0, 1]

ndcg1 = ndcg(q1)   # ≈ 0.993  (highly relevant docs appear at ranks 1 and 2)
ndcg2 = ndcg(q2)   # ≈ 0.624  (first relevant doc only appears at rank 2)

ndcg_at_10 = (ndcg1 + ndcg2) / 2  # ≈ 0.808
```

- q1 scores close to 1.0 because the highest-grade documents appear first.
- q2 scores lower because rank 1 returns an irrelevant document.

### Computing MAP@5

```python
def ap_at_k(rels, k=5):
    num_rel, sum_prec = 0, 0.0
    for i in range(min(len(rels), k)):
        if rels[i] >= 1:
            num_rel += 1
            sum_prec += num_rel / (i + 1)
    return sum_prec / num_rel if num_rel > 0 else 0.0

ap1 = ap_at_k(q1)   # ≈ 0.917  (relevant docs at positions 1, 2, 4)
ap2 = ap_at_k(q2)   # ≈ 0.450  (relevant docs at positions 2, 5)

map_at_5 = (ap1 + ap2) / 2  # ≈ 0.683
```

- q1: precision at rank 1 = 1.0, rank 2 = 1.0, rank 4 = 0.75 → AP = (1.0 + 1.0 + 0.75) / 3 ≈ 0.917.
- q2: precision at rank 2 = 0.5, rank 5 = 0.4 → AP = (0.5 + 0.4) / 2 = 0.45.

### Computing MRR@10

```python
def rr_at_k(rels, k=10):
    for i in range(min(len(rels), k)):
        if rels[i] >= 1:
            return 1.0 / (i + 1)
    return 0.0

rr1 = rr_at_k(q1)   # 1.0   (first relevant doc is at rank 1)
rr2 = rr_at_k(q2)   # 0.5   (first relevant doc is at rank 2)

mrr_at_10 = (rr1 + rr2) / 2  # 0.75
```

### Summary Table

| Metric     | q1     | q2     | Average |
|------------|--------|--------|---------|
| NDCG@10    | 0.993  | 0.624  | **0.808** |
| MAP@5      | 0.917  | 0.450  | **0.683** |
| MRR@10     | 1.000  | 0.500  | **0.750** |

---

## Key Takeaways

| Metric | Relevance type | Cares about position | Cares about all relevant docs |
|--------|---------------|---------------------|-------------------------------|
| NDCG@K | Graded         | Yes (log discount)   | Yes                           |
| MAP@K  | Binary         | Yes                  | Yes                           |
| MRR@K  | Binary         | Yes (first-hit only) | No                            |

- Use **NDCG** when your ground truth has graded relevance levels (e.g., 0/1/2/3).
- Use **MAP** when relevance is binary and you care about overall recall within the top K.
- Use **MRR** when the primary goal is surfacing at least one correct answer as high as possible (e.g., question answering, navigational search).

In most RAG or dense retrieval benchmarks, reporting all three gives a complete picture: NDCG captures ranking quality holistically, MAP measures how well all relevant items are surfaced early, and MRR tells you how quickly users find their first useful result.
