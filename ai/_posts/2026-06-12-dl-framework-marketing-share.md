---
layout: post
title: "Where to Find (Near) Real-Time Deep Learning Framework Market Share Data"
date: 2026-06-12
tags: [Deep Learning, PyTorch, TensorFlow, JAX, Market Share, Data]
---

## Why this question is tricky

If you ask for **real-time market share** of deep learning frameworks, there is no single official dashboard that gives a perfect answer.

What we can get in practice is **near real-time adoption signals** from multiple public sources.

## The best web sources to track framework share

### 1) Papers With Code
- Website: [paperswithcode.com](https://paperswithcode.com/)
- What it tells you: research popularity (which framework appears in SOTA repos and paper implementations)
- Strength: very good for research trend momentum
- Limitation: not equal to enterprise production usage

### 2) Kaggle Survey
- Website: [kaggle.com](https://www.kaggle.com/)
- What it tells you: practitioner self-reported usage (annual snapshot)
- Strength: broad data-science community signal
- Limitation: annual, not real-time

### 3) Stack Overflow Survey + Trends
- Websites:
  - [survey.stackoverflow.co](https://survey.stackoverflow.co/)
  - [stackoverflow.com/trends](https://stackoverflow.com/trends)
- What it tells you: community discussion and interest trends
- Strength: useful long-term direction signal
- Limitation: discussion volume is not direct market share

### 4) PyPI Stats
- Website: [pypistats.org](https://pypistats.org/)
- What it tells you: package download activity (e.g., `torch`, `tensorflow`, `jax`)
- Strength: frequent updates, easy to track over time
- Limitation: downloads can be inflated by CI/CD and mirrors

### 5) GitHub + Octoverse
- Websites:
  - [github.com](https://github.com/)
  - [octoverse.github.com](https://octoverse.github.com/)
- What it tells you: repo activity, stars, contributors, ecosystem growth
- Strength: captures open-source developer momentum
- Limitation: stars are noisy and can lag actual production adoption

### 6) Hugging Face ecosystem signals
- Website: [huggingface.co](https://huggingface.co/)
- What it tells you: practical model and library adoption in modern ML workflows
- Strength: strong signal for current GenAI/NLP/CV usage patterns
- Limitation: ecosystem is broad, not a strict framework vote

## How to build a “best-effort live” market-share dashboard

Track three dimensions weekly:

1. **Research share**
	- Metric examples: Papers With Code framework mentions, SOTA implementation distribution

2. **Developer share**
	- Metric examples: GitHub stars growth, active contributors, issue/PR velocity

3. **Usage proxy share**
	- Metric examples: PyPI download trend for `torch`, `tensorflow`, `jax`

Then normalize each metric and combine with weighted scoring:

$$
	ext{Composite Share} = 0.4 \cdot \text{Research} + 0.3 \cdot \text{Developer} + 0.3 \cdot \text{Usage Proxy}
$$

> The weights are configurable. If your focus is production usage, increase the usage proxy weight.

## Suggested tracking cadence

- **Weekly**: PyPI + GitHub metrics
- **Monthly**: Papers With Code trend snapshot
- **Quarterly**: interpretation review (remove anomalies, tune weights)
- **Yearly**: compare against Kaggle/Stack Overflow survey baselines

## Practical conclusions (2026)

- **PyTorch** remains strongest in research mindshare and modern open-source momentum.
- **TensorFlow/Keras** still shows meaningful production and legacy enterprise footprint.
- **JAX** continues to grow in high-performance research niches.
- Smaller frameworks can win specific verticals, but ecosystem depth decides long-term share.

## Final takeaway

There is no perfect real-time “market share” API for deep learning frameworks.
The reliable approach is to combine multiple public signals and maintain a transparent, repeatable scoring method.

If you publish the metric definition and weights, your trend report becomes far more credible than any single-number claim.