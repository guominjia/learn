# Does DP Always Need Probabilities? A Look at DP vs. Monte Carlo in Reinforcement Learning

While reading *Reinforcement Learning: An Introduction (2nd edition)* by Sutton & Barto, I came across this sentence:

> "All of the probabilities must be computed before DP can be applied, and such computations are often complex and error-prone."

My first reaction was: **Does Dynamic Programming always require probabilities?**

---

## DP in General — No Probabilities Needed

In the general sense, Dynamic Programming is just an algorithm design paradigm built on two ideas:

- **Optimal substructure**: the optimal solution to a problem can be built from optimal solutions to its subproblems
- **Overlapping subproblems**: subproblems recur, so caching their solutions saves work

Classic examples like the knapsack problem, Longest Common Subsequence, or Floyd-Warshall shortest paths — none of these involve probabilities at all.

So in general, **DP has nothing to do with probabilities**.

---

## DP in Reinforcement Learning — Probabilities Are Mandatory

The context of that quote changes everything. In reinforcement learning, DP methods like **Policy Iteration** and **Value Iteration** are built on the Bellman expectation equation:

$$V(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right]$$

The term $p(s', r \mid s, a)$ is the **state transition probability** — the probability that taking action $a$ in state $s$ leads to state $s'$ and reward $r$.

This means DP in RL is **model-based**: you need a complete and accurate model of the environment's dynamics before you can do anything. In practice, building that model is often the hard part. The probabilities may be unknown, difficult to estimate, or computationally expensive to compute across all states and actions. That's exactly what Sutton & Barto are pointing out.

---

## How Monte Carlo Sidesteps This Problem

Monte Carlo methods take a completely different approach: **don't model the environment at all**. Instead, just let the agent interact with it and collect real experience.

The value of a state is estimated by averaging the actual returns observed after visiting it:

$$V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}$$

No transition probabilities. No model. Just sampled trajectories. This makes Monte Carlo a **model-free** method — and in many real-world problems, that's a significant practical advantage.

---

## DP vs. Monte Carlo — Core Differences

| | Dynamic Programming | Monte Carlo |
|---|---|---|
| Requires probability model | Yes — needs $p(s', r \mid s, a)$ | No |
| Update mechanism | Iterative computation (bootstrapping) | Full-episode return averaging |
| Data source | Environment model | Sampled trajectories |
| Bias / Variance | Low variance, but model error risk | Unbiased, but high variance |
| Applicability | Only when model is known | Works in model-free settings |

---

## Takeaway

The quote from Sutton & Barto is specifically about DP as used in reinforcement learning, where it is a model-based method that fundamentally depends on knowing the environment's transition probabilities. Computing those probabilities in advance is not only technically demanding — it's sometimes impossible.

This is one of the key reasons why model-free methods like Monte Carlo (and TD learning, Q-learning, etc.) are often more practical in real-world applications: they learn directly from experience, without needing to know how the environment works under the hood.
