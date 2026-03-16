# Is the Convergence of Monte Carlo ES Still an Open Problem?

> An investigation into one of the most fundamental open theoretical questions in reinforcement learning — the convergence of Monte Carlo Exploring Starts to the optimal fixed point.

---

## 1. The Original Question

In Chapter 5 of *Reinforcement Learning: An Introduction*, Sutton & Barto describe the **Monte Carlo Exploring Starts (MCES)** algorithm and make the following remark:

> *"Convergence to this optimal fixed point seems inevitable as the changes to the action-value function decrease over time, but has not yet been formally proved. In our opinion, this is one of the most fundamental open theoretical questions in reinforcement learning."*

The argument for why MCES *should* converge is intuitive:

1. All returns for each state–action pair are accumulated and averaged, **irrespective of what policy was in force** when they were observed.
2. If MCES converged to a **suboptimal** policy, the value function would eventually converge to the value function for that policy.
3. That would cause the policy to **change** (because greedy improvement over a suboptimal value function yields a different policy).
4. Therefore, **stability is achieved only when both the policy and the value function are optimal**.

The logic is compelling — yet turning it into a formal proof has proven remarkably difficult.

---

## 2. Why Is This So Hard to Prove?

The core difficulty lies in the structure of the algorithm itself:

| Challenge | Explanation |
|-----------|-------------|
| **Coupled dynamics** | The policy and value function change simultaneously, forming a coupled dynamical system |
| **Incomplete evaluation** | Each iteration uses only a single episode to update — policy evaluation is never "finished" before improvement happens |
| **Non-uniform update rates** | Different state–action pairs are visited at different frequencies, so their estimates converge at different rates |
| **Exploring Starts assumption** | While it guarantees coverage, it doesn't provide the uniform sampling rates needed by standard stochastic approximation theory |

Standard proof techniques — contraction mappings, stochastic approximation, ODE methods — all struggle with this combination of challenges.

---

## 3. Key Papers and Partial Results

The problem has attracted significant research attention. Here is a timeline of key results:

### 3.1 Tsitsiklis (2002) — Modified MCES Converges

- **Paper:** *"On the Convergence of Optimistic Policy Iteration,"* JMLR, 3:59–72, 2002
- **Result:** Proved convergence of a **modified** version of MCES where:
  - Q-function estimates are updated at the **same rate** for all state–action pairs
  - Discount factor $\gamma < 1$
- **Limitation:** Requires modification to the original algorithm — the equal update rate assumption is not realistic for standard MCES

### 3.2 Jun Liu (2021) — Undiscounted Case

- **Paper:** *"On the Convergence of Reinforcement Learning with Monte Carlo Exploring Starts,"* Automatica, 2021
- **Result:** Investigated convergence for the **stochastic shortest path problem** (undiscounted costs)
- **Significance:** Complements Tsitsiklis (2002) by settling the problem for a specific undiscounted setting

### 3.3 Wang, Yuan, Shao & Ross (2022) — Feed-Forward MDPs

- **Paper:** *"On the Convergence of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning,"* ICLR 2022
- **Result:** Proved **almost sure convergence** of the **original, unmodified** MCES algorithm for **Optimal Policy Feed-Forward MDPs** — MDPs where states are not revisited within any episode under an optimal policy
- **Key insight:** Uses a novel inductive proof technique based only on the **strong law of large numbers**, avoiding stochastic approximation entirely
- **Coverage:** Includes all deterministic environments and many episodic environments with monotonically changing state features
- **Limitation:** Does not cover **all** finite MDPs

### 3.4 Winnicki & Srikant (2023) — Lookahead Policy Improvement

- **Paper:** *"On the Convergence of Policy Iteration-Based Reinforcement Learning with Monte Carlo Policy Evaluation,"* AISTATS 2023
- **Result:** Proved convergence when the policy improvement step uses **lookahead** rather than simple greedy improvement
- **Limitation:** Requires a **modified** policy improvement step, so does not resolve the original open problem

### 3.5 Winnicki & Srikant (2024) — TD-Learning Variant

- **Paper:** *"Convergence of Monte Carlo Exploring Starts with TD-Learning,"* IEEE CDC 2024
- **Result:** Studied convergence of MCES when **TD($\lambda$)** is used for policy evaluation combined with lookahead for policy improvement

### 3.6 Chen, Ross & Youssef (2024) — Finite-Sample Analysis

- **Paper:** *"Finite-Sample Analysis of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning,"* arXiv:2410.02994, 2024
- **Result:** Goes beyond convergence to analyze the **convergence rate** of a modified MCES, proving sample complexity of $\tilde{O}(SAK^3 \log^3(1/\delta))$ episodes for the stochastic shortest path problem

---

## 4. Current Status at a Glance

| Variant | Status | Reference |
|---------|--------|-----------|
| **Unmodified MCES + Arbitrary finite MDPs** | **Still Open** | Sutton & Barto (2018) |
| Modified MCES (equal update rates, $\gamma < 1$) | Proved | Tsitsiklis (2002) |
| Unmodified MCES + Feed-Forward MDPs | Proved | Wang et al. (ICLR 2022) |
| MCES + Lookahead policy improvement | Proved | Winnicki & Srikant (AISTATS 2023) |
| MCES + TD($\lambda$) + Lookahead | Proved | Winnicki & Srikant (CDC 2024) |
| Undiscounted / Stochastic shortest path | Proved | Liu (Automatica 2021) |
| Finite-sample bounds (modified MCES) | Proved | Chen et al. (2024) |

---

## 5. Why It Matters

This isn't just a theoretical curiosity. The MCES convergence question sits at the heart of **Generalized Policy Iteration (GPI)** — the fundamental paradigm underlying most RL algorithms. GPI interleaves incomplete policy evaluation with policy improvement, and proving that this process converges in the simplest tabular setting (Monte Carlo + greedy improvement) would provide foundational assurance for the entire family of RL methods.

The fact that MCES **always converges in practice** despite lacking a proof makes this a fascinating gap between theory and practice — reminiscent of how the simplex method in linear programming worked reliably for decades before smoothed analysis explained its behavior.

---

## 6. Conclusion

As of 2026, the original question posed by Sutton & Barto — whether the **unmodified** Monte Carlo Exploring Starts algorithm with greedy policy improvement converges to the optimal policy for **all finite MDPs** — **remains an open problem**. However, the research community has made substantial progress:

- Convergence has been proved for important subclasses (feed-forward MDPs, stochastic shortest paths)
- Modified versions of the algorithm have been shown to converge
- Finite-sample complexity bounds have been established for variants

The problem is now much better understood than when it was first posed, but the general proof continues to elude us — making it one of the most enduring open questions in reinforcement learning theory.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Tsitsiklis, J. N. (2002). On the Convergence of Optimistic Policy Iteration. *JMLR*, 3, 59–72.
3. Wang, J., Yuan, B., Shao, H., & Ross, S. (2022). On the Convergence of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning. *ICLR 2022*.
4. Liu, J. (2021). On the Convergence of Reinforcement Learning with Monte Carlo Exploring Starts. *Automatica*.
5. Winnicki, A., & Srikant, R. (2023). On the Convergence of Policy Iteration-Based Reinforcement Learning with Monte Carlo Policy Evaluation. *AISTATS 2023*.
6. Winnicki, A., & Srikant, R. (2024). Convergence of Monte Carlo Exploring Starts with TD-Learning. *IEEE CDC 2024*.
7. Chen, W., Ross, S., & Youssef, A. (2024). Finite-Sample Analysis of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning. *arXiv:2410.02994*.