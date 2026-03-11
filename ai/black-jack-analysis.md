# Understanding the Blackjack Value Function — Notes on Sutton & Barto Figure 5.1

> A reflection on Monte Carlo methods and what the blackjack value function is actually telling us.

---

## Background

While working through *Reinforcement Learning: An Introduction* (Sutton & Barto), Chapter 5 introduces Monte Carlo methods using the game of Blackjack as a canonical example. Figure 5.1 shows the estimated value functions learned by a Monte Carlo agent — one for states where the player holds a **usable ace**, and one where they do not.

The figure is deceptively simple but packed with insight. Three visual patterns immediately jump out, and I want to write down my understanding of each.

---

## The Three Observations

### 1. Why does the value function spike up at player sums of 20 and 21?

This is the most intuitive of the three. When a player's hand totals **20 or 21**, they are in an overwhelmingly strong position:

- A sum of **21** is essentially a guaranteed win unless the dealer also hits 21 — an unlikely coincidence.
- A sum of **20** means the dealer needs to land exactly 20 or 21 to avoid losing, which is statistically rare.

Crucially, at both these sums, the correct policy is to **stick** (stop drawing cards). The agent doesn't risk busting, and the high win probability translates directly into a high expected return. So the value landscape rises sharply in those last two rows — not by coincidence, but because the policy converges to a near-optimal decision at those states.

---

### 2. Why does the value function drop off at the leftmost column (dealer showing an Ace)?

My initial instinct was that this drop comes from a higher probability of draws — but that's not quite right. The real reason is more direct: **when the dealer is showing an Ace, the dealer is at their strongest**.

An Ace in the dealer's hand offers maximum flexibility:
- It counts as 11 by default, giving the dealer a strong starting position.
- If the dealer would bust, the Ace drops to 1, absorbing the damage.

This flexibility means the dealer has a much higher probability of assembling a strong final hand — often 17 through 21. From the player's perspective, all states in this column face a much stronger opponent, so the **expected return is systematically lower across all player sums**. The entire leftmost column sags relative to the rest of the surface.

---

### 3. Why are the frontmost values higher in the usable-ace diagrams (upper) than in the no-ace diagrams (lower)?

Having a **usable ace** (i.e., an ace currently counted as 11) is not just about having a high card — it provides structural protection against busting.

If the player draws a card that would normally cause a bust, the ace automatically drops from 11 to 1, effectively absorbing the penalty. This means the player can **afford to be more aggressive**, drawing into situations that would be fatal without the ace.

This flexibility increases the expected return of almost every state in the usable-ace surface. The value function sits noticeably higher across the board in the upper diagram compared to the lower one. The ace isn't just a strong card — it changes the *decision space* available to the player.

---

## Summary

| Visual Pattern | Root Cause |
|---|---|
| Spike at sums 20–21 | Player holds a dominant hand; stick policy yields near-certain win |
| Drop at dealer's Ace column | Dealer's Ace provides maximum flexibility, disadvantaging the player |
| Upper diagram higher throughout | Usable Ace acts as a safety net, raising expected return across all states |

---

## Takeaway

What I find compelling about this example is how much information is embedded in the shape of the value function. The surface isn't arbitrary — every ridge and valley is a consequence of the underlying game dynamics, the fixed dealer policy, and the structure of the state space.

Monte Carlo methods learn this surface purely from experience, with no model of the game's rules. Yet the learned values accurately reflect subtle strategic realities like the dealer's ace advantage. That, to me, is a clean demonstration of what value-based RL is doing at its core.

---

*Reference: Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.), Chapter 5.*
