---
layout: post
title: "Why AI Helps More with Performance Optimization Than Feature Development"
date: 2026-07-15
categories: [architecture, ai, software-engineering]
tags: [ai, performance, feature-development, requirements, testing, architecture]
---

AI often produces better results for performance optimization than for feature development. The main reason is not that optimization is inherently easier to implement. It is that optimization usually has a clearer objective, faster feedback, and more stable correctness constraints.

## Performance Optimization Has a Measurable Target

An optimization task commonly starts with a concrete problem statement:

> Reduce product-page P99 latency from 1,200 ms to less than 300 ms without changing the returned data.

This gives AI a well-bounded search space. It can inspect traces, profiles, database queries, and benchmarks, then propose familiar improvements:

- parallelize independent I/O;
- remove duplicate queries;
- cache short-lived, repeatedly computed results;
- add or adjust database indexes;
- reduce unnecessary allocations, serialization, or network calls;
- replace an inefficient algorithm or data structure.

The result is easy to validate. The system either meets the target or it does not.

| Metric | Before | After |
|---|---:|---:|
| P50 latency | 250 ms | 90 ms |
| P99 latency | 1,200 ms | 240 ms |
| Database queries per request | 8 | 3 |
| CPU utilization | 75% | 48% |

In this situation, the desired behavior is already known. The optimization constraint can be expressed as:

$$
	ext{preserve observable behavior} \land \text{improve a measurable metric}
$$

AI does not need to understand every product decision to find a slow query, identify repeated work, or generate a benchmark that confirms the improvement.

## Feature Development Requires Decisions That Code Cannot Reveal

Now consider a feature request for an e-commerce system:

> Add coupon support.

AI can quickly generate database tables, APIs, UI components, and basic redemption logic. However, the important rules are often absent from the codebase and the request itself:

- Is a coupon reserved when a user claims it or only when an order is created?
- Can a coupon be combined with a membership discount?
- Can platform, merchant, and new-user coupons be used together?
- Which coupon wins when several coupons are eligible?
- Is a coupon returned after cancellation, payment timeout, or refund?
- Must a used new-user coupon remain consumed after a refund?
- What happens when two devices claim the final available coupon concurrently?
- Do rule changes affect coupons that were already issued?
- Are audit records, fraud controls, rate limits, and operations tools required?

None of these questions has a universally correct answer. They are product, legal, financial, and operational decisions. If AI fills the gaps by choosing reasonable defaults, it may deliver code that works technically but violates the business policy.

For example, an implementation may automatically restore a coupon after every refund. That looks sensible, but a business may explicitly require that a redeemed new-user coupon is never restored. Both behaviors can be implemented correctly; only the product owner can decide which one is correct.

## The Difference Is the Feedback Loop

Performance work has an immediate, objective feedback loop:

1. Capture a baseline with a profiler or benchmark.
2. Change an implementation.
3. Run the same measurement.
4. Confirm that latency, throughput, CPU, memory, or query count improved.
5. Confirm that existing behavior did not regress.

Feature work has a broader and less objective loop:

1. Discover and define the intended business behavior.
2. Design interfaces, state transitions, data ownership, and failure handling.
3. Implement the feature across services, storage, UI, permissions, and observability.
4. Validate normal flows, edge cases, migration behavior, concurrency, and user experience.
5. Obtain stakeholder acceptance.

The difference can be summarized as:

$$
	ext{feature development} = \text{define correct behavior} \rightarrow \text{implement it} \rightarrow \text{validate it}
$$

For optimization, the definition of correct behavior is usually already in place. For a feature, discovering that behavior is often the most difficult part.

## Make Feature Requests More AI-Friendly

AI becomes much more effective for feature development when the implicit decisions are turned into explicit contracts. Before implementation, provide acceptance criteria, API contracts, state diagrams, and representative edge cases.

For the coupon example, a concise acceptance table could be enough to eliminate much of the ambiguity:

| Scenario | Expected result |
|---|---|
| A ¥100 order uses a ¥20 threshold coupon | The payable amount is reduced by ¥20. |
| An order is cancelled before fulfillment | The coupon is returned. |
| A used new-user coupon is refunded | The coupon is not returned. |
| A platform coupon and a merchant coupon are eligible | One of each may be used together. |
| Two requests claim the final coupon concurrently | Exactly one request succeeds. |
| Coupon value exceeds order value | The payable amount is ¥0; no cash balance remains. |

With these rules, AI no longer needs to infer the product policy. It can focus on the implementation details: transactions, idempotency, locking, APIs, tests, and observability.

## Practical Division of Responsibility

Use AI primarily for tasks with clear constraints and machine-verifiable feedback:

- profiling analysis and optimization candidates;
- benchmark and regression-test generation;
- hot-path refactoring;
- SQL, caching, batching, and serialization improvements;
- implementation of well-specified feature contracts;
- edge-case test generation from stated business rules.

Keep people accountable for decisions that require domain judgment:

- feature scope and priorities;
- business rules and exception policies;
- architectural trade-offs;
- security, compliance, and operational risk decisions;
- final acceptance criteria.

The key is not to avoid AI for feature development. Instead, convert feature intent into precise, testable constraints first. Once the expected behavior is as measurable as a performance target, AI can contribute with much greater reliability.
