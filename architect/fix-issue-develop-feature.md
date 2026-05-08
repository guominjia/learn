# Fact-Finding, Fix Issue, and Develop Feature: Three Types of Engineering Tasks

When working on software projects, not every task is the same.
Understanding the **type of task** you're facing shapes how you approach it,
what tools you reach for, and how you measure "done."

This post breaks engineering work into three categories
and explains why the distinction matters —
especially in the age of AI-assisted development.

---

## 1. Fact-Finding Questions

These are the simplest category.
The answer already exists somewhere — in a spec, a wiki, or the codebase —
you just need to **find it**.

### Examples

- "What is this component?"
- "What does this register do?"
- "Who uses this interface and how?"

### Characteristics

| Property | Description |
|----------|-------------|
| Answer source | Already exists in documentation / code |
| Method | Search → Extract → Present |
| Complexity | Low to moderate |
| Output | Information (text) |

Some fact-finding questions are **single-hop** ("What is X?"),
while others are **multi-hop**
("What is X, who uses it, and how does the data flow?"),
requiring aggregation from multiple sources.

A well-built **RAG (Retrieval-Augmented Generation)** system
handles these questions effectively:
retrieve relevant chunks, extract the answer, and present it.

---

## 2. Fix Issue (Diagnostic & Remediation)

This is where things get interesting.
You start with a **symptom** — an error, a crash, a test failure —
and your job is to trace it back to a **root cause** and then fix it.

### The Mindset: Detective Work

Fixing an issue is fundamentally a **convergent** process.
There is (usually) one root cause.
Your job is to narrow down from all possible causes to the actual one.

### The Process

```
Step 1: Understand the problem
        → What exactly is failing? What are the symptoms (error message or log, how to reproduce, BIOS or OS)?
        → What is the expected behavior?

Step 2: Locate the code
        → Search for relevant modules
        → Read the error-related functions
        → Trace the call chain

Step 3: Root cause analysis (the core step)
        → From symptom (error message or log) → throw site (which code throw it) → input (what's input) → caller (who call it)
        → Compare with working state: what changed?
        → Narrow down: config? data? logic?

Step 4: Fix (minimal change)
        → Change the smallest scope possible
        → Don't introduce new problems

Step 5: Verify (closed loop)
        → Does it compile? → Can the bug still be reproduced?
        → Do related tests pass? → Any side effects?
```

### Why This Is Not a Fact-Finding Problem

The root cause is **not written anywhere**.
No document says "the bug is on line 42 of handler.go."
You have to **reason** your way there.

A typical "Fix issue" task embeds multiple sub-problems:

| Sub-problem | Type |
|-------------|------|
| What does the error log say? | Information extraction |
| Which code handles this? | Code search |
| Why does it return an error here? | **Reasoning** (not retrievable) |
| How did it work before? | Git history lookup |
| What change will fix it? | Solution generation |
| Did I break anything else? | Verification |

---

## 3. Develop Feature (Generative / Creative Task)

Feature development starts with a **requirement** and ends with **working code**.
The answer doesn't exist yet — you have to **create** it.

### The Mindset: Architecture & Construction

Unlike fixing a bug (convergent), feature development is **divergent** —
there are many possible designs, and you choose one.

### The Process

```
Step 1: Understand the requirement
        → What exactly should be built (doc/spec/wiki, add/modify)?
        → What are the acceptance criteria?
        → If unclear → ASK, don't guess

Step 2: Gather context
        → Search existing code for related implementations
        → Understand the architecture and dependencies
        → Find reusable patterns

Step 3: Design the plan
        → Break into trackable sub-tasks
        → Identify dependencies between tasks
        → Define interfaces / data structures first

Step 4: Implement incrementally
        → One sub-task at a time
        → Verify after each step (compile / test / lint)
        → When facing design uncertainty → ask, don't assume

Step 5: Verify end-to-end
        → All compilation passes?
        → Tests cover the new functionality?
        → Meets the original acceptance criteria?
```

### Sub-tasks Inside a Feature

A single "develop feature" task nests many sub-problems of different types:

```
Develop Feature X
  ├── What is Feature X?              ← fact-finding
  ├── What exists in the codebase?    ← code search
  ├── How should it be designed?      ← creative / architectural
  ├── Implement module A              ← code generation
  ├── Implement module B              ← code generation
  ├── Integrate with existing code    ← code modification
  └── Does it work end-to-end?        ← verification
```

---

## Side-by-Side Comparison

| Dimension | Fact-Finding | Fix Issue | Develop Feature |
|-----------|-------------|-----------|-----------------|
| **Starting point** | A question | A symptom / failure | A requirement |
| **Direction** | Retrieve | Converge (narrow down) | Diverge (design choices) |
| **Core skill** | Search + Extract | Diagnosis + Reasoning | Design + Generation |
| **Output** | Information (text) | Code change (action) | New code (action) |
| **Answer exists?** | Yes, in docs/code | No, must be deduced | No, must be created |
| **Analogy** | Librarian | Detective | Architect + Builder |

---

## Why This Matters for AI-Assisted Development

Understanding these categories helps you use AI tools effectively:

### RAG Systems (Search + Retrieval)
Great for **fact-finding**. They retrieve existing information from docs and code.
But they cannot diagnose bugs or design features.

### AI Coding Agents (Agentic Workflows)
Needed for **fix issue** and **develop feature**.
They combine multiple capabilities in a loop:

- Code understanding
- Multi-file search
- Reasoning and planning
- Code generation
- Tool invocation (compile, test, run)
- Iterative verification

The key insight:
**every complex task (fix / feature) contains fact-finding sub-problems inside it.**
RAG is a component of the agent, not a replacement.

```
┌─────────────────────────────┐
│     Develop Feature         │  ← Agent workflow
│  ┌───────────────────────┐  │
│  │  What is this API?    │  │  ← RAG / fact-finding
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Design the solution  │  │  ← Reasoning
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Write the code       │  │  ← Code generation
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Verify it works      │  │  ← Tool use
│  └───────────────────────┘  │
└─────────────────────────────┘
```

---

## Takeaway

Next time you pick up a task, ask yourself:

1. **Am I looking for information that already exists?** → Fact-finding. Search for it.
2. **Am I diagnosing something broken?** → Fix issue. Trace from symptom to root cause.
3. **Am I building something new?** → Develop feature. Plan, design, implement, verify.

Knowing which game you're playing determines which tools and thinking modes you reach for.
