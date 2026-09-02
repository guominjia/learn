---
title: "How to Analyze a Business: A Practical SaaS Framework"
date: 2026-09-02
tags: [Business Analysis, SaaS, Finance, Linear]
---

Business analysis is the process of explaining how a company creates value, captures
value, and converts that value into cash. It is broader than reading a profit-and-loss
statement and more disciplined than writing a product description.

A useful analysis answers five questions:

1. What problem does the company solve, and for whom?
2. How does it acquire and retain customers?
3. How does pricing turn usage into revenue?
4. What costs are required to deliver and grow the product?
5. Which facts are reported, and which numbers are only estimates?

Linear is a useful case study because it publishes meaningful operating information,
but not the complete financial detail that an investor would receive from a public
company. The method below keeps those two kinds of information separate.

## Start With the Company's Status

The first question is not “What is the revenue?” It is “What disclosure regime applies
to this company?”

A domestic public company in the United States files annual Form 10-K reports. The
SEC describes Form 10-K as a comprehensive overview of the business and financial
condition that includes audited financial statements. A private company does not
normally provide the public with the same standardized package.

That distinction changes the research method:

| Company type | Strongest starting point | Typical evidence |
| --- | --- | --- |
| Public company | Regulatory filings and earnings releases | Revenue, COGS, operating expenses, cash flow, debt, segments, and risks |
| Private company | Official announcements, pricing, product pages, and financing announcements | Selected operating metrics, packaging, customers, hiring, and fundraising |
| Early-stage company | Founder material, customer evidence, and careful third-party research | Product adoption, funding, pricing experiments, and qualitative signals |

The following checklist is useful when deciding whether a piece of business
information normally belongs in a public company's filing:

| Information | Usually present in a public-company filing? | Explanation |
| --- | --- | --- |
| Business model | Partly | The Business, Business Model, and Risk Factors sections may describe it, but it is not a core financial-statement line item. |
| Industry conditions | Partly | The company may describe its market, growth opportunities, and industry risks, but this is management narrative rather than a complete independent industry report. |
| Competitors | Partly | A filing may discuss competition and competitive pressure, but it may not list every competitor and may avoid naming specific companies. |
| Financing | Not necessarily in the main financial statements | Public companies usually disclose debt, stock issuance, repurchases, and capital structure. A startup's Seed, Series A, and Series B history usually comes from financing announcements, regulatory filings, or databases. |
| Team size | Sometimes | Public companies commonly disclose employee counts. Some also disclose geography, functions, or labor costs, but a complete team structure is not guaranteed. |
| Revenue | Yes | Revenue is a core income-statement item. Annual revenue, quarterly revenue, and revenue growth are commonly reported. |
| ARR | Not necessarily | ARR is a common non-GAAP SaaS metric. A company may disclose it voluntarily, but it is not a required standard accounting line item. |
| Operating costs / COGS | Usually | Public companies commonly report cost of revenue, cost of sales, or COGS. For software companies, cloud infrastructure and customer support may be included. |
| Gross profit / gross margin | Usually present or calculable | If revenue and operating costs are disclosed, gross profit and gross margin can be calculated. |
| Revenue breakdown | Sometimes | Revenue may be disclosed by geography, business segment, or product line, but not necessarily by Basic, Business, Enterprise, or AI plan. |
| Profit breakdown | Usually incomplete | Gross profit, operating profit, net income, and segment profit may be available, but product-level profit such as AI, seat, or Enterprise profit is rarely disclosed. |
| Operating expenses | Yes | These commonly include research and development, sales and marketing, and general and administrative expenses. |
| Net income | Yes | Net income is a core income-statement item, but it is not the same as cash flow or the founders' personal income. |

Linear should therefore not be analyzed as if it had published a public-company annual
report. Its official pages disclose selected metrics, while detailed revenue
composition, COGS, operating expenses, and product-level profitability remain
undisclosed in the sources reviewed here.

## Build an Evidence Ledger

Before calculating anything, classify every important statement. This prevents an
estimate from quietly becoming a “fact” during editing.

| Evidence class | Meaning | Example from Linear |
| --- | --- | --- |
| Company-reported fact | The company states it directly | More than $100M ARR, more than 40,000 paying companies, 177% NRR, and positive cash flow |
| Public product fact | Visible on an official product or pricing page | Free, Basic, Business, and custom Enterprise plans |
| Third-party estimate | A research provider compiles or estimates the number | Sacra's financing history and ARR compilation |
| Analytical inference | A conclusion drawn from several facts | Enterprise expansion is likely important to the revenue mix |
| Scenario estimate | A number created from explicit assumptions | $18M of illustrative COGS at 18% of a $100M revenue proxy |

Use language that matches the evidence. “Linear reported” is appropriate for the
company's growth announcement. “Sacra estimates” is appropriate for third-party
research. “This model assumes” is appropriate for a margin calculation.

## Analyze the Product and Industry

Describe the product in terms of the job it performs, not only its feature list. For
Linear, the core workflow is product development: teams organize issues, projects,
initiatives, cycles, code changes, requests, and agent activity in one system.

Then define the market narrowly enough to identify real alternatives. Linear sits at
the intersection of:

- engineering issue tracking;
- product and portfolio planning;
- work intake and customer feedback;
- AI-assisted execution and agent workflows.

Competition should be analyzed by the buyer's alternative, not by a random list of
similar applications. Jira represents an established work-management incumbent with
free, Standard, Premium, and Enterprise tiers, automation, AI features, and enterprise
controls. Plane represents a different alternative: it combines project management,
documentation, AI agents, and cloud, self-hosted, and air-gapped deployment options.

The useful question is not “Which product has more features?” It is “Why would a
customer switch, and what would make switching painful?” Relevant dimensions include
workflow quality, integrations, data control, security, implementation effort,
ecosystem, and the amount of organizational context stored in the product.

## Explain the Business Model

Linear's visible model is product-led subscription SaaS:

- the Free plan lowers the cost of trial and supports bottom-up adoption;
- Basic is listed at $10 per user per month;
- Business is listed at $16 per user per month and adds advanced workflow, analytics,
	integration, and AI-related features;
- Enterprise uses custom annual pricing and adds capabilities such as SAML, SCIM,
	granular administration, security controls, onboarding, priority support, and account
	management.

This structure suggests several growth loops:

1. A team adopts the product for one workflow.
2. More seats and teams are added as the workflow becomes valuable.
3. The customer upgrades for privacy, administration, security, support, or AI features.
4. More organizational context makes the product harder to replace.

These are business-model mechanisms, not proof of a particular financial result. The
pricing page does not disclose plan mix, discounts, contract values, or the percentage
of revenue attributable to AI.

## Separate ARR From Revenue

ARR is an annualized measure of recurring subscription run rate. It is not necessarily
the same as GAAP revenue for the period. It may exclude one-time services and it does
not, by itself, describe cash collections, deferred revenue, COGS, or net income.

Linear reported that it had passed $100M ARR, had more than 40,000 paying companies,
and had reached 177% net revenue retention. A simple division gives:

```text
$100M ARR / 40,000 paying companies = $2,500 ARR per company on average
```

This is an arithmetic average, not actual ARPA. A small number of enterprise accounts
may contribute much more than small teams, and the customer-count definition and cohort
details are not fully published.

NRR is also a cohort metric, not a profit metric. A high NRR can indicate expansion,
upgrades, and seat growth among existing customers, but it does not reveal acquisition
cost, gross margin, or cash conversion.

## Model Revenue Composition Carefully

When a private company does not publish segment revenue, begin with mechanisms rather
than invented percentages.

| Component | What can be said | What cannot be claimed from public pricing |
| --- | --- | --- |
| Basic subscriptions | Recurring per-seat revenue at a published list price | Total Basic revenue or active Basic seats |
| Business subscriptions | Higher-priced per-seat subscriptions with additional features | Business revenue or AI revenue percentage |
| Enterprise subscriptions | Custom contracts with advanced controls and support | Average contract value or Enterprise share |
| Expansion revenue | More seats, teams, upgrades, and broader adoption are plausible mechanisms | Exact contribution to ARR |
| AI-related value | AI can increase product value and support higher-tier adoption | Separate AI revenue or AI profit |

Do not infer revenue mix directly from price. A $10 plan and a $16 plan do not tell us
how many customers use each plan, what discounts were granted, or how Enterprise
contracts are priced.

## Map the Cost Structure

For a SaaS business, distinguish delivery costs from growth costs.

### COGS and gross profit

COGS may include cloud infrastructure, databases, storage, bandwidth, backups,
observability, third-party services, delivery-related support, and AI inference. Gross
profit is:

```text
Gross profit = Revenue - COGS
Gross margin = Gross profit / Revenue
```

AI makes the model more usage-sensitive. Inference cost depends on model choice, request
volume, context size, caching, and whether usage is included or metered. The product
page can establish that AI features exist; it cannot establish Linear's AI COGS.

### Operating expenses

Operating expenses generally include:

- research and development, product, design, and security;
- sales, marketing, customer success, and account management;
- legal, finance, recruiting, compliance, and administration;
- offices, travel, insurance, and other corporate expenses.

Linear's growth announcement says it had more than 30 open roles across engineering,
product, design, sales, support, and other functions. That indicates continued hiring,
but it does not disclose total headcount, compensation, stock-based compensation, or
operating expenses.

## Use an Explicit Profit Scenario

If the income statement is unavailable, a scenario can still make the economics
concrete. It must be labeled as a model, and every assumption must be visible.

The following example treats the reported $100M ARR as a revenue proxy. It is not a
reported Linear income statement.

| Item | Assumption | Illustrative amount |
| --- | ---: | ---: |
| Revenue proxy | $100M ARR | $100.0M |
| COGS | 18% of revenue | -$18.0M |
| Gross profit | Revenue minus COGS | $82.0M |
| Operating expenses | 58% of revenue | -$58.0M |
| Operating profit | Gross profit minus operating expenses | $24.0M |

The implied margins are 82% gross margin and 24% operating margin. They are outputs of
the assumptions, not facts about Linear. A sensitivity table makes the uncertainty more
honest:

| Scenario | COGS | Operating expenses | Illustrative operating profit |
| --- | ---: | ---: | ---: |
| Efficient | 12% ($12M) | 48% ($48M) | $40M |
| Base | 18% ($18M) | 58% ($58M) | $24M |
| Investment-heavy | 28% ($28M) | 75% ($75M) | -$3M |

Do not call $18M “Linear's COGS” or $24M “Linear's net profit.” The model does not
include taxes, interest, stock-based compensation, depreciation, one-time items, or
cash-flow timing. It estimates operating profit, not net income.

## Treat Financing as Context, Not Revenue

Sacra compiles the following primary financing history for Linear:

| Round | Amount |
| --- | ---: |
| Seed | $4.2M |
| Series A | $13M |
| Series B | $35M |
| Series C | $82M |
| Total compiled primary funding | $134.2M |

Linear separately announced a $99M tender offer at a $2.5B valuation in August 2026.
The company described the tender as liquidity for current and former employees. A
tender offer is a secondary transaction when existing shareholders sell shares; it is
not automatically a new primary capital raise and should not be added to revenue.

Financing helps explain how a company reached its current product and team scale. It
does not prove product-market fit, profitability, or the value of the latest valuation.

## What the Analysis Can Conclude

A disciplined conclusion is narrower than a confident spreadsheet. For Linear, the
available evidence supports these statements:

- the company sells a recurring subscription product with Free, per-seat, Business,
	and custom Enterprise packaging;
- the company reported more than $100M ARR, more than 40,000 paying companies, 177%
	NRR, and positive cash flow in its August 2026 growth update;
- the product competes with broad work-management platforms, developer tools, and
	AI-oriented alternatives;
- Sacra reports approximately $134.2M of primary funding;
- detailed revenue mix, COGS, operating expenses, team size, net income, and
	product-level profit are not established by the reviewed public sources.

The right final sentence is therefore not “Linear earns $24M in operating profit.” It
is: “Under an explicitly stated 18% COGS and 58% operating-expense scenario, a $100M
ARR revenue proxy would produce $24M of illustrative operating profit. Linear's actual
income statement is not publicly disclosed in the sources reviewed.”

That distinction is the core skill in business analysis: calculate when calculation is
useful, but preserve the boundary between evidence and inference.

## References

- [Investor.gov: Form 10-K](https://www.investor.gov/additional-resources/general-resources/glossary/form-10-k) — explains the public-company annual report, audited financial statements, and ongoing filing obligations.
- [Linear Pricing](https://linear.app/pricing) — published plan prices, packaging, Enterprise controls, AI features, and customer-count statement.
- [Linear About](https://linear.app/about) — Linear's founding year, product positioning, customer scale, and distributed-team description.
- [Linear: Sharing Linear's growth](https://linear.app/now/sharing-growth-with-the-people-building-linear) — company-reported ARR, paying companies, NRR, cash-flow position, tender offer, valuation, agent adoption, and open roles.
- [Sacra: Linear](https://sacra.com/c/linear/) — third-party ARR, revenue, and financing compilation.
- [Atlassian Jira Pricing](https://www.atlassian.com/software/jira/pricing) — Jira plans, automation, AI features, enterprise controls, and pricing context.
- [Plane](https://plane.so/) — Plane's project-management, AI, migration, self-hosting, and air-gapped product positioning.