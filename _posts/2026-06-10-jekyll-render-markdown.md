---
title: Why Jekyll Renders HTML as Code Blocks (and How to Fix It)
categories: [jekyll, markdown]
tags: [liquid, kramdown, troubleshooting]
---

When writing Jekyll pages with Liquid and HTML, you may see raw tags such as `<section>` rendered as a code block instead of normal HTML.

This issue is common in category or tag archive pages where loop output is mixed with Markdown content.

## Symptom

- The page shows literal HTML like `<section class="card archive-group">`.
- Sidebar anchor links do not jump to the expected section.
- The page looks like a large `<pre><code>` block.

## Root Cause

In Markdown (Kramdown), block content indented by 4+ spaces can be interpreted as a code block.

When Liquid loops output HTML with leading indentation, the final rendered content can accidentally enter code-block context.

That means your intended DOM nodes are not created, so anchors (for example `#history` or `#jekyll`) cannot find target IDs.

## Fix

1. Keep loop-generated block HTML left-aligned (no extra leading spaces before `<section>`, `<ul>`, `<li>`, etc.).
2. Add explicit IDs to each category/tag section.
3. Make sidebar links point to those IDs.

Example:

```liquid
{% for tag in site.tags %}
{% assign tag_name = tag[0] %}
<section id="{{ tag_name | slugify }}" class="card archive-group">
	<h2>#{{ tag_name }}</h2>
</section>
{% endfor %}
```

## Best Practices for Jekyll Blog Pages

- Prefer Liquid-driven HTML sections for archive pages.
- Avoid mixing deeply indented HTML blocks inside Markdown list contexts.
- Use `slugify` for stable anchor IDs.
- Validate generated HTML by checking browser dev tools.

## Takeaway

If Jekyll pages unexpectedly render as code blocks, check indentation first.

In most cases, moving Liquid-generated HTML to top-level alignment resolves both rendering and navigation issues immediately.