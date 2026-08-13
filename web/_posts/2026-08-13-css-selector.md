---
title: "A Jekyll + kramdown Bug That Explains How CSS Selectors Actually Work"
categories: [web]
tags: [css, kramdown, jekyll, selectors]
---

I wanted the date and title in my archive/category/tag lists to sit on one line, like:

```
08-13 · Some post title
```

The Markdown source for each list item looked like this:

```markdown
- <time datetime="2026-08-13">08-13</time> · [Some post title](/some/url)
```

The date and the title kept rendering on two separate lines no matter what CSS I threw at the `<p>` inside the `<li>`. Chasing that bug down turned into a useful refresher on how kramdown parses raw HTML, and on how CSS selectors are actually evaluated — which is worth writing up on its own.

## The wrong DOM assumption

My first fix targeted the paragraph:

```css
.prose li > p {
  display: flex;
  align-items: baseline;
  gap: 0.5em;
}
```

This assumes the rendered DOM looks like:

```html
<li>
  <p><time>08-13</time> · <a href="...">Some post title</a></p>
</li>
```

It doesn't. kramdown's [HTML Blocks](https://kramdown.gettalong.org/syntax.html#html-blocks) rule explains why: a line is treated as the start of a raw HTML block if it begins with a tag that is **not** on kramdown's span-level HTML tag list. That list is:

```
a abbr acronym b big bdo br button cite code del dfn em i img input
ins kbd label option q rb rbc rp rt rtc ruby samp select small span
strong sub sup textarea tt var
```

`time` is not in it. So a line starting with `<time>` is parsed as an HTML block, not as inline span content. kramdown's docs spell out what happens next: "If the HTML/XML tag content should be handled as raw HTML, then only HTML/XML tags are parsed from this point onwards... If there is text after an end tag, it will be parsed as if it appears on a separate line."

In other words, `<time>08-13</time>` is consumed as its own raw HTML block, and the remaining `· [Some post title](/some/url)` on the same source line is parsed as if it started a new line — which wraps it in its own paragraph. The actual DOM is:

```html
<li>
  <time datetime="2026-08-13">08-13</time>
  <p>· <a href="/some/url">Some post title</a></p>
</li>
```

`<time>` and `<p>` are **siblings**, not parent/child. A rule scoped to `.prose li > p` only ever touches the paragraph's own inline content (the "·" and the link) — it has no way to reach `<time>`, so the two nodes kept stacking as separate block-level elements.

## The fix: style the `<li>`, not the `<p>`

Since the only common ancestor of `<time>` and `<p>` is the `<li>` itself, the flex container has to be the `<li>`:

```css
.prose li:has(> time) {
  display: flex;
  align-items: baseline;
  gap: 0.5em;
  margin: 0.3em 0;
}

.prose li:has(> time) > p {
  margin: 0;
}

.prose li > time {
  flex: none;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
```

Three things are doing distinct jobs here:

- **`.prose li:has(> time)`** uses [`:has()`](https://developer.mozilla.org/en-US/docs/Web/CSS/:has) to select only the `<li>` elements whose direct child is a `<time>` — i.e., exactly the archive/category/tag list items, without touching ordinary bullet lists elsewhere in a post. Making the `<li>` itself `display: flex` lines up its two children, `<time>` and `<p>`, on one row.
- **`> p { margin: 0; }`** removes the paragraph's default margin. Browsers apply a block-level top/bottom margin to `<p>` by default; once the `<li>` is a flex container, that margin would otherwise push the flex items apart vertically instead of letting them share a baseline.
- **`li > time`** sets `flex: none` so the date keeps its natural width instead of shrinking (the flex default is `flex: 0 1 auto`, which allows shrinking), plus a muted color and `tabular-nums` so the day numbers align visually.

`:has()` reached [Baseline "widely available" support](https://developer.mozilla.org/en-US/docs/Web/CSS/:has#browser_compatibility) in Chrome/Edge 105, Firefox 121, and Safari 15.4, so it's safe to rely on in a 2026 static site without a fallback.

## The selector-parsing rule that actually matters here

The part of this debugging session worth remembering longer than the CSS itself: **a CSS selector always filters the rightmost (last) compound selector; everything before it is a condition on that element's ancestry, not a separate target.**

- `.prose li > p` selects `<p>` elements that are a direct child of an `<li>` that is a descendant of `.prose`. It does not select "`<li>` elements containing a `<p>`."
- `.prose li p` (all descendant combinators) selects `<p>` anywhere inside an `<li>` anywhere inside `.prose` — the loosest possible ancestry constraint.
- `.prose li:has(> time)` is the exception that proves the rule: `:has()` is specifically designed to let the *left-hand* compound (the `<li>`) be the one that's selected, based on a condition about its descendants. It's the only standard way to select an element based on its children rather than its ancestors.

A related trap: comma-separated selectors don't share context. `.prose li p, br` is two entirely independent selectors — `.prose li p` and a bare `br` that matches every `<br>` on the page, not just ones inside `.prose li`. To apply a rule to multiple tags under the same scope, repeat the full ancestor chain for each (`.prose li p, .prose li br`) or use `:is()`/`:where()` to factor it out (`.prose li :is(p, br)`).

## Takeaway

When a CSS rule silently does nothing, don't assume the selector syntax is wrong before checking the actual generated DOM — with any Markdown processor, an unfamiliar-looking source line can be parsed very differently than it visually appears. kramdown's span-level HTML tag whitelist is short and easy to overlook; anything not on it, placed at the start of a line, becomes a block-level raw HTML node instead of inline content.

## References

- [kramdown Syntax — HTML Blocks](https://kramdown.gettalong.org/syntax.html#html-blocks) — defines the span-level HTML tag whitelist and the raw-HTML-block parsing behavior that explains why `<time>` and the trailing text end up as sibling nodes instead of nested inside one paragraph.
- [MDN — `:has()` CSS pseudo-class](https://developer.mozilla.org/en-US/docs/Web/CSS/:has) — documents `:has()` syntax, the sibling/parent-selection use case applied in the fix, and current browser support.
- [MDN — `:is()` CSS pseudo-class](https://developer.mozilla.org/en-US/docs/Web/CSS/:is) — confirms that `:is()` takes on the specificity of its most specific argument, while `:where()` always has zero specificity, supporting the comparison used for factoring the comma-selector example.