---
title: "Where `li { display: list-item; unicode-bidi: isolate; }` Actually Comes From"
categories: [web]
tags: [css, html, user-agent-stylesheet, unicode-bidi, list-style]
---

Open any browser's DevTools, inspect an `<li>` element, and look at the "user agent stylesheet" rules. Chrome and Firefox both show something like:

```css
li {
  display: list-item;
  unicode-bidi: isolate;
  text-align: match-parent;
}
```

It's tempting to assume this is one rule the browser vendor wrote specifically for `li`. It isn't. It's the result of two *separate* rule blocks in the HTML Standard's rendering chapter landing on the same selector for unrelated reasons.

## The two source rules

The [HTML Standard's rendering section](https://html.spec.whatwg.org/multipage/rendering.html) defines the suggested default styles that user agents are expected to implement as their UA stylesheet. Section 15.3 ("Non-replaced elements") breaks these into topical subsections, and `li` is targeted by two of them independently.

**15.3.7 Lists** is where `list-item` comes from:

```css
dir, dd, dl, dt, menu, ol, ul { display: block; }
li { display: list-item; text-align: match-parent; }
```

This block only cares about making lists render as lists — block containers for `ul`/`ol`, and `list-item` display (which generates the marker box) for `li`.

**15.3.5 Bidirectional text** is where `unicode-bidi: isolate` comes from. This block has nothing to do with lists; it exists to give a large set of block-level and sectioning elements bidi isolation, so text direction in one element doesn't leak into a sibling:

```css
address, blockquote, center, div, figure, figcaption, footer, form, header, hr,
legend, listing, main, p, plaintext, pre, summary, xmp, article, aside,
:heading, hgroup, nav, search, section, table, caption, colgroup, col, thead,
tbody, tfoot, tr, td, th, dir, dd, dl, dt, menu, ol, ul, li, bdi, output,
[dir=ltr i], [dir=rtl i], [dir=auto i] {
  unicode-bidi: isolate;
}
```

`li` just happens to be one of the ~40 selectors in that list, alongside `dd`, `dl`, `dt`, `menu`, `ol`, `ul`, and plenty of elements that have nothing to do with lists at all (`table`, `blockquote`, `header`...).

## Why DevTools shows them merged

Both rules match `li` with the same specificity context in the UA stylesheet, so when a browser's inspector displays "the effective UA styles for this element," it coalesces every matching UA declaration into one flattened view — regardless of which original rule block each declaration came from. That's a DevTools presentation detail, not evidence that the spec (or the browser) defines a single combined `li` rule. The spec keeps them in entirely different subsections written for different purposes: one is about list rendering, the other is about bidirectional text isolation.

## Why it matters

If you ever override `unicode-bidi` on `li` (for example, while debugging RTL/LTR mixed content) and wonder why `display: list-item` didn't move along with it, or vice versa, the answer is that they're not coupled at all in the standard. Overriding one has zero effect on the other — they were never the same rule to begin with, just two independent UA-stylesheet declarations that happen to target the same tag.

## References

- [HTML Standard — 15 Rendering](https://html.spec.whatwg.org/multipage/rendering.html): Source of both rule blocks; section 15.3.7 "Lists" defines `li { display: list-item; text-align: match-parent; }`, and section 15.3.5 "Bidirectional text" defines the shared `unicode-bidi: isolate` rule that includes `li` among ~40 other selectors.
