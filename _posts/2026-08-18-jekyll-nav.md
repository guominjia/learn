---
title: "\"is-active\" and \"aria-current\" Are Build-Time Output, Not Runtime State"
categories: [jekyll]
tags: [liquid, nav, aria, static-site]
---

Looking at a rendered nav link, it's tempting to assume some JavaScript toggles the current item on click:

```html
<a href="..." class="is-active" aria-current="page">...</a>
```

There is no JavaScript involved. `class="is-active"` and `aria-current="page"` are not attributes of the `<nav>` element or a runtime state — they belong to the current item's `<a>` tag, and Jekyll decides whether to emit them **at build time**, via Liquid.

## The template logic

```liquid
{% assign is_active = false %}
{% if item.url == '/' and page.url == '/' %}
  {% assign is_active = true %}
{% elsif item.url != '/' and page.url contains item.url %}
  {% assign is_active = true %}
{% endif %}
```

When the current page's `page.url` matches a nav entry's `item.url`, the template emits:

- `class="is-active"` — a CSS hook for styling the current menu item.
- `aria-current="page"` — ARIA semantics telling assistive tech "this is the page you're on."

If it doesn't match, neither attribute is written to that `<a>`. This isn't an interactive state — it's already baked into the final HTML when the site is generated.

## Why nothing "switches" after a click

For a static site, a click is just a normal page navigation:

1. The user clicks a nav link, e.g. one pointing to `/docs/`.
2. The browser requests that URL, which corresponds to a separate HTML file Jekyll already generated in advance.
3. While building `/docs/`, `page.url` is `/docs/`, so the Liquid condition emits:
   ```html
   <a href="/docs/" class="is-active" aria-current="page">Docs</a>
   ```
4. The browser loads this new HTML file, so visually the active nav item appears to have changed.

The HTML before and after the click is two entirely different files. There's no JavaScript, and nothing mutates the DOM.

For example, the home page's generated nav might look like:

```html
<a href="/" class="is-active" aria-current="page">Home</a>
<a href="/docs/">Docs</a>
```

While the `/docs/` page's generated nav looks like:

```html
<a href="/">Home</a>
<a href="/docs/" class="is-active" aria-current="page">Docs</a>
```

`class="is-active"` only gives CSS a styling hook; `aria-current="page"` only carries accessibility semantics. Both are written into every page's HTML **at Jekyll build time**.

## Why this isn't wasteful

The key point: clicking a nav link was always going to change the article content, so the browser needs a new HTML document regardless. Fetching a fresh static page isn't an extra request incurred just to flip `is-active` — that attribute is just a few extra bytes riding along with the HTML you needed anyway.

| Approach | What a click fetches | Nav state source |
|---|---|---|
| Static Jekyll | New page HTML, plus any images/assets | Written into HTML at build time |
| JS SPA | API/JSON data, then JS executes and updates the DOM | Updated at runtime by JS |

Adding JS purely to highlight the active menu item is usually a net loss:

- Native browser navigation and caching are mature; HTML renders directly, so first paint is fast.
- No JS dependency — search engines, RSS readers, accessibility tools, and JS-disabled environments still work.
- Every page is a full, addressable URL — bookmarkable, shareable, refreshable, cacheable offline.
- Jekyll's generated HTML is easy for a CDN to cache; shared CSS/JS is typically downloaded once.
- An SPA still has to fetch the new article's data, plus pay for JS bundle download, parsing, and DOM updates.

## When JS is actually worth it

A JS-driven approach makes sense when you don't want a full page reload on navigation, need to preserve player/editor state across views, want instant client-side filtering or search, need infinite scroll, or the page itself is a highly interactive application.

For a blog like this one, letting static HTML decide `aria-current` is the right call — the page navigation was happening anyway, and Liquid just makes sure each generated page carries the correct nav semantics. If chapter-by-chapter navigation without a reload becomes a real need, that's when partial JS loading and the History API would actually pay for themselves.
