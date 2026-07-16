---
title: How RAGFlow Renders SVG Icons from iconfont.js
categories: [ai, rag, ragflow, frontend]
tags: [ragflow, react, svg, iconfont, typescript]
---

RAGFlow's web application uses an icon-font build artifact, `web/public/iconfont.js`, to deliver its SVG icons. The application does not render those icons as font glyphs or separate image files. Instead, the generated script registers SVG `<symbol>` elements in the document, and React components reference the symbols with SVG's `<use>` element.

The rendering path is:

```text
web/public/iconfont.js
		-> SVG symbols with IDs such as icon-search
		-> IconFont renders <use xlinkHref="#icon-search" />
		-> SvgIcon supplies the requested icon name
```

This post explains the responsibilities of `iconfont.js`, `IconFont`, and `SvgIcon`, and why their naming convention must remain aligned.

## iconfont.js Registers SVG Symbols

`web/public/iconfont.js` is generated icon-font output. When it is loaded by the browser, it adds an SVG sprite to the page. The sprite contains reusable `<symbol>` definitions similar to the following:

```html
<svg aria-hidden="true">
	<symbol id="icon-search" viewBox="0 0 1024 1024">
		<path d="..." />
	</symbol>
	<symbol id="icon-settings" viewBox="0 0 1024 1024">
		<path d="..." />
	</symbol>
</svg>
```

The symbols are generally hidden because they are definitions, not visible instances. Each `id` becomes an addressable icon resource within the current document.

For example, the search icon is identified by:

```text
icon-search
```

The JavaScript file must be available to the browser before a component tries to reference these IDs. In a React application, it is typically loaded from the public assets through the application entry page or build configuration.

## IconFont Creates a Visible Icon Instance

`web/src/components/icon-font.tsx` contains the `IconFont` component. Its core operation uses an SVG `<use>` element:

```tsx
<use xlinkHref={`#icon-${name}`} />
```

If `name` is `"search"`, React produces:

```html
<use xlink:href="#icon-search"></use>
```

The leading `#` is important. It means "look up an element ID in this document." The browser finds `<symbol id="icon-search">` registered by `iconfont.js`, then renders that symbol's paths inside the component's enclosing `<svg>` element.

Conceptually, `IconFont` converts a semantic icon name into an SVG fragment reference:

```text
name = "search"
"#icon-" + name
		-> "#icon-search"
		-> <symbol id="icon-search">
```

The component can also own presentation concerns such as `width`, `height`, `className`, `style`, and inherited color. Because the symbol is rendered within the page's SVG tree, CSS can style the icon like other SVG content.

## SvgIcon Wraps IconFont

`web/src/components/svg-icon.tsx` uses `IconFont` rather than talking to the SVG sprite directly. It is the higher-level component used by application code.

An application component can express intent with a compact call such as:

```tsx
<SvgIcon name="search" />
```

`SvgIcon` forwards the icon name and any supported presentation properties to `IconFont`. In turn, `IconFont` constructs `#icon-search` and renders the `<use>` reference.

This layering keeps call sites simple and centralizes the low-level SVG behavior:

| Layer | Responsibility |
|---|---|
| `web/public/iconfont.js` | Defines and registers SVG symbols in the document |
| `web/src/components/icon-font.tsx` | Renders an SVG `<use>` reference to a symbol |
| `web/src/components/svg-icon.tsx` | Provides the application-facing icon component API |

## The Icon Name Is a Contract

The `name` passed to `SvgIcon` and `IconFont` must match the suffix of a symbol ID in `iconfont.js`.

For example:

| Component usage | Generated reference | Required symbol ID |
|---|---|---|
| `<SvgIcon name="search" />` | `#icon-search` | `icon-search` |
| `<SvgIcon name="settings" />` | `#icon-settings` | `icon-settings` |

If the requested name does not exist in the generated sprite, the browser cannot resolve the reference. The `<svg>` element may still occupy space, but it will appear empty.

Common causes are:

- Passing a name with the `icon-` prefix already included, which produces `#icon-icon-search`.
- Renaming an icon in the source icon set without regenerating `iconfont.js`.
- Loading the application without loading `iconfont.js`.
- Using a name whose case, hyphens, or spelling differs from the generated symbol ID.

The expected API is therefore the suffix only:

```tsx
// Correct: IconFont adds the "icon-" prefix.
<SvgIcon name="search" />

// Incorrect: resolves to #icon-icon-search.
<SvgIcon name="icon-search" />
```

## Why Use an SVG Sprite

Referencing symbols through `<use>` gives RAGFlow a single loaded icon collection without requesting an image file for each icon. It also keeps icons scalable and allows them to participate in CSS styling, including sizing and color inheritance.

The tradeoff is that the generated symbol names become part of a small but important integration contract. When adding or changing icons, update the icon source, regenerate `web/public/iconfont.js`, and use the resulting symbol suffix as the `name` passed to `SvgIcon`.

In short, `SvgIcon` is the convenient React interface, `IconFont` is the SVG reference renderer, and `iconfont.js` is the symbol registry that supplies the icon geometry.
