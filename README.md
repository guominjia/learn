# Learn

Learning notes repository + Jekyll knowledge website.

- Website: https://learn.biosrag.com
- Source: markdown-first notes, organized by domain
- Goal: document practice, retain reusable knowledge, and keep topics searchable

## Site Navigation

- Home: [/](./)
- About: [about.md](about.md)
- Archives (by date): [archives.md](archives.md)
- Categories (navigation + auto archive): [categories.md](categories.md)
- Tags (auto-generated): [tags.md](tags.md)

## Writing Guide

### 1) Add a post

Create a file under `_posts/` with format `YYYY-MM-DD-title.md`:

```yaml
---
title: Your Post Title
categories: [topic]
tags: [tag1, tag2]
---
```

`categories` and `tags` will be automatically aggregated in `categories.md` and `tags.md`.

### 2) Add a page

Create a normal markdown file in root or any folder with front matter:

```yaml
---
layout: page
title: Page Title
permalink: /your-page/
---
```

### 3) Local preview (optional)

```bash
bundle exec jekyll serve
```

## Notes

- This repository is Jekyll-first; homepage and archives are rendered from markdown/pages.
- Directory `README.md` files are kept as lightweight entry docs, and navigation is unified in [categories.md](categories.md).

## Useful Links

- [iconfont](https://www.iconfont.cn/)
- [Material Symbols](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded)
- [Supabase](https://www.supabase.com/)
- [Jina Embeddings](https://jina.ai/embeddings/)
- [HIWEPY Wiki](https://wiki.hiwepy.com/)
- [Emergent Mind](https://www.emergentmind.com/)