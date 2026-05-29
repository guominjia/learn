# Documentation

## [PlantUML](https://github.com/plantuml/plantuml-server)

## [Mermaid](mermaid/README.md)
Mermaid diagram syntax, flowcharts, and text-based visualization techniques

## [ExcaliDraw](https://excalidraw.com/)

## Top 10 Documentation Site Generators

| # | Tool | Language | Best For | License |
|---|------|----------|----------|---------|
| 1 | [ReadTheDocs / Sphinx](https://about.readthedocs.com/) | Python | Technical / API docs | Open Source |
| 2 | [Docusaurus](https://docusaurus.io/) | React | Open source project docs | Open Source |
| 3 | [MkDocs / Material](https://www.mkdocs.org/) | Python | Markdown-based project docs | Open Source |
| 4 | [VitePress](https://vitepress.dev/) | Vue | Lightweight static docs | Open Source |
| 5 | [GitBook](https://www.gitbook.com/) | SaaS | Team knowledge bases | Freemium |
| 6 | [Mintlify](https://mintlify.com/) | SaaS | Beautiful API docs | Freemium |
| 7 | [Doxygen](https://www.doxygen.nl/) | C/C++ | Source code reference docs | Open Source |
| 8 | [Swagger / Redoc](https://swagger.io/) | Any | REST API documentation | Open Source |
| 9 | [Confluence](https://www.atlassian.com/software/confluence) | SaaS | Enterprise wiki | Paid |
| 10 | [Notion](https://www.notion.so/) | SaaS | General-purpose docs & wiki | Freemium |

### Pros & Cons

#### 1. [ReadTheDocs / Sphinx](https://about.readthedocs.com/)
- **Pros**: Free hosting for open-source; reStructuredText + Markdown; auto-build from Git; versioned docs; PDF/ePub export; mature ecosystem with rich extensions
- **Cons**: Steep learning curve (rST syntax); theming is limited compared to modern tools; build can be slow for large projects

#### 2. [Docusaurus](https://docusaurus.io/)
- **Pros**: Built by Meta; React-based with MDX support; versioning, i18n, search built-in; excellent plugin ecosystem; modern UI
- **Cons**: Requires Node.js; heavier build compared to static-only tools; React knowledge needed for deep customization

#### 3. [MkDocs / Material for MkDocs](https://www.mkdocs.org/)
- **Pros**: Pure Markdown; extremely simple setup; [Material theme](https://squidfunk.github.io/mkdocs-material/) is beautiful & feature-rich (search, tabs, admonitions); fast builds; great for internal docs
- **Cons**: Less flexible than Sphinx for complex docs; Material theme's advanced features require paid sponsorship (Insiders edition)

#### 4. [VitePress](https://vitepress.dev/)
- **Pros**: Vite-powered, extremely fast; Vue 3 components in Markdown; lightweight; great DX; ideal for library/framework docs
- **Cons**: Younger ecosystem; fewer plugins than Docusaurus; Vue knowledge needed for advanced customization

#### 5. [GitHub Pages](https://pages.github.com/)
- **Pros**: Free for public repos; deeply integrated with GitHub; supports Jekyll (built-in) or any static site generator; custom domains; HTTPS; CI/CD via GitHub Actions; massive community; ideal for open-source project docs
- **Cons**: Only static sites (no server-side); Jekyll build can be slow; limited to public repos on free tier; no built-in search (needs Algolia etc.); no WYSIWYG editor; 1GB site size limit

#### 6. [GitBook](https://www.gitbook.com/)
- **Pros**: Beautiful WYSIWYG editor; Git sync; team collaboration; hosted solution with zero maintenance; great for non-technical contributors
- **Cons**: Free tier is limited; less customizable than self-hosted tools; vendor lock-in; export options are restricted

#### 7. [Mintlify](https://mintlify.com/)
- **Pros**: Gorgeous default theme; built-in API playground; AI-powered search; fast setup with MDX; excellent for developer-facing products
- **Cons**: Closed-source SaaS; pricing scales with usage; less community ecosystem; limited self-hosting options

#### 8. [Doxygen](https://www.doxygen.nl/)
- **Pros**: De facto standard for C/C++/Java source code docs; auto-generates from code comments; supports call graphs via Graphviz; cross-referencing; multi-language support
- **Cons**: Outdated UI; complex configuration (Doxyfile); output looks dated without custom CSS; overkill for non-code documentation

#### 9. [Docsify](https://docsify.js.org/)
- **Pros**: No build step — renders Markdown at runtime; extremely simple setup (single `index.html`); lightweight; plugin system; full-text search built-in; GitHub Pages friendly; great for small-to-medium docs
- **Cons**: Client-side rendering hurts SEO; not suitable for very large doc sites; slower initial load than pre-built static sites; limited theming compared to Docusaurus/VitePress

#### 10. [Swagger (OpenAPI) / Redoc](https://swagger.io/)
- **Pros**: Industry standard for REST API docs; interactive "Try It" console; auto-generated from OpenAPI spec; [Redoc](https://github.com/Redocly/redoc) provides clean three-panel layout
- **Cons**: Only for API docs (not general documentation); spec file maintenance can be tedious; limited narrative/guide content support

#### 11. [Confluence](https://www.atlassian.com/software/confluence)
- **Pros**: Deep Jira/Atlassian integration; rich editor; permissions & spaces; enterprise-grade; templates; widely adopted in corporations
- **Cons**: Expensive at scale; performance can be slow; Markdown support is weak; pages become disorganized without governance; vendor lock-in

#### 12. [Notion](https://www.notion.so/)
- **Pros**: All-in-one workspace (docs, databases, kanban); drag-and-drop block editor; real-time collaboration; public page sharing; flexible structure
- **Cons**: Not purpose-built for technical docs; no native versioning for docs; SEO is limited for public pages; export format is lossy; offline support is weak

### Comparison Matrix

| Feature | ReadTheDocs | Docusaurus | MkDocs | VitePress | GitHub Pages | GitBook | Mintlify | Doxygen | Docsify | Swagger | Confluence | Notion |
|---------|:-----------:|:----------:|:------:|:---------:|:------------:|:-------:|:--------:|:-------:|:-------:|:-------:|:----------:|:------:|
| Markdown | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ⚠️ | ✅ |
| Self-hosted | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Versioning | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Search | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| API Playground | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Free/OSS | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ✅ | ❌ | ⚠️ |
| i18n | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ✅ | ❌ |
| WYSIWYG | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |

> ✅ = Full support, ⚠️ = Partial / Plugin needed, ❌ = Not supported

### Recommendation

- **Open source project docs** → Docusaurus or MkDocs Material
- **API documentation** → Mintlify or Swagger/Redoc
- **C/C++ source code docs** → Doxygen
- **Python project docs** → ReadTheDocs + Sphinx
- **Lightweight & fast** → VitePress
- **Free hosting from GitHub** → GitHub Pages + Jekyll/Docusaurus/VitePress
- **Zero-build simplicity** → Docsify
- **Non-technical team collaboration** → GitBook or Notion
- **Enterprise internal wiki** → Confluence

## Sphinx Themes
- <https://www.sphinx-doc.org/en/master/usage/theming.html>
    - <https://sphinx-themes.org/>
    - <https://pypi.org/search/?q=&o=&c=Framework+%3A%3A+Sphinx+%3A%3A+Theme>
    - <https://github.com/search?utf8=%E2%9C%93&q=sphinx+theme>
    - <https://gitlab.com/explore?name=sphinx+theme>
- <https://docs.sunpy.org/projects/sunpy-sphinx-theme/latest/>
- <https://docs.python.org/3/library/>
- <https://xinetzone.github.io/DaoField/>

## [Algolia](https://www.algolia.com/)
Focus on document search, provide fast, full search function, integrated with static doc site (for example ReadTheDocs, Docusaurus, VitePress)

## [Markdownify](https://github.com/matthewwithanm/python-markdownify)
Convert HTML to Markdown
```python
markdown = markdownify(
            body_html,
            heading_style="ATX",
            bullets="-",
            strip=['script', 'style'],
            escape_asterisks=False,
            escape_underscores=False,
        )
```

## [MarkItDown](https://github.com/microsoft/markitdown)
Covert any file to Makrdown

1. <https://deepwiki.com/search/markitdownconverttextcontentma_84769b62-fb26-4696-8888-e9b2217020cc>
2. <https://deepwiki.com/microsoft/markitdown>
3. <https://github.com/microsoft/markitdown>

## [MdownDown-It](https://github.com/executablebooks/markdown-it-py)
Convert Markdown to HTML

## [mistune](https://github.com/lepture/mistune)
Python fast Markdown parser

## [mammoth](https://github.com/mwilliamson/python-mammoth)
Convert `.docx` files to HTML or Markdown, focusing on semantic content structure while ignoring complex formatting styles.

## [ScreenToGif](https://www.screentogif.com/)
**ScreenToGif** is a free, open-source screen recording tool for Windows that allows you to:
- Record your **screen**, **webcam**, or **sketchboard**
- Edit frames directly in the built-in editor
- Export as **GIF**, **video** (MP4, AVI), **WebP**, or **images**

### Similar Tools

| Tool | Platform | Key Feature |
|------|----------|-------------|
| [LICEcap](https://www.cockos.com/licecap/) | Windows/macOS | Lightweight, simple GIF capture |
| [Peek](https://github.com/phw/peek) | Linux | Simple animated GIF recorder |
| [Kap](https://getkap.co/) | macOS | Clean UI, exports GIF/MP4/WebM |
| [ShareX](https://getsharex.com/) | Windows | Full-featured, screenshots + GIF + video |
| [Gyroflow](https://gyroflow.xyz/) | Cross-platform | Advanced video stabilization |
| [OBS Studio](https://obsproject.com/) | Cross-platform | Professional screen/video recording |
| [Recordit](https://recordit.co/) | Windows/macOS | Quick GIF sharing via cloud |
| [Gifox](https://gifox.app/) | macOS | Menubar GIF recorder |

### Recommendation

- **Windows**: ShareX (most powerful free option) or ScreenToGif
- **macOS**: Kap or Gifox
- **Linux**: Peek
- **Cross-platform**: OBS Studio