# Handling Large Tables When Converting Web Pages to PDF with Playwright

Converting web pages to PDF is a common task in web scraping and automation pipelines. Tools like Playwright make it straightforward — until you encounter **large tables**. This post walks through the pain points, the solutions attempted, and the best practical approach.

## The Setup: Playwright PDF Generation

A typical Playwright-based converter looks like this:

```python
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import os

STORAGE_FILE = "storage_state.json"

def webpage_to_pdf(url: str, output_path: str, timeout: int = 30000):
    with sync_playwright() as p:
        browser = p.chromium.launch()

        PREFIX = urlparse(url).netloc + "_"
        if os.path.exists(PREFIX + STORAGE_FILE):
            context = browser.new_context(storage_state=PREFIX + STORAGE_FILE)
        else:
            context = browser.new_context()

        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout)

        if "login" in page.url or "signin" in page.url:
            print("Authentication required. Please run save_login_state() first.")
            browser.close()
            return

        page.pdf(
            path=output_path,
            format="A4",
            landscape=True,
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"}
        )

        browser.close()
        print(f"PDF saved to {output_path}")
```

This works fine for most pages — until you hit a page with a wide table (10+ columns). Then two problems emerge.

## Problem 1: Landscape Mode Is Slow

Setting `landscape=True` gives you roughly 40% more horizontal space, which helps wide tables fit. But it noticeably slows down PDF generation because Chromium internally:

1. **Recalculates the entire page layout** for the new aspect ratio
2. **Re-evaluates print media queries** (`@media print` + `orientation: landscape`)
3. **Triggers component re-renders** in SPA frameworks (React, Vue) that detect viewport size changes

## Problem 2: Without Landscape, Content Gets Clipped

This is a **known limitation of Chromium's PDF renderer**. When a table's total column width exceeds the page width, everything beyond the right margin is simply cut off. There's no overflow, no wrapping — just gone.

## Attempted Solution 1: CSS Injection

The first instinct is to inject print-specific CSS to force tables to fit within the page:

```python
page.add_style_tag(content="""
    @media print {
        table {
            width: 100% !important;
            table-layout: fixed !important;
            word-break: break-all !important;
            font-size: 8px !important;
        }
        th, td {
            overflow: hidden !important;
            white-space: normal !important;
        }
        * {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }
    }
""")
```

**Verdict:** Helps for moderately wide tables, but `table-layout: fixed` + small font hits a readability floor. For tables with 20+ columns, the text becomes unreadable or columns still overflow.

## Attempted Solution 2: Alternative Libraries

| Library | Pros | Cons |
|---|---|---|
| **WeasyPrint** | Excellent `@page` CSS support, good table pagination | No JavaScript rendering (SPA pages won't work) |
| **pdfkit / wkhtmltopdf** | Flexible config, decent table handling | Requires external binary, weak JS support |
| **Puppeteer (Node.js)** | Nearly identical to Playwright, larger community | Requires Node.js environment |

**Verdict:** All PDF solutions share the same fundamental constraint — PDF pages have a fixed physical width. No library can escape this.

## Attempted Solution 3: Dynamic Scale

Playwright's `page.pdf()` accepts a `scale` parameter. You can dynamically calculate the ratio needed to fit the page content:

```python
scroll_width = page.evaluate("document.body.scrollWidth")
a4_landscape_width = 1122  # A4 landscape width in pixels at 96 DPI

scale = min(1.0, a4_landscape_width / scroll_width)
scale = max(0.1, scale)  # floor at 10%

page.pdf(
    path=output_path,
    format="A4",
    landscape=True,
    print_background=True,
    scale=scale,
    margin={"top": "5mm", "bottom": "5mm", "left": "5mm", "right": "5mm"}
)
```

**Verdict:** This is the **only way to guarantee no content loss** in a PDF. But for very wide tables, the entire page shrinks to a point where nothing is legible.

## The Real Solution: Don't Put Large Tables in PDFs

The realization is straightforward once you step back:

- **HTML** supports infinite horizontal scrolling
- **Excel** supports infinite horizontal scrolling
- **PDF** does not — it simulates a fixed-size piece of paper

**PDF is fundamentally the wrong format for wide tabular data.**

### Extract Tables to Excel Instead

```python
import pandas as pd
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse
import os

def extract_tables(url: str, output_path: str, timeout: int = 30000):
    with sync_playwright() as p:
        browser = p.chromium.launch()

        PREFIX = urlparse(url).netloc + "_"
        if os.path.exists(PREFIX + STORAGE_FILE):
            context = browser.new_context(storage_state=PREFIX + STORAGE_FILE)
        else:
            context = browser.new_context()

        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout)

        if "login" in page.url or "signin" in page.url:
            print("Authentication required.")
            browser.close()
            return

        # Extract all table data via JavaScript
        tables_data = page.evaluate("""
            () => {
                const tables = document.querySelectorAll('table');
                return Array.from(tables).map(table => {
                    const rows = table.querySelectorAll('tr');
                    return Array.from(rows).map(row => {
                        const cells = row.querySelectorAll('th, td');
                        return Array.from(cells).map(cell => cell.innerText.trim());
                    });
                });
            }
        """)

        browser.close()

        # Write each table to a separate Excel sheet
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for i, table in enumerate(tables_data):
                if not table:
                    continue
                df = pd.DataFrame(table[1:], columns=table[0])
                sheet_name = f"Table_{i+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")

        print(f"Tables saved to {output_path}")
```

### Side-by-Side Comparison

| | PDF Conversion | Table Extraction to Excel |
|---|---|---|
| Wide tables | Truncated or unreadably small | Complete, scrollable |
| Data editable | No | Yes |
| Downstream analysis | Requires re-parsing | Direct pandas/Excel use |
| Speed | Slow (layout + render) | Fast (DOM query only) |
| File size | Large | Small |

## A Hybrid Strategy

In practice, the best approach for scraping pages that contain both narrative content and large tables is a **hybrid**:

1. **Save the page as PDF** (with `scale` or CSS injection) for the text/visual content
2. **Extract tables separately** into Excel/CSV for the data
3. Or simply **save the raw HTML** — a browser can render it with full horizontal scrolling, preserving both layout and data fidelity

```python
# Save raw HTML — simplest approach for full fidelity
html_content = page.content()
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_content)
```

## Key Takeaways

1. **PDF is paper simulation** — it has a fixed width and cannot scroll. This is a format-level limitation, not a tool limitation.
2. **`landscape=True`** gives ~40% more width but slows rendering due to full layout recalculation.
3. **CSS injection** and **`scale`** are workarounds, not solutions — they trade readability for completeness.
4. **For tabular data, extract to Excel/CSV.** The data stays intact, editable, and analyzable.
5. **For visual fidelity, save as HTML.** Browsers handle horizontal overflow natively.

> Choose the output format that matches your actual use case. If you need the data, don't flatten it into a fixed-width image of paper.
