# Converting HTML to PDF in Python: A Complete Guide

When building web scraping or document processing pipelines, you often need to convert web content into portable formats like PDF. This post walks through the landscape of Python tools for HTML-to-PDF conversion—from static parsers to headless browser automation—and clarifies common misconceptions along the way.

---

## Understanding the Tool Landscape

A common point of confusion is mixing up which tool does what. Here's a quick reference:

| Tool | Direction |
|------|-----------|
| **markdownify** | HTML → Markdown |
| **weasyprint / pandoc / markdown-pdf** | Markdown / HTML → PDF |
| **BeautifulSoup4 (bs4)** | HTML → Parsed DOM (static) |
| **Playwright / Selenium** | URL → Rendered Page → PDF |

**Key takeaway:** `markdownify` converts HTML *to* Markdown, not the other way around. For Markdown-to-PDF or HTML-to-PDF, you need different tools.

---

## How markdownify Handles Images

When `markdownify` converts HTML to Markdown, it simply translates `<img>` tags into Markdown image syntax. It does **not** download or embed the actual image data:

```python
# HTML input
# <img src="https://example.com/image.png" alt="example image">

# Markdown output
# ![example image](https://example.com/image.png)
```

The image URL is preserved as-is—no network requests are made.

---

## Markdown to PDF: Three Quick Options

If your goal is to convert Markdown files into PDF, here are the most common approaches:

### Option 1: WeasyPrint + markdown (Python)

```python
import markdown
from weasyprint import HTML

md_content = open("file.md").read()
html_content = markdown.markdown(md_content)
HTML(string=html_content).write_pdf("output.pdf")
```

### Option 2: Pandoc (CLI)

```bash
pandoc input.md -o output.pdf
```

### Option 3: markdown-pdf (Node.js)

```bash
npm install -g markdown-pdf
markdown-pdf input.md
```

**Image handling during conversion:**
- **Local images**: Paths must be correct, relative to the Markdown file location.
- **Remote images**: The conversion tool downloads them automatically (requires network access).
- **Base64 images**: Embedded directly into the PDF.

---

## How PDFs Store Media Content

A PDF is fundamentally a **self-contained** document format. It does **not** automatically fetch external resources when opened.

### Images in PDF

| Storage Method | Behavior |
|---------------|----------|
| **Embedded** | Image data is encoded inside the PDF file (most common) |
| **External link** | PDF references an external URL; **will not auto-download** on open |

### Video / Audio in PDF

| Storage Method | Behavior |
|---------------|----------|
| **Embedded** | Media file stored inside the PDF |
| **External link** | Requires explicit user interaction to trigger |

```
When a PDF file is opened:
  ✅ Embedded images    → Display immediately
  ❌ External images    → Show as blank or placeholder (no auto-fetch)
  ❌ External video     → Not downloaded unless user clicks
```

### Extracting Embedded Images from PDF

If you're building a RAG pipeline and need to extract images from a PDF, use **PyMuPDF**:

```python
import fitz  # PyMuPDF

doc = fitz.open("file.pdf")
for page in doc:
    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]  # Direct bytes, no network needed
```

---

## BeautifulSoup4: A Purely Static Parser

BeautifulSoup4 (bs4) is a **static** HTML/XML parser. It parses the document structure as a string—nothing more.

```
What bs4 DOES:
  ✅ Parse HTML structure (tags, attributes, text)
  ✅ Extract image src URLs (as strings)
  ✅ Extract script content (as strings)

What bs4 does NOT do:
  ❌ Download images
  ❌ Execute JavaScript
  ❌ Make network requests
  ❌ Render CSS
  ❌ Handle dynamic content (Ajax / React / Vue)
```

### Example

```python
from bs4 import BeautifulSoup

html = """
<html>
  <body>
    <img src="https://example.com/image.png" alt="test">
    <script src="https://example.com/app.js"></script>
    <script>console.log("hello")</script>
  </body>
</html>
"""

soup = BeautifulSoup(html, "html.parser")

# Only retrieves the URL string — does NOT download the image
img_url = soup.find("img")["src"]
print(img_url)  # https://example.com/image.png

# Only retrieves the script content string — does NOT execute it
script = soup.find("script").string
print(script)  # console.log("hello")
```

### When You Need More Than Static Parsing

| Need | Tool |
|------|------|
| Execute JavaScript | `Selenium` / `Playwright` |
| Download images | `requests` / `httpx` |
| Render a full page | `Playwright` + screenshot |

**In short:** bs4 is an HTML string parser. It has no awareness of the web and makes no network calls.

---

## Automating Webpage-to-PDF with Headless Browsers

Chrome's built-in "Print to PDF" feature produces high-fidelity PDFs because it fully renders the page first. You can automate this with headless browsers.

### Option 1: Playwright (Recommended)

```python
from playwright.sync_api import sync_playwright

def webpage_to_pdf(url: str, output_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Wait for complete page load including async resources
        page.goto(url, wait_until="networkidle")

        # Print to PDF (mirrors Chrome's print functionality)
        page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={
                "top": "20mm",
                "bottom": "20mm",
                "left": "15mm",
                "right": "15mm"
            }
        )

        browser.close()
        print(f"PDF saved to {output_path}")

webpage_to_pdf("https://example.com", "output.pdf")
```

Installation:

```bash
pip install playwright
playwright install chromium
```

### Option 2: Selenium + Chrome DevTools Protocol

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json, base64, time

def webpage_to_pdf_selenium(url: str, output_path: str):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    app_state = {
        "recentDestinations": [{"id": "Save as PDF", "origin": "local"}],
        "selectedDestinationId": "Save as PDF",
        "version": 2
    }
    options.add_experimental_option("prefs", {
        "printing.print_preview_sticky_settings.appState": json.dumps(app_state),
        "savefile.default_directory": "C:\\output"
    })
    options.add_argument("--kiosk-printing")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(2)  # Wait for page load

    # Use CDP command to print
    pdf_data = driver.execute_cdp_cmd("Page.printToPDF", {
        "paperWidth": 8.27,    # A4 width in inches
        "paperHeight": 11.69,  # A4 height in inches
        "printBackground": True,
        "marginTop": 0.8,
        "marginBottom": 0.8,
    })

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(pdf_data["data"]))

    driver.quit()
```

### Option 3: Pyppeteer (Python port of Puppeteer)

```python
import asyncio
import pyppeteer

async def webpage_to_pdf(url: str, output_path: str):
    browser = await pyppeteer.launch(
        headless=True,
        args=["--no-sandbox"]
    )
    page = await browser.newPage()
    await page.goto(url, {"waitUntil": "networkidle0"})

    await page.pdf({
        "path": output_path,
        "format": "A4",
        "printBackground": True
    })

    await browser.close()

asyncio.run(webpage_to_pdf("https://example.com", "output.pdf"))
```

---

## Comparison of Headless Browser Approaches

| Feature | Playwright | Selenium + CDP | Pyppeteer |
|---------|-----------|---------------|-----------|
| Actively maintained | ✅ Yes | ✅ Yes | ⚠️ Minimal |
| PDF quality | ✅ Best | ✅ Good | ✅ Good |
| Ease of use | ✅ Simplest | ⚠️ More complex | ⚠️ Async-only |
| Dynamic content support | ✅ Built-in | ⚠️ Manual waits | ⚠️ Manual waits |

---

## Playwright PDF Parameters Reference

```python
page.pdf(
    path="output.pdf",
    format="A4",              # Paper size
    print_background=True,    # Include background images/colors
    landscape=False,          # Portrait or landscape
    scale=1.0,                # Scale factor (0.1 - 2)
    page_ranges="1-5",        # Print specific pages
    header_template="<div>Header</div>",
    footer_template="<div>Footer</div>",
    wait_for_timeout=3000     # Extra wait time in ms
)
```

---

## Summary

| Task | Best Tool |
|------|-----------|
| HTML → Markdown | `markdownify` |
| Markdown → PDF | `weasyprint`, `pandoc` |
| Static HTML parsing | `BeautifulSoup4` |
| Full webpage → PDF (with JS/CSS) | **Playwright** (recommended) |
| Extract images from existing PDF | `PyMuPDF (fitz)` |

**Playwright** is the recommended choice for webpage-to-PDF conversion. It produces the highest quality output, has the simplest API, and handles dynamic content natively through built-in wait strategies.