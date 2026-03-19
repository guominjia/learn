# Scrapy: Why `custom_settings` on Your Spider Gets Ignored

## The Problem

You define `custom_settings` on your Spider class, expecting Scrapy to pick them up — but they silently have no effect. The spider runs with default (or externally provided) settings as if `custom_settings` doesn't exist.

```python
class DocSpider(scrapy.Spider):
    name = "doc"
    custom_settings = {
        "DEPTH_LIMIT": 3,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.5,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408],
    }
```

Despite this, none of these settings actually take effect at runtime.

---

## Root Cause

The issue arises when you **instantiate `Crawler` directly** and pass in external settings, instead of using the standard `CrawlerProcess` or `CrawlerRunner`.

```python
# This bypasses Spider.custom_settings
crawler = Crawler(settings=cfg.crawler_settings)
deferred = crawler.crawl(DocSpider, urls=urls)
```

### How `custom_settings` Normally Works

In Scrapy's standard flow:

1. `Crawler` is initialized with base settings.
2. When `crawler.crawl(SpiderClass, ...)` is called, Scrapy internally invokes `_apply_settings()`.
3. `_apply_settings()` **merges** the Spider's `custom_settings` into the active settings object.

This merge step is guaranteed when you use `CrawlerProcess` or `CrawlerRunner`, because they follow the full initialization protocol.

### What Goes Wrong with Direct `Crawler` Instantiation

When you create a `Crawler` instance directly, the `crawl()` method may **skip** the `_apply_settings()` call. In that case, `custom_settings` on the Spider class is never read, and the settings you passed into the `Crawler` constructor are the only ones in effect.

| Approach | `custom_settings` Applied? |
|---|---|
| `CrawlerProcess` / `CrawlerRunner` | Yes — merged automatically |
| Direct `Crawler(settings=...)` | **No** — unless `_apply_settings()` is explicitly called |

---

## How to Verify

Print the active settings after creating the crawler to confirm what is actually loaded:

```python
crawler = Crawler(settings=cfg.crawler_settings)
print(dict(crawler.settings))  # Check if custom_settings values appear
```

If `DEPTH_LIMIT`, `CONCURRENT_REQUESTS`, etc. are missing or have default values, you know the merge didn't happen.

---

## Solutions

### Option 1: Manually Merge Spider Settings

Before constructing the `Crawler`, read the Spider's `custom_settings` and merge them into your configuration:

```python
def main(urls, crawler_cfg_file):
    install_reactor('twisted.internet.asyncioreactor.AsyncioSelectorReactor')
    from twisted.internet import reactor

    cfg = CrawlerCfg.from_yaml(crawler_cfg_file)

    # Manually merge Spider's custom_settings into the config
    if hasattr(DocSpider, 'custom_settings') and DocSpider.custom_settings:
        cfg.crawler_settings.update(DocSpider.custom_settings)

    configure_logging(cfg.crawler_settings)

    crawler = Crawler(settings=cfg.crawler_settings)
    deferred = crawler.crawl(DocSpider, urls=urls)

    deferred.addBoth(lambda _: reactor.stop())
    reactor.run()
```

This ensures the Spider's settings are present **before** the `Crawler` is initialized.

### Option 2: Move Settings to the YAML Config

Remove `custom_settings` from the Spider class entirely and define everything in your external configuration file:

```yaml
# crawler.yaml
DEPTH_LIMIT: 3
CONCURRENT_REQUESTS: 8
DOWNLOAD_DELAY: 0.5
RETRY_TIMES: 3
RETRY_HTTP_CODES: [500, 502, 503, 504, 408]
```

This is the **recommended approach**. It keeps all configuration in one place and avoids the split-brain problem of settings defined in two locations.

### Option 3: Use `CrawlerProcess` / `CrawlerRunner`

If your architecture allows it, switch to Scrapy's standard entry points so that `custom_settings` is merged automatically:

```python
from scrapy.crawler import CrawlerProcess

process = CrawlerProcess(settings=cfg.crawler_settings)
process.crawl(DocSpider, urls=urls)
process.start()
```

---

## Key Takeaway

Scrapy's `custom_settings` class attribute relies on the framework calling `_apply_settings()` during the crawl setup. When you bypass the standard `CrawlerProcess` / `CrawlerRunner` flow and instantiate `Crawler` directly, that merge step may be skipped. The fix is straightforward: either merge manually, centralize settings in your config file, or use the standard Scrapy entry points.