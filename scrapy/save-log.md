# Scrapy Log Saving

## Overview

By default, Scrapy prints log messages to the console (stdout) but does **not** persist them to disk. Once the spider finishes, all log output is lost. This document explains how to control log verbosity with `LOG_LEVEL` and how to persist logs to a file using Scrapy's built-in settings.

---

## Log Settings in `settings.py`

### `LOG_LEVEL`

Controls the minimum severity of messages that are displayed. Any message below this level is silenced.

| Value | Description |
|-------|-------------|
| `DEBUG` | Most verbose — all internal Scrapy details |
| `INFO` | General progress information (default) |
| `WARNING` | Non-critical problems |
| `ERROR` | Errors that need attention |
| `CRITICAL` | Fatal errors only |

```python
# settings.py
LOG_LEVEL = 'INFO'   # default
LOG_LEVEL = 'DEBUG'  # show everything
LOG_LEVEL = 'WARNING'  # production-friendly, less noise
```

---

### `LOG_FILE` — Persist Logs to Disk

Set `LOG_FILE` to a file path to redirect all log output from the console to a file.

```python
# settings.py
LOG_FILE = 'scrapy.log'
```

Once set, **nothing** is printed to the console; everything goes to the specified file.

> **Tip:** Use an absolute path to ensure the file lands in a predictable location regardless of where you invoke `scrapy crawl`.

```python
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'scrapy.log')
```

---

### `LOG_FILE_APPEND`

Controls whether each run appends to the existing log file or overwrites it.

```python
LOG_FILE_APPEND = True   # append — keeps history across runs (default: True)
LOG_FILE_APPEND = False  # overwrite — fresh log every run
```

---

### `LOG_ENCODING`

Character encoding used when writing the log file. Defaults to `utf-8`.

```python
LOG_ENCODING = 'utf-8'
```

---

### `LOG_FORMAT` / `LOG_DATEFORMAT`

Customize the format of each log line.

```python
LOG_FORMAT    = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'
```

---

## Minimal Production Configuration

```python
# settings.py

LOG_LEVEL       = 'WARNING'          # suppress DEBUG / INFO noise
LOG_FILE        = 'logs/scrapy.log'  # save to file
LOG_FILE_APPEND = True               # keep log history
LOG_ENCODING    = 'utf-8'
LOG_FORMAT      = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT  = '%Y-%m-%d %H:%M:%S'
```

---

## Override via Command Line

You can override any setting at runtime without editing `settings.py`:

```bash
# Save logs to a specific file for this run only
scrapy crawl myspider -s LOG_FILE=run_$(date +%F).log -s LOG_LEVEL=DEBUG
```

---

## Rotating Logs (Advanced)

Scrapy's built-in `LOG_FILE` does not support rotation. For production use, integrate Python's `logging.handlers.RotatingFileHandler` in a custom extension or use an external tool like `logrotate` (Linux).

### Example: Custom Scrapy Extension for Rotating Logs

```python
# myproject/extensions/rotating_log.py
import logging
from logging.handlers import RotatingFileHandler
from scrapy import signals

class RotatingLogExtension:
    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        return ext

    def spider_opened(self, spider):
        handler = RotatingFileHandler(
            filename='logs/scrapy.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # keep 5 rotated files
            encoding='utf-8',
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        ))
        logging.getLogger().addHandler(handler)
```

Register it in `settings.py`:

```python
EXTENSIONS = {
    'myproject.extensions.rotating_log.RotatingLogExtension': 100,
}
```

---

## Summary

| Setting | Purpose | Default |
|---------|---------|---------|
| `LOG_LEVEL` | Minimum log severity shown | `DEBUG` |
| `LOG_FILE` | Path to persist logs; `None` = console only | `None` |
| `LOG_FILE_APPEND` | Append vs overwrite on each run | `True` |
| `LOG_ENCODING` | File encoding | `utf-8` |
| `LOG_FORMAT` | Log line format string | Scrapy default |
| `LOG_DATEFORMAT` | Date format in log lines | `%Y-%m-%d %H:%M:%S` |
