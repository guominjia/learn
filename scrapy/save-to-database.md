# Saving Scraped Data to a Database in Scrapy

## Overview

Scrapy's pipeline architecture makes it straightforward to persist scraped items to a
database. Combining a **database pipeline** with a **custom duplicate filter** lets you:

1. Store every successfully scraped item.
2. Skip URLs that have already been crawled across spider runs (persistent deduplication).

---

## Architecture

```
Spider → Item → Pipeline → Database
                              ↑
                    DupeFilter checks DB before
                    the request is scheduled
```

---

## 1. Database Pipeline (SQLAlchemy + SQLite / PostgreSQL)

### Install dependencies

```bash
pip install sqlalchemy psycopg2-binary   # PostgreSQL
# or
pip install sqlalchemy                   # SQLite (built-in)
```

### Define the ORM model — `models.py`

```python
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    url        = Column(String, unique=True, nullable=False, index=True)
    title      = Column(String)
    content    = Column(String)
    scraped_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_url: str):
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return engine
```

### Write the pipeline — `pipelines.py`

```python
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from .models import Article, get_engine


class DatabasePipeline:
    """Persist scraped items and silently drop duplicates (unique URL constraint)."""

    def open_spider(self, spider):
        db_url = spider.settings.get("DATABASE_URL", "sqlite:///scrapy.db")
        engine = get_engine(db_url)
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def close_spider(self, spider):
        self.session.close()

    def process_item(self, item, spider):
        article = Article(
            url     = item["url"],
            title   = item.get("title"),
            content = item.get("content"),
        )
        self.session.add(article)
        try:
            self.session.commit()
            spider.logger.info(f"Saved: {item['url']}")
        except IntegrityError:
            # Duplicate URL — the unique constraint fired
            self.session.rollback()
            spider.logger.debug(f"Duplicate skipped: {item['url']}")
        return item
```

### Enable the pipeline — `settings.py`

```python
DATABASE_URL = "postgresql://user:password@localhost/scrapy_db"
# DATABASE_URL = "sqlite:///scrapy.db"   # quick local testing

ITEM_PIPELINES = {
    "myproject.pipelines.DatabasePipeline": 300,
}
```

---

## 2. Persistent Duplicate Request Filter

The pipeline above deduplicates **items**. To avoid even *making* the HTTP request again,
implement a custom `RFPDupeFilter` that checks the database before scheduling a request.

### `dupefilter.py`

```python
from scrapy.dupefilters import RFPDupeFilter
from sqlalchemy.orm import sessionmaker
from .models import Article, get_engine


class DBDupeFilter(RFPDupeFilter):
    """
    Extends Scrapy's default fingerprint filter with a DB look-up so that
    URLs already stored in the database are never re-requested, even across
    spider restarts.
    """

    @classmethod
    def from_settings(cls, settings):
        instance = super().from_settings(settings)
        db_url   = settings.get("DATABASE_URL", "sqlite:///scrapy.db")
        engine   = get_engine(db_url)
        Session  = sessionmaker(bind=engine)
        instance.db_session = Session()
        return instance

    def request_seen(self, request):
        # 1. Check in-memory fingerprint set (fast path)
        if super().request_seen(request):
            return True

        # 2. Check persistent DB (survives restarts)
        url   = request.url
        exists = (
            self.db_session.query(Article)
            .filter_by(url=url)
            .first()
        )
        return exists is not None

    def close(self, reason):
        self.db_session.close()
        super().close(reason)
```

### Register the filter — `settings.py`

```python
DUPEFILTER_CLASS = "myproject.dupefilter.DBDupeFilter"
```

---

## 3. Full Project Layout

```
myproject/
├── scrapy.cfg
├── myproject/
│   ├── __init__.py
│   ├── settings.py          ← DATABASE_URL, ITEM_PIPELINES, DUPEFILTER_CLASS
│   ├── models.py            ← SQLAlchemy ORM + get_engine()
│   ├── pipelines.py         ← DatabasePipeline
│   ├── dupefilter.py        ← DBDupeFilter
│   └── spiders/
│       └── article_spider.py
```

---

## 4. Spider Example

```python
import scrapy


class ArticleSpider(scrapy.Spider):
    name            = "article"
    start_urls      = ["https://example.com/articles"]

    def parse(self, response):
        for link in response.css("a.article-link::attr(href)").getall():
            yield response.follow(link, callback=self.parse_article)

        # Follow pagination
        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_article(self, response):
        yield {
            "url"    : response.url,
            "title"  : response.css("h1::text").get(default="").strip(),
            "content": " ".join(response.css("article p::text").getall()),
        }
```

---

## 5. Deduplication Strategy Summary

| Layer | Mechanism | Scope |
|---|---|---|
| **Request filter** (`DBDupeFilter`) | DB look-up before HTTP request | Across spider restarts |
| **Pipeline** (`DatabasePipeline`) | `UNIQUE` constraint + `IntegrityError` catch | Within a single run |
| **Scrapy default** (`RFPDupeFilter`) | In-memory fingerprint set | Current run only |

- Use **all three layers** for the most robust deduplication.
- The DB constraint is the final safety net — it prevents corrupt data even if the
  filter is bypassed.

---

## 6. Running the Spider

```bash
# First run — crawls and saves everything
scrapy crawl article

# Subsequent runs — DB filter skips already-seen URLs automatically
scrapy crawl article
```

---

## 7. Tips

- **Index the `url` column** for fast look-ups (`index=True` in the ORM model above).
- Use **connection pooling** (`pool_size`, `max_overflow` in `create_engine`) for
  high-concurrency crawls.
- For very large crawl sets consider a **Bloom filter** (e.g. `scrapy-crawl-once`) as
  a memory-efficient alternative to DB look-ups on every request.
- Set `DOWNLOAD_DELAY` and `AUTOTHROTTLE_ENABLED = True` to be respectful to target
  servers.
