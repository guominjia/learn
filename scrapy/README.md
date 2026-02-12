# Scrapy
Discuss scrapy, spider, crawler

## References

- [Docs](https://docs.scrapy.org/en/latest/index.html)
- [Example](https://github.com/guominjia/learn/tree/code_study/scrapy_mycrawler)

## Usage
```bash
scrapy startproject mycrawler
cd mycrawler
scrapy genspider example example.com
scrapy crawl example -L INFO - o example.json
```
Or
```python
from scrapy.crawler import CrawlerRunner

runner = CrawlerRunner()
runner.crawl(
    ExampleSpider,
    some_config={'key': 'value'}
)

class ExampleSpider(scrapy.Spider):
    name = "example"

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        """Scrapy use this method to instance spider first"""
        spider = super().from_crawler(crawler, *args, **kwargs)
        
        spider.key = crawler.settings.get('KEY', {})
        
        return spider

    def __init__(self, auth_config=None, *args, **kwargs):
        """Maybe below is unnecessary if Spider have implement it
        
        TODO need to verify it
        """
        super().__init__(*args, **kwargs)
        # super() have below logic to initialize key/value
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
```

## Event
```python
from scrapy import signals

@classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s
```

## Request
```python
request = scrapy.Request(
    url='https://example.com',
    headers={'Authorization': 'Bearer xxx'},  # ← Send to server
    meta={'auth_config': {...}}               # ← Keep inside
)
```

## Response
```python
from scrapy.http import TextResponse

response = TextResponse(
                url=response.url,
                status=response.status,
                headers=response.headers,
                body=response.body,
                encoding='utf-8',
                request=response.request
            )
```

## Middlewares

### 返回值规则

| 返回值 | 效果 |
|--------|------|
| `None` | 继续传递给下一个 middleware 或 downloader |
| `Request` | 重新调度这个请求（会再次经过 scheduler 和去重检查）|
| `Response` | 直接返回响应，跳过下载 |
| `raise IgnoreRequest` | 忽略这个请求 |

## Cache

### 默认（内存）
Scrapy 使用 `RFPDupeFilter`（Request Fingerprint Duplicate Filter），去重记录保存在**内存**中，进程结束后自动清除。

### 持久化（文件/数据库）

```python
DUPEFILTER_CLASS = 'scrapy.dupefilters.RFPDupeFilter'
JOBDIR = 'crawls/somejob'  # 会在这里保存去重数据
```

## Logging
```python
import scrapy
import logging

logging.getLogger('spnego').setLevel(logging.WARNING)
```
Or
```python settings.py
LOG_LEVEL = 'INFO'

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'scrapy': {
            'level': 'INFO',
        },
        'mycrawler': {
            'level': 'DEBUG',
        },
        'spnego': {
            'level': 'WARNING',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}
```

## Flowchart

```
Spider
    ↓
Spider.start_requests()
    ↓ Yield Request
    ↓ (default start_requests will read start_urls to yield Request)
    ↓
Spider Middleware.process_start_requests()
    ↓ Modify Request
    ↓
Scheduler
    ↓ Read Request
    ↓
Downloader Middleware.process_request()
    ↓ Modify Request (Add auth, proxy, etc)
    ↓
Downloader
    ↓ Send HTTP Request
    ↓
Downloader Middleware.process_response()
    ↓ Handle Response (Decompress, Convert)
    ↓
Spider Middleware.process_spider_input()
    ↓ Preprocess Response
    ↓
Spider.parse(response)
    ↓ Parse Response
    ↓
    ├─→ yield Request (New Request)
    │       ↓
    │   Back Scheduler again
    │
    └─→ yield Item (Data) ← ★ Items appear here
            ↓
Spider Middleware.process_spider_output()
            ↓ Filter/Modify Items
            ↓
Item Pipeline.process_item()  ← ★ Pipeline process Items
            ↓
    ┌───────┴───────┐
    │               │
Save to DB       Save to file
(MongoDB/MySQL)  (JSON/CSV)
```

## Example

````python
import scrapy
from mycrawler.items import MycrawlerItem

class ExampleSpider(scrapy.Spider):
    name = "example"

    # start_urls = ["https://example.com"] Can keep this and remove start_requests
    def start_requests(self):
        yield scrapy.Request("https://example.com", callback=self.parse)
    
    def parse(self, response):
        # Method 1: yield dict
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),
            'content': response.text,
        }
        
        # Method 2: yield Item object
        item = MycrawlerItem()
        item['url'] = response.url
        item['title'] = response.css('title::text').get()
        item['markdown'] = self._convert_to_markdown(response)
        yield item
        
        # Method 3: yield new Request to continue crawl
        for link in response.css('a::attr(href)').getall():
            yield response.follow(link, callback=self.parse)
````

---

````python
import scrapy

class MycrawlerItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    markdown = scrapy.Field()
    html = scrapy.Field()
    status_code = scrapy.Field()
    crawled_at = scrapy.Field()
    metadata = scrapy.Field()
````

**Item 的优势**:
- ✅ 类型安全：定义了哪些字段可以使用
- ✅ IDE 自动补全
- ✅ 字段验证
- ✅ 序列化支持（JSON/XML）

---

````python
class MycrawlerSpiderMiddleware:
    async def process_spider_output(self, response, result, spider):
        async for item_or_request in result:
            if isinstance(item_or_request, scrapy.Request):
                yield item_or_request
            else:
                if isinstance(item_or_request, dict):
                    from datetime import datetime
                    item_or_request['crawled_at'] = datetime.now().isoformat()
                
                if item_or_request.get('title'):
                    yield item_or_request
                else:
                    spider.logger.warning(f"Dropped item without title: {response.url}")
````

---

````python
from itemadapter import ItemAdapter
from datetime import datetime

class MycrawlerPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        if adapter.get('title'):
            adapter['title'] = adapter['title'].strip()
        
        if not adapter.get('url'):
            from scrapy.exceptions import DropItem
            raise DropItem(f"Missing URL in {item}")
        
        adapter['crawled_at'] = datetime.now().isoformat()
        
        spider.logger.info(f"✓ Processed item: {adapter['url']}")
        return item

class MarkdownPipeline:   
    def process_item(self, item, spider):
        from markdownify import markdownify
        
        adapter = ItemAdapter(item)
        
        if adapter.get('html'):
            adapter['markdown'] = markdownify(
                adapter['html'],
                heading_style="ATX",
                strip=['script', 'style']
            )
        
        return item

class JsonExportPipeline:
    def open_spider(self, spider):
        import json
        self.file = open(f'{spider.name}_output.json', 'w', encoding='utf-8')
        self.items = []
    
    def close_spider(self, spider):
        import json
        json.dump(self.items, self.file, ensure_ascii=False, indent=2)
        self.file.close()
        spider.logger.info(f"✓ Exported {len(self.items)} items to JSON")
    
    def process_item(self, item, spider):
        from itemadapter import ItemAdapter
        self.items.append(ItemAdapter(item).asdict())
        return item

class DatabasePipeline:
    def open_spider(self, spider):
        import sqlite3
        self.conn = sqlite3.connect('crawled_data.db')
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                markdown TEXT,
                crawled_at TEXT
            )
        ''')
        self.conn.commit()
    
    def close_spider(self, spider):
        self.conn.close()
    
    def process_item(self, item, spider):
        from itemadapter import ItemAdapter
        adapter = ItemAdapter(item)
        
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO pages (url, title, markdown, crawled_at)
                VALUES (?, ?, ?, ?)
            ''', (
                adapter.get('url'),
                adapter.get('title'),
                adapter.get('markdown'),
                adapter.get('crawled_at')
            ))
            self.conn.commit()
            spider.logger.info(f"✓ Saved to database: {adapter['url']}")
        
        except Exception as e:
            spider.logger.error(f"Database error: {e}")
        
        return item
````

---

````python
ITEM_PIPELINES = {
    'mycrawler.pipelines.MycrawlerPipeline': 100,      # Data cleanup
    'mycrawler.pipelines.MarkdownPipeline': 200,       # Markdown Conversion
    'mycrawler.pipelines.DatabasePipeline': 300,       # Save to DB
    'mycrawler.pipelines.JsonExportPipeline': 400,     # Export JSON
}
````

---

```
Spider.parse() yield item
    ↓
Spider Middleware.process_spider_output()
    ↓
MycrawlerPipeline.process_item()       (Priority 100)
    ↓ data cleanup
    ↓
MarkdownPipeline.process_item()        (Priority 200)
    ↓ Convert Markdown
    ↓
DatabasePipeline.process_item()        (Priority 300)
    ↓ Save to DB
    ↓
JsonExportPipeline.process_item()      (Priority 400)
    ↓ Export JSON
    ↓
Finish
```