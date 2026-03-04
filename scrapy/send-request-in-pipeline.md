在 Scrapy Pipeline 中发送额外的 HTTP 请求**不违反框架设计**，这是一个常见的使用场景。

## 建议的实现方式

### 方案 1：在 Pipeline 中发送同步请求（推荐）

````python
import requests

class OnedriveContentPipeline:
    def process_item(self, item, spider):
        # 获取文件内容的 URL
        content_url = item.get('download_url') or f"{item['url']}/content"
        
        # 使用 requests 同步获取内容
        try:
            response = requests.get(
                content_url,
                headers={'Authorization': f"Bearer {spider.access_token}"},
                timeout=30
            )
            response.raise_for_status()
            
            item['content'] = response.content  # 二进制内容
            # 或者 item['text'] = response.text  # 文本内容
            
        except requests.RequestException as e:
            spider.logger.error(f"Failed to download {content_url}: {e}")
            item['content'] = None
            
        return item
````

### 方案 2：使用 Scrapy 的异步下载（更符合框架）

````python
from scrapy.http import Request
from scrapy.exceptions import DropItem
from twisted.internet import defer

class OnedriveContentPipeline:
    def process_item(self, item, spider):
        # 标记需要下载内容的 item
        if item.get('type') == 'file' and not item.get('content_downloaded'):
            content_url = f"{item['url']}/content"
            
            # 创建新的 Request 并放回调度队列
            request = Request(
                content_url,
                headers={'Authorization': f"Bearer {spider.access_token}"},
                callback=spider.parse_file_content,
                meta={'item': item}
            )
            spider.crawler.engine.crawl(request, spider)
            
            # 先不返回 item，等内容下载完成后再返回
            raise DropItem("Waiting for content download")
            
        return item
````

## 推荐做法

**使用方案 1（同步请求）**，因为：
- ✅ 简单直接，易于维护
- ✅ Pipeline 中处理下载是常见模式（类似 `ImagesPipeline`、`FilesPipeline`）
- ✅ 不会打乱 Scrapy 的调度流程
- ✅ 更容易处理错误和重试

## 性能优化建议

如果担心性能，可以使用线程池：

````python
from concurrent.futures import ThreadPoolExecutor
import requests

class OnedriveContentPipeline:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def process_item(self, item, spider):
        if item.get('type') == 'file':
            # 在线程池中异步下载
            future = self.executor.submit(self._download_content, item, spider)
            item['content'] = future.result()  # 等待结果
        return item
    
    def _download_content(self, item, spider):
        content_url = f"{item['url']}/content"
        response = requests.get(
            content_url,
            headers={'Authorization': f"Bearer {spider.access_token}"},
            timeout=30
        )
        return response.content if response.ok else None
````

这样既不违反框架设计，又能保持良好的性能。Scrapy 的 `ImagesPipeline` 和 `FilesPipeline` 就是这么做的。