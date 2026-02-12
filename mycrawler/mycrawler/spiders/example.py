import scrapy
from mycrawler.items import MycrawlerItem


class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        title = (
            response.css('title::text').get() or
            response.xpath('//h1/text()').get() or
            response.xpath('//meta[@name="title"]/@content').get() or
            'Untitled'
        )
        
        markdown = response.text
        
        self.result = {
            'url': response.url,
            'title': title.strip(),
            'markdown_len': len(markdown),
            'status_code': response.status,
        }
        
        self.logger.info(
            f"✓ Parsed {response.url}\n"
            f"  Title: {title.strip()}\n"
            f"  Markdown: {len(markdown)} chars"
        )
        
        item = MycrawlerItem()
        item["url"] = response.url
        item["title"] = title.strip()
        #yield self.result
        yield item