Scrapy 默认不会自动保存爬取的数据到文件。如果要保存 item 结果，有以下几种方法：

## 方法 1: 使用命令行导出（最简单）

运行 Scrapy 时使用 `-o` 参数指定输出文件：


```bash
# 保存为 JSON 格式
scrapy crawl myspider -o items.json

# 保存为 CSV 格式
scrapy crawl myspider -o items.csv

# 保存为 XML 格式
scrapy crawl myspider -o items.xml

# 保存为 JSON Lines 格式（推荐大数据量）
scrapy crawl myspider -o items.jl
```


## 方法 2: 在 settings.py 中配置 Feed Exports


```python
# settings.py
FEEDS = {
    'items.json': {
        'format': 'json',
        'encoding': 'utf8',
        'store_empty': False,
        'overwrite': True,
    },
}
```


## 方法 3: 使用 Item Pipeline（最灵活）

在 `pipelines.py` 中创建自定义 Pipeline