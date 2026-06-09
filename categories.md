---
layout: page
title: Categories
permalink: /categories/
description: 目录导航 + 文章分类归档
---

## 目录导航

### Core Tracks

- [AI]({{ '/ai/' | relative_url }})
- [Architect]({{ '/architect/' | relative_url }})
- [Algorithm]({{ '/algorithm/' | relative_url }})
- [Math]({{ '/math/' | relative_url }})
- [Python]({{ '/python/' | relative_url }})
- [Database]({{ '/database/' | relative_url }})
- [Web]({{ '/web/' | relative_url }})
- [Security]({{ '/security/' | relative_url }})
- [Linux]({{ '/linux/' | relative_url }})
- [GitHub]({{ '/github/' | relative_url }})

### Systems & Infra

- [CPU]({{ '/cpu/' | relative_url }})
- [GPU]({{ '/gpu/' | relative_url }})
- [Compiler]({{ '/compiler/' | relative_url }})
- [Disk]({{ '/disk/' | relative_url }})
- [Network]({{ '/network/' | relative_url }})
- [Virtualization]({{ '/virtualization/' | relative_url }})
- [xwindow]({{ '/xwindow/' | relative_url }})
- [vnc]({{ '/vnc/' | relative_url }})
- [edk2]({{ '/edk2/' | relative_url }})

### Tools & Engineering

- [Editor]({{ '/editor/' | relative_url }})
- [Bash]({{ '/bash/' | relative_url }})
- [Numpy]({{ '/numpy/' | relative_url }})
- [Scrapy]({{ '/scrapy/' | relative_url }})
- [Documentation]({{ '/doc/' | relative_url }})
- [Visualization]({{ '/visualization/' | relative_url }})
- [Flutter]({{ '/flutter/' | relative_url }})
- [Video]({{ '/video/' | relative_url }})

### Domain Notes

- [Invest]({{ '/invest/' | relative_url }})
- [Research]({{ '/research/' | relative_url }})
- [Robot]({{ '/robot/' | relative_url }})
- [Science]({{ '/science/' | relative_url }})
- [University]({{ '/university/' | relative_url }})
- [Share]({{ '/share/' | relative_url }})
- [Microsoft]({{ '/microsoft/' | relative_url }})

## 文章分类（自动）

{% assign sorted_categories = site.categories | sort %}

{% if sorted_categories.size > 0 %}
{% for category in sorted_categories %}
{% assign category_name = category[0] %}
{% assign category_posts = category[1] %}
<section class="card archive-group" id="{{ category_name | slugify }}">
  <h2>{{ category_name }} <span class="meta">({{ category_posts.size }})</span></h2>
  <ul>
    {% for post in category_posts %}
    <li>
      <span class="meta">{{ post.date | date: "%Y-%m-%d" }}</span>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
  </ul>
</section>
{% endfor %}
{% else %}
<p class="meta">暂无分类数据，给文章添加 front matter 字段 `categories` 即可自动生成。</p>
{% endif %}
