---
layout: page
title: Categories
permalink: /categories/
description: 目录导航 + 文章分类归档
---

## 目录导航

### Core Tracks

- [AI](ai/README.md)
- [Architect](architect/README.md)
- [Algorithm](algorithm/README.md)
- [Math](math/README.md)
- [Python](python/README.md)
- [Database](database/README.md)
- [Web](web/README.md)
- [Security](security/README.md)
- [Linux](linux/README.md)
- [GitHub](github/README.md)

### Systems & Infra

- [CPU](cpu/README.md)
- [GPU](gpu/README.md)
- [Compiler](compiler/README.md)
- [Disk](disk/README.md)
- [Network](network/README.md)
- [Virtualization](virtualization/README.md)
- [xwindow](xwindow/README.md)
- [vnc](vnc/README.md)
- [edk2](edk2/README.md)

### Tools & Engineering

- [Editor](editor/README.md)
- [Bash](bash/README.md)
- [Numpy](numpy/README.md)
- [Scrapy](scrapy/README.md)
- [Documentation](doc/README.md)
- [Visualization](visualization/README.md)
- [Flutter](flutter/README.md)
- [Video](video/README.md)

### Domain Notes

- [Invest](invest/README.md)
- [Research](research/README.md)
- [Robot](robot/README.md)
- [Science](science/README.md)
- [University](university/README.md)
- [Share](share/README.md)
- [Microsoft](microsoft/README.md)

## 文章分类（自动）

{% assign sorted_categories = site.categories | sort %}

{% if sorted_categories.size > 0 %}
  {% for category in sorted_categories %}
    {% assign category_name = category[0] %}
    {% assign category_posts = category[1] %}
    <section class="card archive-group">
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
