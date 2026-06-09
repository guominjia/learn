---
layout: page
title: Categories
permalink: /categories/
description: 按分类自动聚合文章
---

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
