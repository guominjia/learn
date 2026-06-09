---
layout: page
title: Tags
permalink: /tags/
description: 按标签自动聚合文章
---

{% assign sorted_tags = site.tags | sort %}

{% if sorted_tags.size > 0 %}
  {% for tag in sorted_tags %}
    {% assign tag_name = tag[0] %}
    {% assign tag_posts = tag[1] %}
    <section class="card archive-group">
      <h2>#{{ tag_name }} <span class="meta">({{ tag_posts.size }})</span></h2>
      <ul>
        {% for post in tag_posts %}
          <li>
            <span class="meta">{{ post.date | date: "%Y-%m-%d" }}</span>
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </li>
        {% endfor %}
      </ul>
    </section>
  {% endfor %}
{% else %}
  <p class="meta">暂无标签数据，给文章添加 front matter 字段 `tags` 即可自动生成。</p>
{% endif %}
