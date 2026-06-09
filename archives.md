---
layout: page
title: Archives
permalink: /archives/
description: 按时间浏览全部文章
---

{% if site.posts.size > 0 %}
  <ul>
    {% for post in site.posts %}
      <li>
        <span class="meta">{{ post.date | date: "%Y-%m-%d" }}</span>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </li>
    {% endfor %}
  </ul>
{% else %}
  <p class="meta">暂无文章。</p>
{% endif %}
