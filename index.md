---
layout: default
title: Home
---

<section class="hero">
  <h1>持续学习、持续记录</h1>
  <p>{{ site.description }}。这里整理 AI、系统、工程实践与个人思考，沉淀可复用的知识资产。</p>
</section>

<section class="grid two">
  <article class="card">
    <h2>最近文章</h2>
    <div class="post-list">
      {% assign recent_posts = site.posts | slice: 0, 6 %}
      {% if recent_posts.size > 0 %}
        {% for post in recent_posts %}
          <div class="post-item">
            <div class="meta">{{ post.date | date: "%Y-%m-%d" }}</div>
            <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
            {% if post.excerpt %}
              <div class="meta">{{ post.excerpt | strip_html | truncate: 90 }}</div>
            {% endif %}
          </div>
        {% endfor %}
      {% else %}
        <p class="meta">还没有文章，先去创建你的第一篇 post 吧。</p>
      {% endif %}
    </div>
  </article>

  <article class="card">
    <h2>快速入口</h2>
    <ul>
      <li><a href="{{ '/ai/' | relative_url }}">AI 专栏</a></li>
      <li><a href="{{ '/algorithm/' | relative_url }}">算法笔记</a></li>
      <li><a href="{{ '/linux/' | relative_url }}">Linux 实践</a></li>
      <li><a href="{{ '/database/' | relative_url }}">数据库记录</a></li>
      <li><a href="{{ '/categories/' | relative_url }}">分类归档</a></li>
      <li><a href="{{ '/tags/' | relative_url }}">标签归档</a></li>
    </ul>
    <p class="meta">你可以继续按目录扩展内容，首页会自动聚合最新 post。</p>
  </article>
</section>
