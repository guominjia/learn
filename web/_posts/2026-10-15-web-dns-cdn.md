---
title: "网站背后的原理：DNS、CDN"
date: 2026-10-15
categories: [web]
tags: [domain, dns, cdn]
---

想把网页发布出去让所有人看到，通常需要两样东西：1）一个域名；2）一台能对外提供网页的主机，比如虚拟主机或虚拟专用服务器（VPS）。如果只是静态博客，GitHub Pages 这类托管服务就能代替主机。

本文不讲具体怎么操作，而是讲讲这背后的原理。

## 域名如何与主机映射

申请主机后，你会得到一个公网 IP，比如 `x.x.x.x`；同时假设你有一个域名 `domain.com`。你需要在**域名的 DNS 托管商**那里添加记录：

```
domain.com.      A      x.x.x.x
www.domain.com.  A      x.x.x.x     ; 也可以 CNAME 到 domain.com
```

- DNS 托管商默认通常就是你的注册商，也可以换成 Cloudflare、DNSPod 等。
- `domain.com` 和 `www.domain.com` 是两个名字，各需要一条记录。

用户在浏览器输入 `www.domain.com` 时，浏览器从缓存或递归解析器拿到这条映射，于是知道去 `x.x.x.x` 取你放在主机上的网页。

## 域名解析过程

**完整流程（以 `www.domain.com` 为例）**

```
浏览器 / 操作系统缓存
   │ 没有
   ▼
递归解析器（运营商 DNS、8.8.8.8、1.1.1.1 等）
   │ 先查自己的缓存；没有，就自己去逐级问 ↓
   │
   ├─① 问根服务器 (.)        "www.domain.com 在哪？"
   │     ← "我不知道，.com 归这几台管：a.gtld-servers.net …"
   │
   ├─② 问 .com 注册局服务器  "www.domain.com 在哪？"
   │     ← "我不知道具体记录，domain.com 的 NS 是 xxx.dns-provider.com"
   │
   ├─③ 问 domain.com 的权威 DNS（你托管 DNS 的那家，比如 Cloudflare、DNSPod）
   │     ← "www.domain.com A x.x.x.x"
   ▼
把结果返回给浏览器，并按 TTL 缓存
```

**几个容易混淆的点**

| 误解 | 实际情况 |
| --- | --- |
| 查询会经过注册商 | 不经过。注册商只负责把你设置的 NS 写进注册局的数据库，解析时完全不参与 |
| 每次都要问到根 | 很少。各级结果都按 TTL 缓存，`.com` 的 NS 这类记录几乎一直在解析器的缓存里 |

**域名解析器**通过 DNS 协议（UDP/TCP 53 端口）去问各级**权威服务器**；权威服务器则根据自己加载的**区数据**作答。

```
递归解析器 ──DNS 查询──▶ 根权威服务器      ← 加载"根区"
           ──DNS 查询──▶ .com 权威服务器    ← 加载".com 区"（注册局发布）
           ──DNS 查询──▶ domain.com 权威   ← 加载"domain.com 区"（你的 DNS 服务商）
```

**每一级的区里存的是什么**

| 区 | 由谁维护 | 内容 | 对查询的回答 |
| --- | --- | --- | --- |
| 根区 `.` | IANA/ICANN | 各后缀的 NS 记录和 glue | "去问 `.com` 的这几台服务器" |
| `.com` 区 | Verisign | 每个 `xxx.com` 的 NS 记录（委派）和必要的 glue | "domain.com 去问这几台 NS" |
| `domain.com` 区 | 你的 DNS 服务商 | A、AAAA、CNAME、MX、TXT 等具体记录 | "www.domain.com A x.x.x.x" |

**几点补充**
- **区文件只是一种格式**：传统上是 RFC 1035 定义的文本格式（BIND 风格）。实际上权威服务器把区数据加载进内存作答，大的注册局和 DNS 服务商一般存在数据库里，再动态生成或增量同步，不一定真有一个 `.zone` 文件。
- **上一级只存委派，不存下一级的具体记录**：`.com` 区里只有 `domain.com` 的 NS，没有 `www.domain.com` 的 CNAME。具体记录在你的 DNS 服务商那里。
- **hold 改变的是 `.com` 区**（hold 见下文"域名停用"）：注册局生成 `.com` 区时跳过带 hold 状态的域名，于是第 ② 步不再返回 NS，链条就断了。你 DNS 服务商那边的 `domain.com` 区数据还在，只是没人能找到它。

来源：[Cloudflare – What is DNS?](https://www.cloudflare.com/learning/dns/what-is-dns/)

## 域名注册生效

**从注册到能解析，大致经过这几步**

```
① 注册商发 <create>（带上 NS）→ 注册局数据库，立即生效，WHOIS 马上能查到
② 注册局把变更发布到 DNS 区 → 发布节奏由各注册局决定，不同后缀差别很大
③ 区文件同步到该后缀的所有权威服务器（全球分布）
④ 各地递归解析器查询时拿到新的 NS → 受缓存影响
```

**为什么有时要等好几个小时**

| 原因 | 说明 |
| --- | --- |
| **否定缓存** | 注册之前，如果有人（包括你自己）查过这个域名，递归解析器会缓存"不存在"的结果（NXDOMAIN），缓存时长取后缀 SOA 记录的 TTL 与其 MINIMUM 字段中较小的那个（RFC 2308）。过期前，这个解析器一直认为域名不存在 |
| **注册局发布节奏** | 有些后缀是批量发布的，或者新注册还要人工审核。比如国内注册商注册后要先过实名审核和命名审核（命名审核通常 1–3 个工作日），未通过实名审核时域名处于 serverHold，不能解析 |
| **NS 变更** | 不是新注册，而是更换 NS 时，旧的委派记录在递归解析器里通常缓存较长时间，切换要慢一些 |
| **你自己 DNS 服务商的记录** | 下一级的 A/CNAME 记录要在你的 DNS 服务商那里配置好；它们的 TTL 和生效速度由这家服务商决定 |

**实际建议**
- 注册前尽量**别反复查询这个域名能否解析**，以免在常用的解析器里留下否定缓存。
- 验证是否生效时，直接问权威服务器，比如 `dig @a.gtld-servers.net example.com NS`，这样能绕开缓存。

来源：[RFC 2308 – Negative Caching of DNS Queries](https://www.rfc-editor.org/rfc/rfc2308)、[腾讯云 – 域名实名认证](https://cloud.tencent.com/document/product/242/6707)

## 域名停用

**域名被停用是怎么发生的**

注册局或注册商给域名打上 hold 状态，注册局发布区数据时就不再发布它的 NS 记录（以 `.com` 为例，就是从 `.com` 区里撤掉），于是解析链在第 ② 步就断了。
- 注册商打的叫 `clientHold`，注册局打的叫 `serverHold`。
- 腾讯云文档写的是"注册局暂停解析（Serverhold）"，实名审核通过后 48 小时内恢复解析。不管哪一种 hold，结果都是全球都解析不到。

**为什么在国外注册商买就不用实名**

| | 国内注册商（阿里云、腾讯云等） | 海外注册商（Cloudflare、Porkbun 等） |
| --- | --- | --- |
| 受谁监管 | 工信部（《互联网域名管理办法》） | ICANN 合同 + 当地法律 |
| 实名要求 | 必须实名，否则停解析 | 只要求 ICANN 规定的 WHOIS 联系信息，不要求中国身份证核验 |
| 谁能把域名 hold 住 | 注册商自己，或配合国内注册局（如 .cn） | 注册商或注册局，一般只在违反它们的规则时 |

所以 `.com` 在海外注册商买时，中国的实名规定管不到那家注册商，它也就没理由停你的解析。

**需要注意的边界**
- **`.cn`、`.中国` 这类国内后缀**的注册局是 CNNIC，在中国境内，不管通过谁买都绕不开实名。
- **能解析不等于国内能访问**。服务器放在国内就仍然要做 ICP 备案；放在海外，能不能从国内访问还要看跨境线路和网络管控。
- 域名后续可以转回国内注册商，但转入时同样要先完成实名。

来源：[腾讯云 – 域名实名认证](https://cloud.tencent.com/document/product/242/6707)、[ICANN – EPP Status Codes](https://www.icann.org/resources/pages/epp-status-codes-2014-06-16-en)

## 域名状态查询

WHOIS（以及它的替代协议 RDAP）查的是**注册局和注册商的注册数据库**，DNS 解析用的是**注册局发布的 DNS 区文件**，两者是分开的。

```
注册局数据库（域名对象：持有人、注册商、到期时间、状态…）
 ├─→ WHOIS / RDAP 查询  ← 直接读数据库，hold 状态的域名照样能查到
 └─→ 生成 DNS 区文件    ← 带 hold 状态的域名，NS 记录不会被发布进去
```

**被 hold 的域名，WHOIS 里能看到什么**

| 字段 | 内容 |
| --- | --- |
| Domain Name | 能查到，说明域名已被注册 |
| Registrar | 托管注册商，比如阿里云、腾讯云 |
| 注册、到期时间 | 正常显示 |
| Domain Status | 会显示 `serverHold` 或 `clientHold` |
| Name Server | 可能显示你设置的 NS，但这些 NS 并没有被发布到 DNS 里 |
| 持有人信息 | 多半被隐藏（GDPR 和注册商的隐私保护），只显示 "REDACTED" 之类 |

- **想排查为什么打不开**，先查一下 WHOIS 的 Status 字段，就能看出是不是被 hold 了。
- **查询方式**：
  - 命令行：`whois example.com`
  - 网页：ICANN Lookup（https://lookup.icann.org/ ，基于 RDAP）
  - 注册商自己的 WHOIS 页面

来源：[ICANN – EPP Status Codes](https://www.icann.org/resources/pages/epp-status-codes-2014-06-16-en)

## 注册与删除

**新注册（`<create>`）**
- 注册商先发 `<check>` 查询，已存在的域名会返回 `avail="0"`（不可注册）。
- 即使跳过 check 直接 `<create>`，注册局也会因为对象已经存在而拒绝。
- 注册成功后，**发起注册的那家就成了托管注册商**（`clID`），以后只有它能修改这个域名。

所以**别人不能抢注已存在的域名**，只能等它被删除、释放后再注册。

**删除（`<delete>`）**

| 谁 | 能否删除 |
| --- | --- |
| 其他注册商 | ❌ 和改状态一样，返回 2201（权限不足） |
| 托管注册商 | ✅ 前提是域名没有 `clientDeleteProhibited` 或 `serverDeleteProhibited` |
| 注册局 | ✅ 按自己的政策，比如法律纠纷、违规 |

**hold 和删除的区别**

| | hold | 删除 |
| --- | --- | --- |
| 解析 | 停止 | 停止 |
| 域名归属 | 还是你的 | 进入回收流程 |
| 能否恢复 | 解除 hold 后恢复，但需要等发布和缓存过期 | 要在赎回期内赎回 |
| 常见场景 | 未实名、欠费、纠纷期间 | 到期不续费、持有人主动删除、违规 |

**删除后的流程**（按 ICANN 的说明，适用于通用顶级域；到期未续费的域名一般先有一段续费宽限期，各后缀政策不同）

```
注册商发 delete
 → redemptionPeriod（30 天，原持有人可通过注册商赎回，通常要额外付费）
 → pendingDelete（约 5 天，不能再赎回）
 → 从注册局数据库清除，任何人都可以重新注册
```

所以注册商和注册局通常**先 hold**：解析先停掉，归属不变，问题解决后就能恢复。删除是不可轻易撤回的最后手段。

来源：[RFC 5731](https://www.rfc-editor.org/rfc/rfc5731)、[ICANN – EPP Status Codes](https://www.icann.org/resources/pages/epp-status-codes-2014-06-16-en)

## 注册商能改什么

EPP 协议规定，**注册商只能修改自己托管的域名**。

**协议原文（RFC 5731 §2.3、§3.2，EPP 的正式标准）**

> Status values can be set only by the client that sponsors a domain object and by the server on which the object resides.
>
> 状态值只能由托管该域名的注册商（sponsoring client）和注册局服务器设置。

> Any attempt to transform an object by an unauthorized client MUST be rejected, and the server MUST return a 2201 response code.
>
> 未授权的注册商对域名对象做任何修改，都必须被拒绝，服务器返回 2201（权限不足）。

**注册局怎么判断**

每个域名对象在注册局里都记着它的托管注册商（`<domain:clID>`）。注册局收到 `<update>` 时会比对：

```
注册商 A 发来: <update> domain.com 添加 clientHold
注册局: domain.com 的 clID = 注册商 B ≠ A → 返回 2201，拒绝
```

**其他保护层**

| 层面 | 作用 |
| --- | --- |
| **认证** | 注册商用自己的账号登录注册局的 EPP 系统，注册局据此知道是哪家在操作 |
| **转移要授权码** | 想把别家的域名转到自己名下，必须提供域名的 authInfo（转移码），由托管注册商提供给持有人 |
| **注册局锁** | 部分注册局提供 `serverUpdateProhibited` 等 server 级锁，连托管注册商自己也改不了 |
| **合同和审计** | 注册商要和注册局、ICANN 签合同，越权操作会被追责，甚至被取消资质 |

**结论**：注册商的写权限是**按域名隔离**的，不是"能写库就能写任何记录"。真正能 hold 你域名的，还是只有你的托管注册商和该后缀的注册局。

来源：[RFC 5731 – EPP Domain Name Mapping](https://www.rfc-editor.org/rfc/rfc5731)

## 注册局能改什么

**每个注册局只能操作自己那个后缀的数据库**，管不到别的后缀。

```
根区 (.)              ← IANA/ICANN 管理：只记录各个后缀由谁负责，不记录具体域名
 ├─ .com 区           ← Verisign 管理：所有 xxx.com 的 NS 记录都在这里
 │   └─ learn.com
 ├─ .cn 区            ← CNNIC 管理：所有 xxx.cn 的 NS 记录都在这里
 └─ .io 区            ← .io 的注册局管理
```

| 谁 | 能 hold 什么 |
| --- | --- |
| Verisign（.com 注册局） | 任意 `xxx.com` |
| CNNIC（.cn 注册局） | 任意 `xxx.cn`，**动不了 `learn.com`** |
| 根区管理方 | 理论上可以撤掉一整个后缀的委派（比如让 `.com` 整体失效），但不会针对单个二级域名操作 |

一个域名被 hold 只有两个来源：**它所属后缀的注册局**（serverHold），以及**托管它的注册商**（clientHold）。第三方注册局或注册商在 EPP 和 DNS 区数据层面都没有权限。

这也是选后缀时要考虑的一点：域名的"命门"握在它所属后缀的注册局手里，而注册局受所在国法律约束。`.com` 对应美国，`.cn` 对应中国，`.io` 等国家后缀则对应各自的管理方。

## 域名污染

有些网站在特定网络里访问不了，DNS 层面常见的一种手段就是域名污染。

域名污染（DNS 污染，也叫 DNS 注入）指在你查询域名的网络路径上，有设备抢先发回一个伪造的答案。它不改注册局、注册商或权威服务器上的任何数据，只是在路上"截胡"。

**为什么能做到**
- 默认情况下，DNS 查询和回答经 UDP 明文传输，路径上的设备能看到你在查哪个域名。
- 解析器靠事务 ID（16 位）、端口和问题来匹配回答，并不校验回答是否真的来自目标服务器。格式对得上的回包里，先到的那个会被采用。

**大致过程**

```
你 ──查询 blocked.com──▶ 出境链路 ──▶ 境外解析器（如 8.8.8.8）
                │
          链路上的设备监听到黑名单域名
                │
                └──抢先伪造回包：blocked.com = 假 IP ──▶ 你（先到，被采信）
                                     真实回包稍后到达 ──▶ 你（被丢弃）
```

因为是在路上注入，即使你用境外的公共 DNS，只要查询经过这段链路，也可能收到假答案。

**和几个相近概念的区别**

| 名称 | 在哪一层做手脚 | 谁做 |
| --- | --- | --- |
| 域名污染（注入） | 查询途经的网络路径 | 链路上的设备 |
| 缓存投毒 | 递归解析器的缓存（往缓存里塞假记录，例如 Kaminsky 攻击） | 攻击者 |
| DNS 劫持 | 你使用的解析器直接返回篡改过的结果 | 运营商或路由器 |
| serverHold / clientHold | 注册局，不发布委派 | 注册局或注册商 |

**怎么应对**

| 方法 | 效果 |
| --- | --- |
| DoH / DoT（加密 DNS） | 查询和回答都加密，路上的设备看不到内容，也无法伪造或篡改。不过加密 DNS 服务器本身可能被封锁 |
| DNSSEC | 用签名校验数据真伪，能识别伪造的回答，但不加密；遇到伪造回包时通常只能解析失败，拿不到正确答案 |

来源：[Cloudflare – DNS over TLS vs. DNS over HTTPS](https://www.cloudflare.com/learning/dns/dns-over-tls/)、[Wikipedia – DNS spoofing](https://en.wikipedia.org/wiki/DNS_spoofing)

## CDN

用了 CDN 之后，**域名不再解析到你的源站 IP，而是解析到 CDN 的边缘节点**。边缘节点按需从源站拉取内容并缓存，不会把整个站点预先 dump 一份。

**不加 CDN**

```
访客 → DNS: www.domain.com → A x.x.x.x
```

**加 CDN 后**

```
访客 → DNS → CDN 边缘节点 IP（离访客近的那个）
            ├─ 缓存命中 → 直接返回，不碰源站
            └─ 未命中   → 回源到 x.x.x.x 拉取 → 缓存 → 返回
```

**域名解析怎么变**

| 接入方式 | 做法 | 解析结果 |
| --- | --- | --- |
| Cloudflare（NS 接入） | 把域名的 NS 改到 Cloudflare，记录开启代理（橙色云朵） | 返回 Cloudflare 的 Anycast IP。全球用同一组 IP，由路由把请求带到最近的机房 |
| 国内 CDN（CNAME 接入） | `www.domain.com` CNAME 到 CDN 给你的一个域名，例如 `xxx.cdn.example.com` | CDN 自己的 DNS 按访客位置返回最近节点的 IP（GeoDNS） |

源站 IP 仍然固定，只是不再直接暴露给访客，CDN 回源时才用到它。

**缓存怎么工作**
- **按需拉取（pull）**：某个 URL 第一次被请求时才回源，之后在缓存有效期（TTL）内都由边缘节点直接返回。
- **能缓存什么**：静态文件（HTML、CSS、JS、图片）最适合缓存。动态请求（登录、API）一般不缓存，直接透传回源站。
- **更新内容后**：可以等 TTL 过期，也可以在 CDN 控制台手动清除缓存（purge）。

**对你的网站有什么用**
- **快**：访客从附近的节点拿内容，不必每次都跨洋到源站。前提是 CDN 在访客附近有节点；Cloudflare 免费版不包含中国内地网络，所以国内访客不一定变快。
- **省源站流量**：大部分请求被缓存挡掉，不容易碰到流量限制。
- **抗攻击、隐藏源站**：DDoS 流量先打到 CDN，而不是源站。
- **附加功能**：HTTPS 证书、跳转规则、防火墙规则等可以统一在 CDN 这一层配置。

| 方案 | 费用 | 适合 |
| --- | --- | --- |
| **Cloudflare 免费版** | $0/月 | 含 DNS、CDN、Universal SSL、不限量的 DDoS 防护和基础 WAF |
| 国内 CDN（阿里云、腾讯云） | 按流量收费 | 加速区域包含中国内地时，域名**必须先做 ICP 备案** |

**什么时候值得加 Cloudflare 免费版**
- 想把 DNS 托管和 HTTPS 统一在一处管理。
- 流量接近 GitHub Pages 的限额。GitHub 文档里，对超限站点给的第一条建议就是在前面加第三方 CDN。
- 想要防护或缓存规则，免费版有 70 条规则可用。

来源：[Cloudflare – What is a CDN?](https://www.cloudflare.com/learning/cdn/what-is-a-cdn/)、[Cloudflare Network & CDN Pricing](https://www.cloudflare.com/plans/network-cdn/)、[GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)