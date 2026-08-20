---
layout: post
title: "A Few Famous Tech Blogs"
date: 2026-08-19
categories: [resources]
tags: [blog, performance, ebpf, dtrace, linux, kernel, gamedev]
---

## Brendan Gregg

[Brendan Gregg](https://www.brendangregg.com/) invented [Flame Graphs](https://www.brendangregg.com/flamegraphs.html) and is a major contributor to `bcc`/`bpftrace` (eBPF) and DTrace. Career: Sun/Oracle → Netflix (wrote *Systems Performance*, *BPF Performance Tools*) → Intel (2022–2025, AI Flame Graphs, GPU heatmaps) → OpenAI (since Feb 2026, ChatGPT performance).

His [Leaving Intel](https://www.brendangregg.com/blog/2025-12-05/leaving-intel.html) post (Dec 2025) prompted this note; the follow-up [Why I joined OpenAI](https://www.brendangregg.com/blog/2026-02-07/why-i-joined-openai.html) explains the move.

## Fabien Sanglard

[Fabien Sanglard](https://fabiensanglard.net/) writes line-by-line source-code deep dives on classic game engines — id Software's Wolfenstein 3D and DOOM chief among them — collected into the *Game Engine Black Book* series. Well known in game/graphics programming circles specifically, not a mainstream tech name.

## Gwern Branwen

[Gwern](https://gwern.net/) is a pseudonymous writer/researcher known for [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis) (2020), an early, widely-cited argument that scaling compute/data drives emergent LLM capability. Prolific on LessWrong and Hacker News, and separately known for [This Waifu Does Not Exist](https://www.thiswaifudoesnotexist.net/), a StyleGAN anime-face generator. Site spans AI, statistics, genetics, and nootropics — niche-famous, not mainstream.

## Nikita Prokopov (tonsky)

[Nikita "Niki" Prokopov](https://tonsky.me/) is a Clojure consultant (web, Datomic, DataScript, performance) best known outside that niche as the author of [Fira Code](https://github.com/tonsky/FiraCode), the ligature-enabled programming font (82k+ GitHub stars). His blog regularly hits Hacker News with opinionated essays like [Software disenchantment](https://tonsky.me/blog/disenchantment/) (2018, on software bloat), [The Absolute Minimum Every Software Developer Must Know About Unicode](https://tonsky.me/blog/unicode/) (2023), [JavaScript Bloat in 2024](https://tonsky.me/blog/js-bloat/), and [Claude is an Electron App because we've lost native](https://tonsky.me/blog/fall-of-native/) (2026), plus recurring UI-detail critiques (checkboxes, font sizing, centering).

## Chen Hao (haoel) — CoolShell

[Chen Hao](https://coolshell.cn/haoel) (陈皓, pen name "左耳朵耗子"/haoel) founded [CoolShell](https://coolshell.cn/) in March 2009, one of the earliest influential architecture/systems blogs in the Chinese-speaking developer community, after years of writing on CSDN starting 2003 (his early hit was the *[Makefile tutorial](https://blog.csdn.net/haoel/article/details/2886)* series). He worked as an R&D manager at Amazon China (global e-commerce and inventory forecasting) and at Alibaba, then founded MegaEase, open-sourcing the API gateway [Easegress](https://github.com/megaease/easegress) and the health-check tool [EaseProbe](https://github.com/megaease/easeprobe).

CoolShell posts on distributed systems, Linux/network internals (TIME_WAIT, eBPF), and REST API design routinely went viral — [是微服务架构不香还是云不香？](https://coolshell.cn/articles/22422.html) (2023) drew 526k+ reads, and [一把梭：REST API 全用 POST](https://coolshell.cn/articles/22173.html) (2022) drew 171k+ reads with 129 comments. Chen Hao died suddenly of a heart attack on May 14, 2023; CoolShell's last post is from days before. Fellow blogger Chen Shuo built a static mirror, [coolshell.org](https://coolshell.org/), and wrote a [memorial post](https://www.cnblogs.com/Solstice/p/haoel.html) describing his influence on the Chinese developer community.

## Draveness — 面向信仰编程

[Draveness](https://draven.co/) (GitHub/[draveness](https://github.com/draveness)) has run the blog “面向信仰编程” since 2014, without publishing a real name. Early posts (2015–2016) covered iOS internals (Auto Layout performance, CocoaPods, Masonry/SDWebImage source reading); the focus shifted around 2017 to distributed systems, Kubernetes, and OS/database internals, anchored by the long-running [为什么这么设计](https://draven.co/whys-the-design/) series (MySQL B+ trees, NUMA, DNS-over-UDP, etc.). Author of [《Go 语言设计与实现》](https://draven.co/golang/), a widely-cited Chinese book on the Go runtime/compiler internals. GitHub bio lists Kubernetes/Go/Istio contributions and “HFT / C++ / Go”, based in Beijing.

## Dan Luu

[Dan Luu](https://danluu.com/) has worked as a full-time engineer at Centaur (a CPU design house), Google, Microsoft, and Twitter — per his own footnote comparing how well each company preserved internal documentation. His blog gets [millions of hits a month](https://danluu.com/about/) and is, by his own account, commonly cited by professors and on Stack Overflow. Known for data-driven, empirically-grounded posts spanning CPU/hardware, performance, reliability, and hiring/workplace culture: [Computer latency: 1977-2017](https://danluu.com/input-lag/), [Files are hard](https://danluu.com/file-consistency/), [Why use ECC?](https://danluu.com/why-ecc/), and [We saw some really bad Intel CPU bugs in 2015](https://danluu.com/cpu-bugs/).

## Dan Abramov

[Dan Abramov](https://overreacted.io/) (GitHub/[gaearon](https://github.com/gaearon)) was a longtime member of the React core team at Meta. He's the top contributor to [Redux](https://github.com/reduxjs/redux) and to [Create React App](https://github.com/react/create-react-app), and [reactjs/react.dev](https://github.com/reactjs/react.dev) (the official React documentation site) is pinned on his GitHub profile. His blog *overreacted* covers React internals, JavaScript semantics, and general programming essays; he's currently active on React Server Components and Next.js.

## Julia Evans (b0rk)

[Julia Evans](https://jvns.ca/) worked in infrastructure/performance engineering, including at Stripe (see her [Service discovery at Stripe](https://jvns.ca/blog/2016/10/31/service-discovery-at-stripe/) post), before running [Wizard Zines](https://wizardzines.com/) full-time, publishing illustrated "zines" on Linux, debugging, DNS, Git, networking, and more. Her blog spans DNS, Kubernetes, eBPF/BPF, Rust, and debugging technique, and is a frequent Hacker News fixture. Active on [Mastodon](https://social.jvns.ca/@b0rk), [Bluesky](https://bsky.app/profile/b0rk.jvns.ca), and [GitHub](https://github.com/jvns).

## Rauno Freiberg

[Rauno Freiberg](https://rauno.me/) is an Estonian interaction designer, currently a Staff Design Engineer at [Vercel](https://github.com/vercel) (previously at The Browser Company, working on the Arc browser). He created [Devouring Details](https://devouringdetails.com/), a paid reference/course on UI micro-interaction and motion design used by designers at Apple, OpenAI, Airbnb, and Stripe, and co-runs [History of Software Design](https://historyofsoftware.org/). He's also a maintainer of [cmdk](https://github.com/pacocoursey/cmdk), a widely-used command-menu component. Find him on [X/Twitter](https://x.com/raunofreiberg) and [GitHub](https://github.com/raunofreiberg).

## Ruan Yifeng (阮一峰)

[Ruan Yifeng](https://www.ruanyifeng.com/blog/) has run one of the longest-running Chinese tech blogs, publishing the weekly [科技爱好者周刊](https://www.ruanyifeng.com/blog/weekly/) (408 issues as of Aug 2026, every Friday) alongside practical tutorials on frontend, Git, Linux, and English learning. He authored [《ES6 标准入门》](https://github.com/ruanyf/es6tutorial) (3rd ed., Publishing House of Electronics Industry, 2017), whose open-source source repo has 21k+ GitHub stars. A recognized, long-active independent tech blogger in the Chinese-speaking developer community.

## Lee Robinson

[Lee Robinson](https://leerob.com/) spent 5 years at Vercel (VP of Product, working on Next.js), then moved to [Cursor](https://leerob.com/cursor); per his own current bio, he now works on ML at SpaceX. He's been coding 15 years and "teaching for the second half," writing explainer posts on developer experience, DevRel, and AI coding agents. Active on [X/Twitter](https://twitter.com/leerob) and [GitHub](https://github.com/leerob), where he's a top contributor to [vercel/next.js](https://github.com/vercel/next.js).

## Josh W Comeau

[Josh W Comeau](https://www.joshwcomeau.com/) writes long-form, interactive CSS/React/animation tutorials with live, editable demos embedded in the page — flagship posts include [An Interactive Guide to Flexbox](https://www.joshwcomeau.com/css/interactive-guide-to-flexbox/), [An Interactive Guide to CSS Grid](https://www.joshwcomeau.com/css/interactive-guide-to-grid/), and [Why React Re-Renders](https://www.joshwcomeau.com/react/why-react-re-renders/), all listed among his site's most popular content. He sells two paid courses built the same way, [CSS for JavaScript Developers](https://css-for-js.dev/) and [The Joy of React](https://joyofreact.com/), well regarded in the frontend/React learning community.

## Arthur Heymans

[Arthur Heymans](https://blog.aheymans.xyz/) has contributed to [coreboot](https://review.coreboot.org/q/owner:arthur) since 2016 (payload/chipset ports, ACPI fixes, LTO build support, Rust integration), and his blog documents that work in depth — cache-as-ram internals, hardware root-of-trust, Intel Bootguard, and porting boards like the ThinkPad X60/X61 and ASRock E3C246D4I. He's affiliated with [9elements](https://github.com/9elements), a firmware/coreboot consultancy, and more recently writes about Rust firmware tooling ([NORbert](https://blog.aheymans.xyz/post/norbert/), a SPI NOR flash emulator; [CrabEFI](https://github.com/ArthurHeymans/CrabEFI), a minimal UEFI implementation).