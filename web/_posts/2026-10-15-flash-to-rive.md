---
title: "网页动画：从 Flash 到 Rive"
date: 2026-10-15
categories: [web]
tags: [flash, rive, lottie, animation, flv, video]
---

网页上做动画的几种常见方案：

| 方案 | 特点/局限 |
|---|---|
| CSS / SVG / JS 手写 | 简单动效够用，复杂角色动画难写，设计师也没法直接交付 |
| GIF / 视频 | 文件大、不能缩放、不能交互 |
| Lottie（从 After Effects 导出） | 主要是从头播到尾的线性动画，交互能力弱 |
| Rive | 自带状态机，动画能根据鼠标、点击、数据切换状态；文件是体积很小的二进制；按矢量实时渲染，放大不糊 |

## Adobe Flash

在 2010 年左右，做二维动画一般使用 `Adobe Flash` 工具来做，但 Adobe 在 2020 年底停止支持 Flash Player。`Flash` 被淘汰原因：
- **要装插件**：浏览器必须装 Flash Player 才能播，内容在浏览器的标准体系之外。
- **安全问题多**：长期是漏洞重灾区。
- **耗电、手机不支持**：iPhone 从一开始就不支持，移动端一兴起它就掉队了。

很多老 Flash 游戏和动画还能玩，靠的是后来做的兼容方案。

| 方案 | 怎么用 | 说明 |
|---|---|---|
| [Ruffle](https://ruffle.rs/) | 浏览器扩展，或网站嵌入它的 JS 包；也有 Windows 桌面版 | 开源的 Flash 模拟器，用 Rust 写、编译成 WebAssembly 运行，不用装插件，也没有当年 Flash 那些安全问题。兼容性还没到 100%，大部分老动画和早期游戏能跑 |
| [Flashpoint Archive](https://flashpointarchive.org/) | 下载离线启动器 | 目前最大的网页游戏保存项目，已经存了 20 多万个作品，覆盖 Flash、Shockwave、早期 HTML5 等上百种技术，兼容性最好 |
| [Internet Archive 的 Flash 合集](https://archive.org/details/softwarelibrary_flash) | 直接在网页里点开玩 | 2020 年开始用 Ruffle 在线播放 |
| Newgrounds 等老 Flash 门户 | 照常访问 | 2020 年那会儿 Newgrounds 就在把 Ruffle 接进站里，好让老内容在原地继续能玩 |

另外，中国大陆曾有一个 Flash 中国特供版，由国内公司代理分发，在官方停止支持后还在继续，但当时因为捆绑弹窗广告争议不小。

## Rive

虽然 Flash 没人用了，但 Flash 当年解决的需求还在：设计师在可视化工具里做出能交互的矢量动画，程序员直接拿去用。
Rive 做的正是这件事，常被看作 Flash 的精神继承者。

**Rive 和 Flash 的区别**

| | Flash | Rive |
|---|---|---|
| 运行方式 | 浏览器插件 | 不用装插件，靠浏览器标准的 WebAssembly 和 WebGL/Canvas（`rive.wasm` 加 `<canvas>`） |
| 平台 | 主要是网页 | 网页、iOS、Android、Flutter、Unity、Unreal，大多数平台共用同一套 C++ 运行时 |
| 开放程度 | 播放器闭源 | 运行时开源（MIT） |
| 定位 | 能做整个网站、整个游戏 | 只做 UI 里的动画组件：按钮、加载动画、吉祥物、游戏界面等 |

### 拆解一个 Rive 社区页面

以 Rive 社区里的一个动画页面（文件 `23461-43911-journey`）为例，查看页面源码会发现：

**1. 源码里的 canvas 是空的**

服务端返回的 HTML 里只有这一句：

```html
<div class="CommunityShot_CommunityShotAnimation__YMVrR">
  <canvas style="vertical-align:top;width:0;height:0"></canvas>
</div>
```

宽高是 0，也没有任何绘制代码。页面加载后，Rive 运行时才把它撑大并开始绘制。

**2. 画面内容在一个二进制 `.riv` 文件里**

`<head>` 里预加载了：

```
https://public.rive.app/community/runtime-files/23461-43911-journey.riv
```

形状、骨骼、关键帧、状态机都在这个文件里，不是 JS 代码，所以在源码里搜不到具体图形。

**3. 绘制代码是编译好的 WebAssembly**

`<head>` 里还预加载了：

```
https://unpkg.com/@rive-app/webgl2@2.44.0/rive.wasm
```

- 渲染器是 Rive 的 C++ 运行时编译成的 `.wasm`，用 **WebGL2** 在 GPU 上画，不走 Canvas 2D。所以搜 `getContext('2d')`、`fillRect`、`arc` 之类的都搜不到。
- 页面自己的 JS 大概只负责把 `.riv` 文件交给运行时去播放（类似 `new Rive({ src, canvas, stateMachines })`）。

**想看实现，去这几个地方：**

| 想看什么 | 去哪 |
|---|---|
| JS 封装和 WASM 绑定 | [rive-app/rive-wasm](https://github.com/rive-app/rive-wasm)（MIT） |
| 解析 `.riv`、驱动状态机、WebGL 渲染的 C++ 代码 | [rive-app/rive-runtime](https://github.com/rive-app/rive-runtime)（MIT） |
| 动画本身怎么设计 | 社区页面上的 Preview in Rive / Remix，在 Rive 编辑器里打开看图层和状态机 |
| 浏览器里自己验证 | DevTools → Network，过滤 `.riv` / `.wasm` |

## 做动画用什么工具

| 用途 | 常用工具 | 导出成什么 |
|---|---|---|
| 接 Flash 的班的 2D 动画 | **Adobe Animate**（Flash Professional 改名而来，界面和时间轴基本一样） | HTML5 Canvas、视频、GIF，也还能导出 SWF |
| 专业 2D 动画、番剧 | Toon Boom Harmony、OpenToonz（开源）、TVPaint、Clip Studio Paint、Krita | 视频、图片序列 |
| 动效、片头、UI 动画 | After Effects | 视频；也可以导出成 **Lottie**（JSON 格式），网页和 App 直接播放 |
| 能交互的 UI 和角色动画 | **Rive** | `.riv` 文件，由各平台的运行时实时渲染 |
| 游戏角色骨骼动画 | Spine、DragonBones | 骨骼数据，交给游戏引擎播放 |
| 虚拟主播、立绘动起来 | Live2D | 模型文件 |
| 3D 动画 | Blender（开源）、Maya、Cinema 4D，或在 Unity、Unreal 里做 | 视频或引擎内的资源 |
| 网页上用代码写 | CSS 动画、SVG、GSAP、Canvas/WebGL（PixiJS、Three.js） | 直接就是网页代码 |
| AI 生成 | 各类 AI 视频生成工具 | MP4 |

## 视频格式：FLV 去哪了

FLV 是 Flash Video 的格式。早年的 YouTube、优酷、B 站都靠 Flash 播放器放 FLV。随着 HTML5 video 普及，点播视频的格式基本换掉了，但直播还在用。

| 场景 | 现在用什么 |
|---|---|
| 点播视频，成品文件 | **MP4**（编码多为 H.264，也有 H.265、AV1），网页上也有 **WebM**（VP9、AV1） |
| 视频网站在线播放 | **HLS**（`.m3u8` 加切成小段的视频）或 **DASH**，按网速自动切换清晰度 |
| 直播推流（比如用 OBS 推流） | **RTMP**，载荷基本就是 FLV tag |
| 国内直播拉流 | **HTTP-FLV** 仍很常见，延迟低；B 站开源的 flv.js 用 MSE 把 FLV 转封装成 fMP4，让 HTML5 不靠 Flash 也能播 FLV |
| 短小循环动画 | GIF、APNG、动态 WebP，或者干脆用不带声音的 MP4 / WebM 代替 GIF |