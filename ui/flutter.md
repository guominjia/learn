# Flutter
Flutter User Group

## 评价

**优势**
- **自绘 UI**：不使用系统控件，而是用自带的 Impeller 引擎渲染，各平台外观一致。Impeller 在 iOS 上是唯一的渲染引擎，在 Android API 29+ 上默认启用；从 3.47 起，桌面端也默认启用。Web 端目前仍用 Skia。[1][2]
- **编译与开发体验**：Dart 以 AOT 方式编译成原生机器码；开发期支持 stateful hot reload。[2]
- **平台覆盖**：一套代码覆盖 iOS、Android、Web、Windows、macOS、Linux 和嵌入式设备。[2]
- **生产使用**：Google 自家的 Google Pay、NotebookLM、Google Earth、Google Ads、Classroom、YouTube Create 等都在用；官方 showcase 也列出了阿里、百度、腾讯。阿里工程师称，新功能开发周期从 1 个月缩短到 2 周。[3]

**短板**
- **Dart 小众**：Stack Overflow 2025 调查中，只有 5.9% 的受访者用过 Dart（JavaScript 为 66%，Kotlin 为 10.8%）。[4]
- **包体积**：引擎会带来"几 MB（压缩后）"的基础体积。[2]
- **Web 弱**：官方明确表示 Flutter 不适合以文本为主的静态网站，输出也不利于 SEO。flutter.dev 和 docs.flutter.dev 自己都已迁到基于 DOM 的 Jaspr。[5]
- **新系统设计要重新实现**：因为不用系统控件，系统推出新的设计语言时，需要 Flutter 团队或社区在 Dart 里重新实现[2]。所以"原生新外观"会天然滞后一步——这一点是根据架构做的推断。
- **没有内置热更新**：需要借助第三方方案，例如 Shorebird。[2]
- **团队稳定性**：2024 年 4 月 Google 裁员波及 Flutter/Dart 团队。Google 确认了裁员，但称属于常规重组；Flutter PM 表示 Flutter/Dart 受影响程度与其他团队相当[6]。此后社区出现了分叉项目 Flock。[7]

## 竞争格局

| 方案 | 主导方 | 路线 | 关键进展 |
|---|---|---|---|
| React Native | Meta | JS/React + 原生控件 | 0.76（2024-10）起默认启用新架构 [8] |
| Compose Multiplatform | JetBrains | Kotlin，逻辑和 UI 都可共享 | 1.8.0（2025-05）iOS 版稳定，相比同等 SwiftUI 应用体积多约 9 MB [9] |
| Lynx | 字节 | Markup/CSS + 类 React，原生渲染 | 2025-03 开源，在 TikTok 中大量使用 [10] |
| Kuikly | 腾讯 | KMP，UI 映射到原生控件 | 覆盖 Android/iOS/鸿蒙/Web/小程序；官方称已用于腾讯 30+ 业务、5 亿+ DAU [11] |
| Flutter for OpenHarmony | CPF-Flutter 社区 | Flutter SDK/Engine 的鸿蒙适配版 | 已有 3.35.7 适配版 [12] |

**判断**：Flutter 仍是"自绘引擎"路线的代表，但正在被三股力量分流：
- RN 新架构已经落地；
- Compose Multiplatform 的 iOS 版已经稳定；
- 国内大厂在自研框架（Lynx、Kuikly），其中 Kuikly 直接支持鸿蒙。

Flutter 在鸿蒙上依赖社区维护的适配版，不是 Google 官方支持。

## 盈利模式

Flutter 以 BSD-3-Clause 协议开源，免费使用[13]，Google 不直接从 Flutter 收费。能查证的变现路径有三条：

- **把开发者导向 Google 服务**：Casual Games Toolkit 的模板默认集成 `google_mobile_ads`、`in_app_purchase`、`crashlytics`；接入 Google Cloud、Firebase、Ads 最多可获得 $900 的优惠额度。[14]
- **降低 Google 自家产品的成本**：Google 自己的 App 大量使用 Flutter。[3]
- **第三方商业生态**：

| 公司 | 产品 | 收费（月） |
|---|---|---|
| Shorebird（2023 年由 Flutter 创始人 Eric Seidel 创办）[15] | Code Push、CI | Free / Pro $20 / Business $400 / Enterprise 定制 [16] |
| FlutterFlow | 可视化低代码 | Free / Basic $39 / Growth $80 起 / Business $150 起 [17] |

## References
- <https://flutter.cn/>
- <https://github.com/cfug>
- [1] <https://docs.flutter.dev/perf/impeller>：Impeller 在各平台是否为默认引擎；Web 端仍用 Skia。
- [2] <https://docs.flutter.dev/resources/faq>：自绘而不用系统控件、AOT 编译、平台覆盖、引擎体积、没有内置 code push。
- [3] <https://flutter.dev/showcase>：Google 自家应用与采用 Flutter 的公司（含阿里、百度、腾讯）。
- [4] <https://survey.stackoverflow.co/2025/technology>：Dart 的使用率为 5.9%。
- [5] <https://docs.flutter.dev/platform-integration/web/faq>：Web 不适合文本为主的静态站点和 SEO；官方站点已迁到 Jaspr。
- [6] <https://techcrunch.com/2024/05/01/google-lays-off-staff-from-flutter-dart-python-weeks-before-its-developer-conference/>：2024 年 Flutter/Dart 团队裁员。
- [7] <https://github.com/join-the-flock/flock>：Flock，"A community fork of Flutter"。
- [8] <https://reactnative.dev/blog/2024/10/23/release-0.76-new-architecture>：RN 0.76 默认启用新架构。
- [9] <https://blog.jetbrains.com/kotlin/2025/05/compose-multiplatform-1-8-0-released-compose-multiplatform-for-ios-is-stable-and-production-ready/>：Compose Multiplatform iOS 版稳定，以及体积开销数据。
- [10] <https://lynxjs.org/blog/lynx-unlock-native-for-more.html>：Lynx 开源及其在 TikTok 的使用。
- [11] <https://kuikly.tds.qq.com/>：Kuikly 的技术路线、支持平台和规模（官方自述）。
- [12] <https://gitcode.com/CPF-Flutter/flutter_flutter>：Flutter 的 OpenHarmony 适配版。
- [13] <https://github.com/flutter/flutter/blob/master/LICENSE>：BSD-3-Clause 协议。
- [14] <https://docs.flutter.dev/resources/games-toolkit>：游戏模板集成广告和内购；Google 服务优惠额度。
- [15] <https://shorebird.dev/about>：Shorebird 由 Eric Seidel 于 2023 年创办。
- [16] <https://shorebird.dev/pricing>：Shorebird 定价。
- [17] <https://www.flutterflow.io/pricing>：FlutterFlow 定价。