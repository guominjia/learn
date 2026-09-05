---
title: "Claude Code and GitHub Copilot CLI: Two Very Different Single-File Runtimes"
date: 2026-09-05
categories: [ai, agent]
tags: [claude, copilot, bun, nodejs, reverse-engineering]
---

# Claude Code and GitHub Copilot CLI: Two Very Different Single-File Runtimes

At first glance, `claude.exe` and `copilot.exe` look like the same kind of artifact: a large executable that installs without a separate JavaScript runtime. Our earlier inspection of their Windows binaries already suggested a sharp difference. Claude Code had Bun-shaped sections and JavaScriptCore-related code; Copilot CLI did not.

Several independent reverse-engineering reports now make that conclusion much stronger:

- Claude Code is a Bun single-file executable.
- GitHub Copilot CLI is a Node.js single executable application, or SEA, with V8 and a packaged application payload.

The distinction matters because “single executable” describes distribution, not implementation. Two programs can arrive as one `.exe` while embedding their JavaScript and native code in completely different ways.

## The Short Version

| | Claude Code (`claude.exe`) | GitHub Copilot CLI (`copilot.exe`) |
| --- | --- | --- |
| Runtime packaging | Bun single-file executable | Node.js SEA |
| JavaScript engine | JavaScriptCore | V8 |
| Application payload | Bun bundle embedded in the executable | Loader plus an embedded application archive |
| Native boundary | Bun, JavaScriptCore, and related native components in the executable | Node runtime plus separately loaded native modules; the Copilot analysis documents a native runtime package |
| Useful forensic clues | `.bun` / `__bun`, `BUN_1.2`, JSC/WebKit section names | SEA loader, `copilot.tgz`, Node package layout, no Bun signature |

The last column describes the analyzed Copilot CLI distribution documented by the reverse-engineering project. These details can change between releases, so a binary should be identified before assuming that every version has exactly the same layout.

## Claude Code: Bun in the Binary

### Linux ELF: runtime fingerprints

The most detailed independent analysis examined Claude Code 2.1.38 for Linux, a 223 MB x86-64 ELF executable. The analysis used ordinary binary tools rather than relying on a product label.

The strongest clue was the dynamic symbol table. Filtering `nm -D` output for Bun's version namespace produced 556 N-API symbols marked `@@BUN_1.2`. That is not merely a generic N-API dependency: the symbol version identifies Bun's runtime ABI.

The binary also contains an explicit runtime string:

```text
Bun v1.3.9-canary.51+d5628db23 (Linux x64 baseline)
```

Section names provide a second, independent signal. The ELF contains `__DATA,__jsc_opcodes` and `__DATA,__wtf_config`, names associated with JavaScriptCore and WebKit. Their Mach-O-style spelling inside an ELF is unusual, but it is consistent with the same runtime family visible in the macOS build.

The file layout explains the executable's size. The ELF sections account for roughly 97 MB, while the file is about 223 MB. The analysis found another roughly 115 MB appended after the ELF structure. A syscall trace then observed the process reading that payload from its own executable during startup. In other words, the native half supplies Bun and JavaScriptCore; the appended data supplies the application bundle.

This also matches Bun's documented compilation model. `bun build --compile` bundles imported code and packages together with a copy of the Bun runtime. Bun can also put bytecode and other assets into the standalone executable.

### macOS arm64: the same architecture with clearer names

A separate Ghidra-based analysis examined Claude Code 2.1.214 for macOS arm64, a 236 MB Mach-O binary. Its load commands expose the layout directly:

```text
__text       native runtime and libraries
__bun        embedded application bundle
__cstring    strings, including application source fragments
__jsc_int    JavaScriptCore internal data
```

The `__bun` section is about 181 MB. The binary also contains the marker `---- Bun! ----` and application build metadata such as the Claude Code version and Git SHA. These are independent observations from a different operating system and a different release.

The cross-platform names line up with the earlier Windows inspection: the exact PE section names differ, but the presence of a Bun payload and JavaScriptCore-oriented native data is the same pattern.

### Version tracking makes the result reproducible

The `minzique/claude-code-re` project continuously downloads Claude Code releases, verifies their hashes, extracts the Bun payload with `bun-demincer`, and records signatures across versions. Its README describes roughly 3,000 JavaScript modules recovered per release.

That does not make every extracted interpretation official or automatically correct. It does, however, turn a one-off inspection into a repeatable observation: the CLI is distributed as a Bun-compiled native binary whose application code can be analyzed as an embedded bundle.

## Copilot CLI: Node SEA, Not Bun

The Copilot CLI evidence points to a different design. The internal-analysis wiki identifies the extracted executable as a Node/V8 single executable application. Its distribution layout is described as:

```text
copilot.exe
└── Node SEA loader
	└── embedded copilot.tgz
		└── @github/copilot package
			├── npm-loader.js
			├── index.js
			└── app.js
```

The important word is *loader*. The executable is not a Bun binary with the whole application presented as a Bun bundle. The SEA loader selects or extracts the packaged application and then imports its JavaScript entry point. The wiki also describes the npm and native launcher paths, update handoff, restart behavior, restricted module loading, and configuration migration before the main CLI flow begins.

This is exactly the kind of layout Node's SEA mechanism is designed to support. Node prepares a blob containing a bundled script, injects it into a Node executable, and executes the embedded script when the binary starts. The SEA API also supports embedded assets and native addons, but those remain distinct from the Node/V8 runtime itself.

That explains the second binary we inspected. It had no `.bun` section, only a small set of ordinary PE sections, and a separate `malloc_h`-like native artifact rather than the Bun/JavaScriptCore fingerprints. The absence of a Bun signature is not proof by itself, but it agrees with the positive SEA evidence and with the Node package layout.

## Why the Native Runtime Difference Matters

The two packaging strategies create different boundaries for native functionality.

In Claude Code, the Bun runtime, JavaScriptCore, and the native pieces needed by the executable are part of one Bun-produced artifact. The JavaScript application is bundled into that artifact as well. There may still be runtime-loaded components, but the primary runtime boundary is inside the single binary.

In Copilot CLI, the JavaScript application remains a Node package carried by the SEA loader. Native capabilities are exposed through Node's native-module interfaces and, according to the Copilot internal analysis, an additional native runtime component used by the CLI. That architecture makes a fallback such as “load the native package, otherwise run `node index.js`” entirely natural: the application is still fundamentally a Node application with a JavaScript entry point.

The distinction is useful when debugging or inspecting a release:

1. Search for Bun markers, `BUN_*` symbol versions, `__bun`, and JavaScriptCore data when investigating Claude Code.
2. Search for Node SEA markers, the loader, the embedded archive, and package entry points when investigating Copilot CLI.
3. Treat the JavaScript payload and the native runtime as separate layers in the Copilot case.

Do not infer the runtime from file size. A large executable may contain a JavaScriptCore runtime, a V8 runtime, an archive, bytecode, source strings, native addons, or several of these at once. Section names, symbol namespaces, strings, and observed startup behavior provide a much better identification method.

## Conclusion

The independent analyses confirm the result suggested by our earlier PE inspection:

- Claude Code is a Bun single-file application. Its executable contains Bun and JavaScriptCore, while the application bundle is embedded in Bun-specific sections or an appended payload.
- GitHub Copilot CLI is a Node.js SEA application. Its executable carries a loader and an application archive, which expands into the `@github/copilot` package and runs on Node/V8.

Both tools are “single executable” distributions, but they are not built on the same runtime model. Claude Code is closer to “Bun plus an embedded application.” Copilot CLI is closer to “Node plus an SEA loader and a packaged Node application.” That is the architectural fact that makes the binary differences, the JavaScript fallback path, and the earlier `.bun`-section observations all line up.

## References

- [Reverse Engineering Claude Code](https://pker.xyz/posts/claude.html) - analyzes Claude Code 2.1.38 on Linux and documents the `@@BUN_1.2` symbols, Bun version string, JavaScriptCore-related sections, appended payload, and startup read.
- [Reverse-engineering Claude's CLI with Kimi K3 and Ghidra](https://www.bem.ai/log/reverse-engineering-claude-cli-kimi-k3-ghidra) - analyzes Claude Code 2.1.214 on macOS arm64 and documents the `__bun` section, JavaScriptCore data, Bun marker, and build metadata.
- [Claude Code Monitor](https://github.com/minzique/claude-code-re) - describes automated extraction and version tracking of Claude Code's Bun binaries with `bun-demincer`.
- [Loader and bootstrap workflows](https://copilot-cli.genisisiq.com/01-runtime-lifecycle/loader-bootstrap/) - documents the Copilot CLI SEA loader, embedded `copilot.tgz`, package files, and bootstrap path.
- [Single executable applications](https://nodejs.org/api/single-executable-applications.html) - documents Node's SEA blob, injection model, embedded assets, native addons, and V8-based runtime behavior.
- [Bun single-file executables](https://bun.sh/docs/bundler/executables) - documents `bun build --compile`, bundling the runtime and application into a standalone executable, and embedded bytecode/assets.
