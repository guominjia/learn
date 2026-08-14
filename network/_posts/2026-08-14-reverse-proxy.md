---
layout: post
title: "A Lightweight Reverse Proxy on Windows with Caddy"
date: 2026-08-14
categories: [network]
tags: [caddy, reverse-proxy, windows, http]
---

A reverse proxy accepts requests on one address and sends them to an application running elsewhere. It is useful when an application listens on an inconvenient local port, or when several applications need to appear behind one public HTTP endpoint.

This post shows a small Windows setup using the Caddy executable. The working example listens on port `80` and forwards requests to an application on `127.0.0.1:8765`.

## DNS Is Not a Reverse Proxy

DNS and reverse proxying solve different problems:

| Component | Responsibility |
|---|---|
| DNS server, such as CoreDNS | Maps a name to an address or returns other DNS records |
| Reverse proxy, such as Caddy | Receives HTTP requests and forwards them to an HTTP backend |

For example, DNS can resolve `app.example.test` to `127.0.0.1`. A reverse proxy then receives `http://app.example.test/` and forwards the HTTP request to the application. CoreDNS can proxy DNS queries, but it is not an HTTP reverse proxy.

## Download and Check Caddy

Download the Windows `amd64` binary from the official Caddy download page. It is a standalone command-line executable; no installer is required for the basic workflow.

In this example, the downloaded executable is named `caddy_windows_amd64.exe` and is stored in the Downloads directory:

```powershell
cd ~/Downloads
.\caddy_windows_amd64.exe version
```

The command prints the embedded Caddy version. Renaming the file to `caddy.exe` is optional, but it makes later commands shorter:

```powershell
Rename-Item .\caddy_windows_amd64.exe caddy.exe
```

The remaining examples use the original downloaded name. Substitute `caddy.exe` if it was renamed.

## Create the Caddyfile

Create a file named `Caddyfile` in the same directory as the executable:

```caddyfile
:80 {
	reverse_proxy 127.0.0.1:8765
}
```

`:80` makes Caddy accept HTTP connections on port `80`. The `reverse_proxy` directive names the upstream backend. With no scheme in the upstream address, Caddy uses plaintext HTTP for that connection.

The backend must already be running and listening on `127.0.0.1:8765`. Use its actual host and port if they differ. Do not point the proxy back at its own listening port; that creates a request loop.

## Validate Before Starting

Validate the configuration before binding a port:

```powershell
.\caddy_windows_amd64.exe validate --config .\Caddyfile --adapter caddyfile
```

`validate` loads and provisions the configuration but does not start it. It catches more than a basic syntax conversion check, including errors that occur while modules are being prepared.

Format the Caddyfile when desired:

```powershell
.\caddy_windows_amd64.exe fmt --overwrite .\Caddyfile
```

Start Caddy in the foreground:

```powershell
.\caddy_windows_amd64.exe run --config .\Caddyfile --adapter caddyfile
```

`run` intentionally blocks while Caddy is active. Keep that terminal open while using the proxy. Stop a foreground instance with `Ctrl+C`.

Open `http://localhost/` in a browser, or test it from another PowerShell window:

```powershell
Invoke-WebRequest http://localhost/ -UseBasicParsing
```

## Understanding a 502 Response

A response such as this is a useful diagnostic:

```text
HTTP 502 Bad Gateway
```

In this setup, a `502` means Caddy accepted the browser request but could not obtain a usable response from its upstream. The initial configuration pointed to `127.0.0.1:8080`, where no application was listening. Caddy logged a connection refusal and returned `502`.

Check the intended backend directly before changing proxy settings:

```powershell
Invoke-WebRequest http://127.0.0.1:8765/ -UseBasicParsing
Get-NetTCPConnection -State Listen |
	Where-Object LocalPort -eq 8765
```

If the first command fails, fix or start the backend. If it succeeds but `http://localhost/` fails, check the upstream address in `Caddyfile`, then inspect the terminal where Caddy is running for its error message.

## Reload a Changed Configuration

After changing the upstream address, validate first and reload the running Caddy process:

```powershell
.\caddy_windows_amd64.exe validate --config .\Caddyfile --adapter caddyfile
.\caddy_windows_amd64.exe reload --config .\Caddyfile --adapter caddyfile
```

`reload` sends the new configuration to Caddy's local administration endpoint. It is the appropriate command for changing the running configuration; stopping and starting the proxy to apply a file edit introduces avoidable downtime.

For the working configuration in this example, a request to `http://localhost/` returned `HTTP 200` after the backend was available on port `8765` and Caddy was reloaded.

## Run It Persistently

Running `caddy run` in a terminal is convenient for local development. Closing that terminal ends the foreground Caddy process. For a persistent Windows installation, Caddy documents two service approaches: register a service with `sc.exe`, or use the WinSW service wrapper.

`sc.exe` registers a command with the Windows Service Control Manager (SCM). A process that can run continuously is a candidate for service-style startup, but staying alive is not enough to make it a reliable Windows service. A native Windows service also reports its state to SCM and handles service control requests such as stop and pause.

| Program type | Recommended approach |
|---|---|
| Native Windows-service-aware program | Register it with `sc.exe` |
| Long-running command-line program without service support | Use a wrapper such as WinSW or NSSM |
| GUI, one-shot, or interactive program | Do not run it as a service |

Service processes run independently of the signed-in user's desktop session. Use absolute paths, configure a working directory and log location, and do not rely on mapped drives, prompts, relative paths, or the current user's environment variables.

The minimal `sc.exe` pattern from the Caddy documentation is:

```powershell
sc.exe create caddy start= auto binPath= "C:\Tools\caddy\caddy.exe run"
sc.exe start caddy
```

Use an absolute path and make sure the service account can read the executable and its `Caddyfile`. A service needs a deliberate directory layout and log-handling plan, so test the foreground configuration first.

## Takeaway

Caddy is a practical lightweight HTTP reverse proxy for a Windows development machine: download one executable, write a short `Caddyfile`, validate it, and run it. A `502 Bad Gateway` is not a Caddy startup failure by itself; it commonly means that the proxy is listening correctly while the configured upstream is unavailable. Once the backend responds directly, reload Caddy and verify the public proxy endpoint.

## References

- [Caddy reverse_proxy directive](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy) - Defines upstream address syntax, HTTP transport defaults, forwarded headers, and reverse-proxy examples.
- [Caddy command line](https://caddyserver.com/docs/command-line) - Documents `validate`, `fmt`, `run`, and `reload`, including that reload is the semantic operation for changing a running configuration.
- [Caddy: Keep Caddy Running](https://caddyserver.com/docs/running) - Documents Windows service installation with `sc.exe` and WinSW.
