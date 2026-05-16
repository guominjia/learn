---
title: "Why Chrome Automatically Redirects HTTP to HTTPS — and How to Bypass It"
date: 2025-05-16
tags: [web, security, https, hsts, chrome]
---

# Why Chrome Automatically Redirects HTTP to HTTPS — and How to Bypass It

If you've ever typed `http://example.com` into Chrome's address bar only to find yourself on `https://example.com`, you're not alone. Chrome aggressively promotes HTTPS for good reason — but sometimes, especially during development or debugging, you need to access the plain HTTP version. This post explains the mechanisms behind the automatic redirect and provides practical methods to bypass it.

---

## 1. Why Does Chrome Force HTTPS?

### 1.1 HSTS (HTTP Strict Transport Security)

HSTS is a web security policy mechanism that allows a server to declare, via a response header, that browsers should only interact with it over HTTPS. A typical header looks like:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

Once a browser receives this header, it will **automatically upgrade all future HTTP requests** to HTTPS for that domain — even if the user explicitly types `http://`. This effectively prevents man-in-the-middle (MITM) attacks that exploit the initial insecure HTTP request.

### 1.2 HSTS Preload List

Chrome ships with a **hardcoded preload list** of domains that must always use HTTPS. Major sites like Google, Facebook, and Twitter are on this list. For these domains, Chrome enforces HTTPS on the very first visit — no prior server response is needed.

You can check if a domain is on the preload list at [hstspreload.org](https://hstspreload.org/).

### 1.3 "Not Secure" Warnings and Mixed Content Blocking

Chrome marks any HTTP page as **"Not Secure"** in the address bar. It also blocks **mixed content** — HTTP sub-resources (scripts, images, iframes) loaded on an HTTPS page — to prevent attackers from injecting malicious assets into an otherwise secure page.

### 1.4 Omnibox Autocomplete

Chrome's address bar (Omnibox) remembers your browsing history. If you've previously visited the HTTPS version of a site, Chrome may autocomplete `http://` to `https://` before the request is even sent.

---

## 2. How to Bypass the Automatic HTTPS Redirect

> **Warning:** Bypassing HTTPS exposes you to real security risks, including credential theft and session hijacking. Only use these methods in controlled environments such as local development or testing.

### Method 1: Clear the HSTS Cache

If the redirect comes from a previously received HSTS header (not from the preload list), you can clear it directly in Chrome:

1. Navigate to `chrome://net-internals/#hsts`.
2. Under **Delete domain security policies**, enter the domain (e.g., `example.com`).
3. Click **Delete**.
4. Optionally, under **Query HSTS/PKP domain**, verify the domain entry is gone.

After clearing, Chrome will no longer force HTTPS for that domain until it receives the header again.

### Method 2: Use Incognito Mode

Incognito mode starts with a clean slate — no HSTS cache, no cookies, no browsing history autocomplete:

| OS            | Shortcut          |
|---------------|-------------------|
| Windows/Linux | `Ctrl+Shift+N`   |
| macOS         | `Cmd+Shift+N`    |

Type the full `http://` URL in the incognito window. Note that **preloaded domains will still redirect** even in incognito mode.

### Method 3: Launch Chrome with Flags

You can start Chrome with command-line flags to disable transport security features:

**Disable HSTS enforcement:**

```bash
# Windows
chrome.exe --disable-hsts

# macOS
open -n -a "Google Chrome" --args --disable-hsts

# Linux
google-chrome --disable-hsts
```

**Disable HTTPS redirect with a fresh profile:**

```bash
chrome.exe --user-data-dir="C:\Temp\ChromeProfile" --disable-features=TransportSecurity
```

This creates a temporary browser profile, leaving your main profile untouched.

### Method 4: Edit the Hosts File

For targeted testing, map the domain directly to an IP address in your system's hosts file. This bypasses DNS but **does not bypass HSTS** — Chrome will still attempt HTTPS if the domain is in HSTS cache or the preload list.

| OS            | Hosts File Path                      |
|---------------|--------------------------------------|
| Windows       | `C:\Windows\System32\drivers\etc\hosts` |
| macOS / Linux | `/etc/hosts`                         |

Add an entry like:

```text
192.168.1.100  example.com
```

This is useful when you need to point a domain to a local or staging server.

### Method 5: Type the Full HTTP URL Manually

Sometimes the fix is simple: type `http://example.com` in full and **press Enter before the autocomplete kicks in**. Chrome's Omnibox aggressively suggests HTTPS URLs, so you need to be deliberate.

Alternatively, paste the full `http://` URL from a text editor to avoid autocomplete interference.

### Method 6: User IP address

Chrome support IP address url with http support

---

## 3. Important Caveats

| Scenario | Note |
|----------|------|
| **Preloaded domains** | Domains on the HSTS preload list (e.g., google.com) **cannot** be bypassed by clearing cache or using incognito. The enforcement is baked into Chrome's source code. |
| **Certificate errors** | If a site's TLS certificate is expired or misconfigured, Chrome will show `NET::ERR_CERT_*` errors. Fixing this requires updating the certificate on the server side. |
| **Security risk** | Accessing HTTP pages on untrusted networks exposes all traffic — including cookies, form data, and credentials — in plaintext. |

---

## 4. Summary

Chrome's HTTP-to-HTTPS redirect is driven by three main mechanisms:

| Mechanism | Scope | Bypassable? |
|-----------|-------|--------------|
| HSTS response header | Per-domain, cached | Yes — clear HSTS cache |
| HSTS preload list | Hardcoded in browser | No — compiled into Chrome |
| Omnibox autocomplete | Based on history | Yes — type full URL or use incognito |

**For development and testing**, the most practical approaches are:

- **Clear the HSTS cache** via `chrome://net-internals/#hsts`
- **Use incognito mode** for a session without cached policies
- **Launch Chrome with flags** for a completely clean security profile

For anything beyond quick testing, the right solution is to **fix the server configuration** — install a valid TLS certificate and properly configure HTTPS. Tools like [Let's Encrypt](https://letsencrypt.org/) make this free and straightforward.
