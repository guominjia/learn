---
layout: post
title: "Python HTTP Clients and TLS Certificate Verification"
date: 2026-07-16
categories: [python, network]
tags: [python, http, https, tls, ssl, certificate, certifi, requests, urllib, httpx, aiohttp]
---

Python has several commonly used HTTP libraries, including `urllib`, `requests`, `httpx`, and `aiohttp`. Their APIs and concurrency models differ, but HTTPS requests in all of them eventually need to answer the same question:

> Which certificate authorities (CAs) should be trusted when verifying the server certificate?

Understanding where that CA information comes from is useful when an HTTPS request fails with an error such as `CERTIFICATE_VERIFY_FAILED`, especially in corporate networks, containers, or minimal Linux systems.

## Common Python HTTP Libraries

The following libraries overlap in purpose but have different design goals:

| Library | Typical style | Notes |
|---|---|---|
| `urllib` | Synchronous | Part of the Python standard library. |
| `requests` | Synchronous | Popular high-level HTTP API built on `urllib3`. |
| `httpx` | Synchronous and asynchronous | A modern API that supports both execution models. |
| `aiohttp` | Asynchronous | Designed around `asyncio`, including HTTP client and server features. |

Choosing synchronous or asynchronous code is separate from certificate verification. A synchronous client can verify certificates correctly, and an asynchronous client can fail verification if it does not have access to the appropriate CA certificates.

## Why HTTPS Needs CA Certificates

When a client connects to `https://example.com`, the server presents a certificate chain. The client verifies that chain against a set of trusted root CA certificates.

Conceptually, the verification path is:

```text
server certificate -> intermediate CA certificate -> trusted root CA certificate
```

If the client cannot find a trusted root, cannot validate the hostname, or sees an expired certificate, it rejects the connection by default. Disabling verification may make a request appear to work, but it removes an important protection against server impersonation and should not be the normal fix.

## `requests` and `certifi`

`requests` depends on `certifi` in its normal installation. `certifi` packages a curated CA bundle derived from Mozilla's trusted root program. This gives `requests` a predictable CA bundle that is not tied directly to the operating system's CA-file location.

Use `certifi.where()` to find the bundle path used by the package:

```python
import certifi

print(certifi.where())
```

The result is usually a path to a PEM file inside the active Python environment, for example inside `site-packages`. The bundle is maintained by the `certifi` project and updated by upgrading the package:

```powershell
python -m pip install --upgrade certifi
```

For a normal `requests` call, certificate verification is enabled by default:

```python
import requests

response = requests.get("https://example.com", timeout=10)
print(response.status_code)
```

An application can explicitly provide a CA bundle with the `verify` argument, but that should normally be reserved for a deliberately managed internal CA bundle:

```python
response = requests.get(
	"https://internal.example",
	verify=r"C:\path\to\company-ca-bundle.pem",
	timeout=10,
)
```

## `urllib` and Python's Default SSL Context

`urllib.request.urlopen()` is part of the standard library. For HTTPS, it uses Python's `ssl` module, which is usually backed by OpenSSL.

`ssl.get_default_verify_paths()` reports the default CA-file and CA-directory settings configured for the OpenSSL build used by Python:

```python
import ssl

paths = ssl.get_default_verify_paths()
print(paths)
```

The returned object includes values such as `cafile` and `capath`, along with environment variable names that can override them. These values are lookup locations, not a guarantee that a file or directory exists on the current machine. For example, a Python distribution may report `/etc/ssl/cert.pem`, while the operating system stores its bundle at a different location.

Create an explicit default context when using `urlopen`:

```python
import ssl
from urllib.request import urlopen

context = ssl.create_default_context()

with urlopen("https://example.com", context=context) as response:
	print(response.status)
```

`ssl.create_default_context()` creates a client context that verifies certificates and checks hostnames. It is the appropriate starting point for HTTPS client code that needs an explicit context.

## Operating System Trust Stores

The source of trust differs by platform and Python build:

- On Windows, Python can load certificates from the Windows `CA` and `ROOT` certificate stores when loading default certificates.
- On Linux and other Unix-like systems, Python asks OpenSSL to use its default CA-file and CA-directory locations. Those locations are commonly connected to the operating system's CA bundle, but their exact paths vary by distribution and Python packaging.
- On macOS, the behavior can also depend on how Python was installed and which SSL backend or certificate bundle that distribution provides.

Therefore, `ssl.get_default_verify_paths()` is a useful diagnostic tool, but it is not a complete inventory of every certificate source available to Python. In particular, a missing `cafile` path does not by itself prove that verification cannot use an operating-system certificate store or another configured CA directory.

## `httpx` and `aiohttp`

`httpx` supports both synchronous and asynchronous calls. Its TLS verification is enabled by default, and it can be configured with a custom SSL context when an application needs a private CA:

```python
import httpx
import ssl

context = ssl.create_default_context(cafile="company-ca-bundle.pem")

with httpx.Client(verify=context) as client:
	response = client.get("https://internal.example")
	print(response.status_code)
```

`aiohttp` is asynchronous and accepts an SSL context through its connector:

```python
import aiohttp
import asyncio
import ssl


async def fetch() -> None:
	context = ssl.create_default_context()
	connector = aiohttp.TCPConnector(ssl=context)

	async with aiohttp.ClientSession(connector=connector) as session:
		async with session.get("https://example.com") as response:
			print(response.status)


asyncio.run(fetch())
```

Using an explicit context is particularly helpful when the same trust configuration must be shared across `urllib`, `httpx`, and `aiohttp`.

## Diagnosing Verification Failures

When a request fails certificate verification, check the following before turning verification off:

1. Verify the system clock. An incorrect clock can make valid certificates appear expired or not yet valid.
2. Inspect the default OpenSSL locations with `ssl.get_default_verify_paths()`.
3. For `requests`, print `certifi.where()` and confirm that the package is current.
4. Determine whether a proxy or internal service uses a private CA. Add that CA to a managed bundle or SSL context instead of disabling verification.
5. Check the Python distribution and runtime environment. Containers, virtual environments, and corporate Python distributions may use different trust sources.

For a quick view of the certificate chain presented by a server, OpenSSL is also useful:

```powershell
openssl s_client -connect example.com:443 -servername example.com -showcerts
```

## Takeaway

`requests` commonly verifies HTTPS certificates with the CA bundle supplied by `certifi`, and `certifi.where()` reveals its path. Standard-library `urllib` relies on Python's SSL defaults; `ssl.get_default_verify_paths()` reports OpenSSL's configured CA lookup locations, which may not all exist as ordinary files and may be supplemented by platform certificate stores.

Keep TLS verification enabled. When an internal CA is required, make the trust decision explicit by providing a carefully maintained CA bundle or SSL context rather than using `verify=False` or an unverified context.

## References

- [Python `ssl` module documentation](https://docs.python.org/3/library/ssl.html)
- [Python `urllib.request` documentation](https://docs.python.org/3/library/urllib.request.html)
- [certifi documentation](https://certifiio.readthedocs.io/)
- [Requests SSL certificate verification](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification)
- [HTTPX SSL documentation](https://www.python-httpx.org/advanced/ssl/)
- [aiohttp client reference](https://docs.aiohttp.org/en/stable/client_reference.html)
