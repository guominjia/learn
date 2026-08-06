---
title: Use the Windows Certificate Store with Python Requests
categories: [security, python]
tags: [requests, tls, ssl, windows, certificates]
---

Python applications running behind a corporate TLS inspection proxy often fail with `CERTIFICATE_VERIFY_FAILED`, even though the same URL opens successfully in a Windows browser. The usual cause is a difference in trust stores: Requests uses the `certifi` CA bundle by default, while the organization root CA is installed in the Windows certificate store.

Do not work around this by passing `verify=False`. That disables certificate validation and hostname checks, which makes the HTTPS connection vulnerable to a man-in-the-middle attack.

For a small application, an adapter can give Requests' HTTPS connection pools a standard-library `SSLContext`. On Windows, `ssl.create_default_context()` loads trusted CA certificates from the `CA` and `ROOT` system stores while retaining certificate validation and hostname verification.

## Adapter

```python
import ssl

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


class WindowsCertificateAdapter(HTTPAdapter):
	def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
		pool_kwargs["ssl_context"] = ssl.create_default_context()
		self.poolmanager = PoolManager(
			num_pools=connections,
			maxsize=maxsize,
			block=block,
			**pool_kwargs,
		)


with requests.Session() as session:
	session.mount("https://", WindowsCertificateAdapter())

	response = session.get(
		"https://example.com",
		timeout=30,
	)
	response.raise_for_status()
```

`Session.mount()` selects an adapter by URL prefix. Mounting at `https://` makes this adapter the transport for direct HTTPS requests issued through that session. The adapter creates a `PoolManager` whose new connection pools receive the supplied `ssl_context`.

## Why It Uses the Windows Store

`ssl.create_default_context()` creates a client TLS context with secure defaults. For server authentication, it requires a valid certificate chain and enables hostname checking. When Python loads its default CA certificates on Windows, it reads the `CA` and `ROOT` system certificate stores.

That makes the code appropriate when the required enterprise root CA has already been deployed and trusted by Windows. It does not import certificates, modify the Windows trust store, or trust an arbitrary certificate presented by a server.

## Scope and Limits

This is a session-local change. Code that calls `requests.get()` directly, or uses a different session, continues to use Requests' normal certificate configuration.

The example configures normal HTTPS connection pools. If the application uses an HTTPS proxy, test that configuration separately: urllib3 has distinct TLS handling for the connection to an HTTPS proxy. Also, this adapter does not configure a client certificate for mutual TLS. Add a client certificate deliberately with Requests' `cert` argument or session setting when the server requires one.

Keep the explicit timeout. Requests does not time out by default, and a single timeout value applies to both connection and read operations. For services where those limits should differ, use a tuple such as `timeout=(5, 30)`.

## When to Use Another Approach

Use this adapter when the application is Windows-only and needs the trust decisions already made by Windows. If the application must be portable, distribute an approved CA bundle and configure `verify` with its path instead. This keeps the trust configuration explicit and reproducible across operating systems.

For a workstation-wide solution, packages that merge the Windows store into certifi exist, but evaluate their maintenance status and deployment implications before making them a dependency. Keeping the behavior in a session adapter is often easier to audit because the scope is visible in the application code.

## References

- [Python `ssl` documentation](https://docs.python.org/3/library/ssl.html#ssl.create_default_context): establishes the secure defaults of `create_default_context()` and that Windows default CA loading reads the `CA` and `ROOT` system stores.
- [Requests advanced usage](https://requests.readthedocs.io/en/latest/user/advanced/#transport-adapters): documents `HTTPAdapter`, session mounting by URL prefix, Requests' default certificate verification, and explicit request timeouts.
- [urllib3 custom SSL contexts](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#custom-ssl-contexts): demonstrates supplying an `ssl_context` to `PoolManager` and distinguishes TLS handling for HTTPS proxies.
