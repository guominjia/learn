---
title: "Serving a Frontend Under an Nginx Subpath"
date: 2026-09-11
tags: [nginx, frontend, reverse-proxy, deployment]
---

# Serving a Frontend Under an Nginx Subpath

Deploying a frontend at the site root is usually straightforward. Deploying the same application at a subpath such as `/app1/` exposes a common boundary problem: the browser resolves frontend URLs, while Nginx routes HTTP requests. A correct reverse-proxy rule cannot repair asset URLs that already point to the wrong place.

## Subdomain or Subpath?

Using a dedicated subdomain is not the only solution, but it is often the least demanding one for an application that expects to run at `/`.

For example, DNS can define an alias:

```text
app1.web.example.com  -> CNAME -> web1.example.com
```

Nginx can then select the application by hostname and proxy the root path unchanged:

```nginx
server {
	listen 443 ssl;
	server_name app1.web.example.com;

	# TLS certificate directives omitted for brevity.

	location / {
		proxy_pass http://127.0.0.1:8080;
		proxy_set_header Host $host;
		proxy_set_header X-Forwarded-Proto $scheme;
	}
}
```

The public URL is now:

```text
https://app1.web.example.com/
```

The application still runs at its expected root path. Root-relative URLs such as `/assets/...`, `/api/...`, and `/favicon.ico` therefore remain correct. A backend redirect such as `Location: /login` also points to the same application host and root path, so there is no `/app1` prefix to add.

## Choose a Certificate That Matches the Hostname

The certificate name must match the hostname in the URL that the client opens. A DNS alias does not make the certificate for `example.com` valid for `app1.example.com`, and the client does not replace the requested hostname with the CNAME target during certificate verification.

Common choices are:

- An individual certificate for each hostname, such as `app1.example.com` and `app2.example.com`.
- One SAN certificate containing several names:

	```text
	example.com
	app1.example.com
	app2.example.com
	```

- A wildcard certificate such as `*.example.com`.

Wildcard matching is label-specific:

| Certificate name | Matches | Does not match |
| --- | --- | --- |
| `*.example.com` | `app1.example.com`, `app2.example.com` | `example.com`, `app1.web.example.com` |
| `*.web.example.com` | `app1.web.example.com` | `web.example.com`, `app1.api.web.example.com` |

Therefore, `*.example.com` does not cover the apex name `example.com`. If the root domain also serves HTTPS, include `example.com` separately, for example as another SAN. It also does not cover two-level names such as `app1.web.example.com`; that hostname needs `*.web.example.com` or an exact SAN entry.

For the example in this article, the Nginx virtual server could use a certificate containing `*.web.example.com`:

```nginx
server {
		listen 443 ssl;
		server_name app1.web.example.com;

		ssl_certificate /etc/letsencrypt/live/web.example.com/fullchain.pem;
		ssl_certificate_key /etc/letsencrypt/live/web.example.com/privkey.pem;

		location / {
				proxy_pass http://127.0.0.1:8080;
				proxy_set_header Host $host;
				proxy_set_header X-Forwarded-Proto $scheme;
		}
}
```

The file paths are examples; the certificate and key must be replaced by the files produced by the certificate-management system in use. Nginx expects the certificate chain and private key through `ssl_certificate` and `ssl_certificate_key`.

When using ACME with Let’s Encrypt, wildcard certificates require the DNS-01 challenge. HTTP-01 validates a token served at `/.well-known/acme-challenge/` and cannot issue wildcard certificates. DNS-01 proves control by placing a TXT record under `_acme-challenge`, so automated issuance normally requires a DNS provider API or another controlled DNS-update process.

The DNS and proxying steps are separate:

- **CNAME only performs DNS aliasing.** It tells DNS that `app1.web.example.com` is an alias for `web1.example.com`; it does not connect an HTTP request to port `8080`.
- **Nginx, an Ingress controller, or a load balancer performs reverse proxying.** It receives the HTTP request, matches its hostname or path, and forwards it to the application.

With a path-based deployment, the public URL remains:

```text
https://web1.example.com/app1/
```

That model is useful when many applications must share one hostname, but the application must support the prefix consistently. Static asset paths, client-side routing, API URLs, WebSocket URLs, cookie `Path` values, and backend redirects may all need `/app1`-aware behavior. For an application that cannot be rebuilt or configured, especially a third-party application, this can be much harder than giving it its own hostname.

Other deployment choices are possible:

- **A separate port**, such as `https://web1.example.com:8080/`, avoids a path prefix but makes the URL less convenient and still requires TLS, firewall, and port-management decisions.
- **Kubernetes Ingress, an API gateway, or a cloud load balancer** can provide the same host- or path-based routing as Nginx. The implementation is managed by a platform, but the boundary is still a reverse proxy.
- **Direct public exposure** removes the shared proxy, but each application then needs its own externally facing TLS, authentication, rate limiting, logging, and security configuration.

For many independent applications, a practical default is one subdomain per application behind one shared Nginx or Ingress entry point. It is not the only architecture, but it keeps applications at their natural root path and usually reduces compatibility work.

## The Browser Resolves the Asset URL First

Suppose the page is opened at:

```text
https://web1/app1/
```

If the generated HTML contains root-relative URLs:

```html
<script src="/resource1.js"></script>
<link href="/resource2.css" rel="stylesheet">
<link rel="icon" href="/favicon.ico">
```

The browser requests:

```text
https://web1/resource1.js
https://web1/resource2.css
https://web1/favicon.ico
```

The leading `/` means the site root. It does not mean “the directory containing the current page,” so the browser does not automatically add `/app1/`.

A URL without that leading slash is resolved relative to the document's base URL. Because the page ends in `/`, these references resolve under the application prefix:

```html
<script src="resource1.js"></script>
<link href="resource2.css" rel="stylesheet">
```

They become:

```text
https://web1/app1/resource1.js
https://web1/app1/resource2.css
```

The trailing slash on the page URL matters. Redirecting `/app1` to `/app1/` avoids a different relative-URL base and gives the application a stable directory-like entry point.

## Configure the Frontend Base Path

The durable fix is to configure the frontend build for the path where it will be served. The exact option depends on the tool. For example, Vite calls it `base`:

```js
// vite.config.js
import { defineConfig } from "vite";

export default defineConfig({
  base: "/app1/",
});
```

The generated HTML should then contain URLs similar to:

```html
<script src="/app1/resource1.js"></script>
<link href="/app1/resource2.css" rel="stylesheet">
<link rel="icon" href="/app1/favicon.ico">
```

Vite's default `base` is `/`, so an unconfigured build generally emits root-relative asset URLs such as `/assets/index-xxxx.js`. That default is appropriate for an application served at the host root, but not automatically for `/app1/`.

The equivalent setting depends on the build tool:

### Vue CLI

Vue CLI calls the setting `publicPath`. Its default is also `/`:

```js
// vue.config.js
module.exports = {
	publicPath: "/app1/",
};
```

### Create React App

Create React App assumes that the application is hosted at the server root unless `homepage` is configured:

```json
{
	"homepage": "/app1"
}
```

For a build-specific value, `PUBLIC_URL` can also be set before running the build:

```powershell
$env:PUBLIC_URL = "/app1"
npm run build
```

Webpack has the corresponding `output.publicPath` setting. Other tools use names such as `assetPrefix` or `basePath`. These settings control generated asset URLs; they do not automatically rewrite every URL written in application code. Check images, fonts, lazy-loaded chunks, API URLs, and the client-side router separately.

## API URLs Are a Separate Decision

The same URL rules apply to requests made by application code. This request is root-relative:

```ts
fetch("/api/users");
```

It always targets:

```text
https://web1/api/users
```

It does not inherit `/app1/` merely because the page was loaded from that subpath.

Removing the leading slash makes the URL document-relative:

```ts
fetch("api/users");
```

The result depends on the current document URL:

| Current document URL | Resolved request URL |
| --- | --- |
| `https://web1/app1/` | `https://web1/app1/api/users` |
| `https://web1/app1/settings` | `https://web1/app1/api/users` |
| `https://web1/app1/settings/` | `https://web1/app1/settings/api/users` |

That last case is usually not what an API client means. A route transition, server redirect, or trailing slash can change the base used by a relative URL, so a relative API root is fragile.

A clearer approach is to define the external prefix once and use it deliberately:

```ts
const basePath = "/app1";

fetch(`${basePath}/api/users`);
```

The API proxy must then agree with that choice. For example, the Nginx rule in this article receives `/app1/api/users` and, because its `proxy_pass` ends in `/`, forwards `/api/users` to the upstream service.

## Configure the Client-Side Router

Asset paths alone are not enough. A browser router also needs to know that `/app1/` is the application's public root.

React Router:

```tsx
<BrowserRouter basename="/app1">
```

Vue Router:

```ts
createRouter({
	history: createWebHistory("/app1/"),
	routes,
});
```

Without a router basename or history base, links and client-side navigation may be generated as if the application were mounted at `/`. A direct request to a client-side route also needs the web server to return the application's `index.html`; otherwise Nginx or the upstream server may look for a file at that route and return 404.

The important invariant is simple:

```text
Application URL:  https://web1/app1/
Asset URL:       https://web1/app1/<asset>
API URL:         https://web1/app1/api/<resource>
Router base:     /app1/
```

## Route the Subpath with Nginx

A typical Nginx reverse-proxy configuration is:

```nginx
location = /app1 {
	return 301 /app1/;
}

location /app1/ {
	proxy_set_header Host $host;
	proxy_set_header X-Forwarded-Proto $scheme;
	proxy_set_header X-Forwarded-Prefix /app1;
	proxy_pass http://127.0.0.1:8080/;
}
```

The final slash in `proxy_pass` is significant. With `location /app1/` and `proxy_pass http://127.0.0.1:8080/`, a request for:

```text
/app1/resource1.js
```

is forwarded upstream as:

```text
/resource1.js
```

Nginx replaces the part of the normalized request URI that matches the location with the URI in `proxy_pass`. If `proxy_pass` has no URI, the original request path is forwarded instead:

```nginx
location /app1/ {
	proxy_pass http://127.0.0.1:8080;
}
```

That second form forwards `/app1/resource1.js` as `/app1/resource1.js`. It is correct only when the upstream application is also configured to serve the `/app1/` prefix.

`X-Forwarded-Prefix` is an optional convention for an upstream framework that knows how to use it. Nginx does not automatically rewrite application-generated HTML, API responses, or router settings because this header is present.

## A Temporary Compatibility Workaround

If the frontend cannot be rebuilt immediately, root-level asset requests can be proxied separately:

```nginx
location ~ ^/(resource1\.js|resource2\.css|favicon\.ico)$ {
	proxy_pass http://127.0.0.1:8080;
}
```

This can get a small application working, but it is a compatibility measure rather than a good deployment boundary. Every root-level asset name is shared with the rest of the host, and adding more exceptions makes the configuration fragile. It can also conflict with another application that legitimately owns a root-level path.

## Debugging Checklist

When a subpath deployment fails, inspect the browser's Network panel before changing Nginx:

1. Confirm whether the failed request starts with `/app1/` or `/`.
2. If it starts with `/`, fix the frontend base or public path.
3. Check API requests independently; `fetch("/api/...")` is still a host-root request.
4. If it starts with `/app1/`, verify whether Nginx should strip the prefix before proxying.
5. Confirm that `/app1` redirects to `/app1/`.
6. Check lazy-loaded chunks, fonts, images, API requests, and client-side navigation, not just the first page load.

The long-term rule is to make the frontend, API client, router, and reverse proxy agree on the same base path. Nginx can route a request, but it cannot make a browser reinterpret a root-relative URL after the browser has already constructed it.

## References

- [MDN: Resolving relative references to a URL](https://developer.mozilla.org/en-US/docs/Web/API/URL_API/Resolving_relative_references) - explains current-directory, parent-directory, and site-root URL resolution.
- [RFC 1034: Aliases and canonical names](https://www.rfc-editor.org/rfc/rfc1034#section-3.6.2) - defines CNAME as a DNS alias from an owner name to a canonical name.
- [RFC 9525: Service Identity in TLS](https://www.rfc-editor.org/rfc/rfc9525) - current specification for certificate hostname matching, multiple DNS names, and one-label wildcard scope.
- [NGINX: Core Module](https://nginx.org/en/docs/http/ngx_http_core_module.html#server_name) - documents `server_name`, virtual server selection, and the request host used by Nginx.
- [NGINX: SSL Module](https://nginx.org/en/docs/http/ngx_http_ssl_module.html#ssl_certificate) - documents the `ssl_certificate` and `ssl_certificate_key` directives.
- [Let's Encrypt: Challenge Types](https://letsencrypt.org/docs/challenge-types/) - documents why HTTP-01 cannot issue wildcard certificates and why DNS-01 is used for them.
- [Vite: Shared Options](https://vite.dev/config/shared-options#base) - documents the `base` option for the public path used in development and production.
- [Vue CLI: Configuration Reference](https://cli.vuejs.org/config/#publicpath) - documents the default `/` value and the `publicPath` setting for subpath deployments.
- [Create React App: Deployment](https://create-react-app.dev/docs/deployment/#building-for-relative-paths) - documents the root-hosting default and the `homepage` setting.
- [Create React App: Advanced Configuration](https://create-react-app.dev/docs/advanced-configuration/) - documents the `PUBLIC_URL` environment variable.
- [React Router: BrowserRouter](https://reactrouter.com/api/declarative-routers/BrowserRouter) - documents the `basename` prop.
- [Vue Router: createWebHistory](https://router.vuejs.org/api/functions/createwebhistory.html) - documents the history `base` parameter.
- [MDN: Using the Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) - documents passing a URL string to `fetch()`; URL resolution follows the browser's base-URL rules.
- [NGINX: Module ngx_http_proxy_module](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) - documents how `proxy_pass` maps a matching location and how behavior differs when its URI is omitted.
- [Kubernetes: Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) - documents host and path routing, TLS termination, load balancing, and the need for an Ingress controller.
