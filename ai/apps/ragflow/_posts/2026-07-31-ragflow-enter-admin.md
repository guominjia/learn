---
title: How to Enter the RAGFlow Admin Page
categories: [ai, rag, ragflow]
tags: [ragflow, admin, routes, administration]
---

RAGFlow has a separate administrator backend. It is not listed in the normal workspace navigation or the user-settings sidebar, so opening the main application and looking for an **Admin** menu will not lead to it.

## Open the Admin URL Directly

Append `/admin` to the address of the running RAGFlow instance:

```text
http://<ragflow-host>/admin
```

For a local default installation, this is commonly:

```text
http://localhost/admin
```

If the deployment uses a non-default port, include it in the URL:

```text
http://localhost:9380/admin
```

Replace `localhost` with the server hostname when RAGFlow is deployed remotely.

## Sign In as an Administrator

The `/admin` route opens the administrator login area. It has a separate authorization layout from the standard RAGFlow workspace, so signing in to the regular application does not by itself guarantee access to the administrator backend.

Use an account with the administrator credentials configured for that deployment. After successful authentication, the available administrative sections can include service status, user management, and sandbox settings. Enterprise builds can expose additional administration features.

## When the Page Does Not Open

Confirm these points before changing frontend code:

1. The browser is visiting the exact `/admin` URL on the running RAGFlow host.
2. The deployed frontend build includes the administrator route.
3. The account has the required administrator credentials or permissions.

If the page is missing in a remote deployment but present in a local source checkout, compare the running container or frontend image version with the source revision. The browser may be serving an older or different build.

## Why There Is No Regular Menu Entry

The frontend registers `/admin` as an independent route tree. Its root renders the admin login page, and protected child routes render the authorized administration interface. This structure deliberately keeps administrator access separate from normal user settings.

## Reference

- [RAGFlow frontend route configuration](https://github.com/infiniflow/ragflow/blob/main/web/src/routes.tsx)
