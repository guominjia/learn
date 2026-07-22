---
layout: post
title: "Authenticate Microsoft Graph Requests with Microsoft Entra ID"
date: 2026-07-22
categories: [security, microsoft]
tags: [microsoft, entra-id, azure, microsoft-graph, oauth, msal, sharepoint, onedrive]
---

Microsoft Graph does not grant access merely because an application knows a resource URL. Before an application can call an endpoint such as [List children of a driveItem](https://learn.microsoft.com/en-us/graph/api/driveitem-list-children), it must first obtain an access token from Microsoft Entra ID (formerly Azure Active Directory).

This post explains how App registrations, client credentials, redirect URIs, and Microsoft Graph permissions fit together. It also highlights an important distinction: a permission declared for an application is not always the same as the data that the resulting token can actually read.

## The Authentication Model

Microsoft Authentication Library (MSAL) client applications acquire tokens from Microsoft Entra ID. Microsoft documents the available application types and flows in [MSAL client applications](https://learn.microsoft.com/en-us/entra/identity-platform/msal-client-applications).

At a high level, the request path is:

```text
application -> Microsoft Entra ID -> access token -> Microsoft Graph -> resource
```

The token tells Microsoft Graph which tenant issued it, which application requested it, and which delegated scopes or application roles were granted. Graph evaluates that token before returning OneDrive, SharePoint, user, mail, or other Graph resources.

## Create an App Registration

Sign in to [Azure portal](https://portal.azure.com/), then open:

```text
Microsoft Entra ID -> App registrations -> New registration
```

After the registration is created, record these values from the Overview page:

- **Application (client) ID**: identifies the application.
- **Directory (tenant) ID**: identifies the Entra tenant that owns the application and its resources.

These identifiers are necessary for token acquisition, but they do not themselves authorize access to tenant data. Authorization comes from Microsoft Graph permissions and the consent granted for them.

## Choose an Authentication Flow

Two common token types have very different security models.

| Token type | Typical permissions | Effective read scope |
|---|---|---|
| App-only | `Files.Read.All`, `Sites.Read.All` as **Application** permissions | Can read the corresponding Graph and SharePoint resources exposed across the tenant. It does not depend on any user's existing permissions. |
| Delegated | `Files.Read.All`, `Sites.Read.All` as **Delegated** permissions | Can read only resources that the signed-in user can already access, within the application's granted scopes. |

For a delegated token, the effective access is approximately:

$$
	ext{effective access} = \text{user's existing SharePoint access} \cap \text{application's delegated scopes}
$$

For an app-only token, the effective access is closer to:

$$
	ext{effective access} = \text{application roles approved by an administrator}
$$

There is no current-user permission layer in the app-only case. This is why broad Application permissions such as `Sites.Read.All` and `Files.Read.All` require administrator consent.

## Client Credentials and Redirect URIs

An app-only service usually uses the OAuth 2.0 client credentials flow. Create a secret or certificate under:

```text
App registrations -> <application> -> Certificates & secrets
```

For production services, prefer a certificate or a managed identity where available. A client secret is simpler for development, but it is sensitive material and must be stored outside source control.

Redirect URIs are different. They are required for interactive web, desktop, or single-page application flows, where Entra ID must return the user to the application after sign-in. They are not required for an app-only client-credentials flow because no user is redirected through a browser.

## Configure Microsoft Graph Permissions

In the registration, open:

```text
API permissions -> Add a permission -> Microsoft Graph
```

Then choose either **Delegated permissions** or **Application permissions**. A permission with the same display name means something different depending on which kind is selected.

For an app-only process that reads OneDrive and SharePoint content, a broad configuration is commonly:

```text
Microsoft Graph / Application permissions
	Files.Read.All
	Sites.Read.All
```

After adding Application permissions, a tenant administrator must select **Grant admin consent**. Until consent is granted, the application cannot receive those roles in an app-only token.

`Files.Read.All` is primarily relevant to OneDrive and file resources. `Sites.Read.All` is the central permission for SharePoint site discovery, document-library enumeration, and content access. A connector that works with both OneDrive and SharePoint commonly needs both, but the exact minimum should be verified against its Graph calls.

`Read.All` does not mean access to every system in an organization. It applies only to the Microsoft Graph resource family associated with that permission. Exchange mail, Teams messages, directory objects, and other Graph services each have their own permissions.

## Use `.default` for App-Only Tokens

Client credentials requests use the resource's `/.default` scope:

```python
scopes = ["https://graph.microsoft.com/.default"]
```

`.default` is not itself a permission, and it does not automatically grant all Microsoft Graph access. It tells Entra ID to issue the Application permissions already configured for Microsoft Graph and already approved by an administrator.

The process is:

```text
API permissions -> Application permissions -> Grant admin consent
	-> acquire token with Graph/.default -> token roles claim
```

For example, after `Files.Read.All` and `Sites.Read.All` have been configured and approved, a token payload can contain roles similar to:

```json
{
	"aud": "https://graph.microsoft.com",
	"roles": [
		"Files.Read.All",
		"Sites.Read.All"
	],
	"appid": "<application-client-id>",
	"tid": "<tenant-id>"
}
```

Microsoft Graph uses the `roles` claim for app-only authorization. If the called endpoint requires `Sites.Read.All`, but that role is not present, Graph returns `403 Forbidden`.

By contrast, an interactive delegated flow explicitly requests delegated scopes, for example:

```python
scopes = [
		"https://graph.microsoft.com/Files.Read",
		"User.Read",
]
```

An app-only client cannot add delegated scopes such as `Files.Read.All` at runtime. It has no signed-in user and can only receive pre-approved Application permissions through `/.default`.

## Prefer Site-Scoped Access When Possible

Broad `Sites.Read.All` is often more access than an integration needs. If an application should read only one SharePoint site, grant the narrower Application permission:

```text
Microsoft Graph / Application permission
	Sites.Selected
```

This permission alone is not enough. A SharePoint or Graph administrator must also explicitly grant that application `read` access to the target site. The site-level grant is what limits a stolen token or misconfigured service from enumerating unrelated sites.

For connectors that resolve a site URL and then enumerate the site's drives, `Sites.Selected` can work as long as the target site has received the explicit application grant. A `403 Forbidden` response usually means that either the application role was not consented to or the site-level assignment was never created.

## Troubleshooting Checklist

For an app-only SharePoint connector, verify all of the following:

1. The token was acquired with `https://graph.microsoft.com/.default`.
2. The registration has the required **Application**, not only Delegated, permissions.
3. A tenant administrator granted admin consent.
4. The access token contains the expected roles in its `roles` claim.
5. When using `Sites.Selected`, the target SharePoint site explicitly grants the application the required role.
6. The Graph endpoint matches the resource type covered by the configured permission.

## Takeaway

Microsoft Entra ID establishes the application's identity, and Microsoft Graph permissions establish its authorization. Delegated tokens remain bounded by the signed-in user's existing access; app-only tokens are bounded by the Application permissions that an administrator has approved.

Treat app-only Graph permissions as service-level privileges. Use `Sites.Selected` and explicit site grants when the integration only needs a known SharePoint site, and use broader permissions only when the product requirement truly needs tenant-wide access.

## References

- [MSAL client applications](https://learn.microsoft.com/en-us/entra/identity-platform/msal-client-applications)
- [Microsoft Graph permissions reference](https://learn.microsoft.com/en-us/graph/permissions-reference)
- [List children of a driveItem](https://learn.microsoft.com/en-us/graph/api/driveitem-list-children)
- [Microsoft identity platform and OAuth 2.0 client credentials flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-client-creds-grant-flow)
- [Sites.Selected application permission](https://learn.microsoft.com/en-us/graph/permissions-selected-overview)
