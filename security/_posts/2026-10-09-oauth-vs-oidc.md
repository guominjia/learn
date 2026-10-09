---
layout: post
title: "Sign in with Google, Microsoft, or GitHub: OAuth 2.0 or OpenID Connect?"
date: 2026-10-09
categories: [security, authentication]
tags: [oauth, oidc, sso, entra-id, azure, google, github, saml]
---

Every "Sign in with ..." button starts with a redirect to the provider, so it is tempting to call all of them "OAuth". Not quite. Google and Microsoft sign-in use **OpenID Connect (OIDC)**, an identity layer on top of OAuth 2.0. GitHub's user sign-in for OAuth apps is plain **OAuth 2.0**.

## OAuth 2.0 vs. OpenID Connect

| | OAuth 2.0 | OpenID Connect |
|---|---|---|
| Problem it solves | Authorization: let an app call an API on the user's behalf | Authentication: tell the app who the user is |
| What the client gets | `access_token` for an API | Also an `id_token`, a signed JWT with `iss`, `sub`, `aud`, `exp`, `iat` |
| How a request opts in | — | `openid` in `scope` |
| Provider metadata | — | `/.well-known/openid-configuration` discovery document |

RFC 6749 states that authenticating resource owners to clients is out of scope for OAuth 2.0. OIDC Core says the same thing from the other side: OAuth 2.0 alone does not define standard methods to provide identity information.

So an app that signs users in with plain OAuth 2.0 must take a detour: get an access token, then call a provider-specific user API. For GitHub, that API is `GET https://api.github.com/user`.

## Provider by provider

| Provider | User sign-in protocol | Evidence |
|---|---|---|
| Google | OIDC (OpenID Certified) | Discovery at `https://accounts.google.com/.well-known/openid-configuration`; the code exchange returns an `id_token` |
| Microsoft Entra ID (`login.microsoftonline.com`) | OAuth 2.0 + OIDC; SAML 2.0 for SAML apps | Microsoft documents "standards-compliant implementations of OAuth 2.0 and OpenID Connect (OIDC) 1.0", plus a separate SAML 2.0 SSO protocol |
| GitHub (OAuth apps) | OAuth 2.0 | The token response contains `access_token`, `scope`, `token_type` (and `refresh_token` for expiring tokens), but no `id_token`. Docs say to call `/user` to identify the user |

GitHub does run an OIDC provider, but for **GitHub Actions**, not for user sign-in. Its issuer is `https://token.actions.githubusercontent.com`. Each workflow job gets a JWT that a cloud provider such as Azure or AWS can exchange for a short-lived credential, so no long-lived cloud secret has to be stored in GitHub.

## Is Azure's authentication OAuth, or just "compatible"?

It is standard OAuth 2.0 and OIDC 1.0, not a proprietary look-alike. Everyday Azure sign-in scenarios map onto standard grants:

| Scenario | Flow |
|---|---|
| Browser sign-in to a web app or SPA | Authorization code (+ PKCE), with OIDC for the `id_token` |
| `az login` on Linux/macOS, or Windows with the broker disabled | Authorization code in a browser; device code if no browser can be opened |
| `az login --use-device-code` | Device code |
| Daemon or service principal calling Graph or ARM | Client credentials (secret or certificate) |
| Web API calling a downstream API as the user | On-behalf-of |
| Workload on an Azure resource | Managed identity: the platform obtains Entra tokens, and the code never handles a credential |

On Windows, `az login` defaults to the Web Account Manager (WAM) broker since Azure CLI 2.61.0.

Each access token's `aud` claim names the resource it is for, for example Microsoft Graph. Clients should still treat access tokens as opaque. Microsoft notes that tokens for its own APIs, such as Graph, use a proprietary format and may not be decodable JWTs.

## Check it yourself

```powershell
Invoke-RestMethod https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration | Select-Object issuer, authorization_endpoint, token_endpoint
Invoke-RestMethod https://accounts.google.com/.well-known/openid-configuration | Select-Object issuer, scopes_supported
Invoke-RestMethod https://token.actions.githubusercontent.com/.well-known/openid-configuration | Select-Object issuer, jwks_uri
```

A provider that supports OIDC Discovery serves a JSON document at its issuer URL plus `/.well-known/openid-configuration`. Its `issuer` value must match the `iss` claim in the ID tokens it issues. Microsoft's multi-tenant `common` endpoint returns the template `https://login.microsoftonline.com/{tenantid}/v2.0`, which is filled in with each token's tenant.

## References

- [RFC 6749: The OAuth 2.0 Authorization Framework](https://www.rfc-editor.org/rfc/rfc6749.html): OAuth 2.0 as an authorization framework; §10.16 puts authenticating resource owners to clients out of scope.
- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0.html): OIDC as an identity layer on OAuth 2.0, the `openid` scope, and the ID Token as a signed JWT with `iss`, `sub`, `aud`, `exp`, `iat`.
- [OpenID Connect Discovery 1.0](https://openid.net/specs/openid-connect-discovery-1_0.html): the `/.well-known/openid-configuration` document and the requirement that `issuer` match the ID Token's `iss`.
- [Google: OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect): Google's OIDC implementation is OpenID Certified; its discovery document URL; the token response includes `id_token`.
- [Microsoft: OAuth 2.0 and OpenID Connect protocols](https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols): Microsoft identity platform implements OAuth 2.0 and OIDC 1.0; `login.microsoftonline.com` endpoints.
- [Microsoft: Single sign-on SAML protocol](https://learn.microsoft.com/en-us/entra/identity-platform/single-sign-on-saml-protocol): Entra ID's SAML 2.0 SSO support.
- [Microsoft: Authentication flow support in MSAL](https://learn.microsoft.com/en-us/entra/identity-platform/msal-authentication-flows): authorization code, client credentials, device code, and on-behalf-of flows.
- [Microsoft: Sign in with Azure CLI](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli-interactively): WAM default on Windows, browser authorization code, device code fallback, `--use-device-code`.
- [Microsoft: Managed identities for Azure resources](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview): applications obtain Entra tokens without managing credentials.
- [Microsoft: Access tokens](https://learn.microsoft.com/en-us/entra/identity-platform/access-tokens): `aud` claim, treating access tokens as opaque, Graph tokens' proprietary format, and the `{tenantid}` issuer template.
- [GitHub: Authorizing OAuth apps](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps): authorization code and device grants, token response fields, and calling `/user` to identify the user.
- [GitHub: OpenID Connect (Actions)](https://docs.github.com/en/actions/concepts/security/openid-connect): Actions OIDC tokens with issuer `https://token.actions.githubusercontent.com`, exchanged for short-lived cloud credentials.