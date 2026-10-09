---
layout: post
title: "What Happens After a Browser Opens a Microsoft Entra ID /authorize URL"
date: 2026-10-09
categories: [security, microsoft]
tags: [microsoft, entra-id, azure, oauth, oidc, pkce, msal, microsoft-graph]
---

A web app that signs users in with Microsoft Entra ID and calls Microsoft Graph usually starts by sending the browser to a URL like this (line breaks added):

```text
https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize
  ?client_id={client-id}
  &response_type=code
  &redirect_uri=https%3A%2F%2Fapp.example.com%2Fauth%2Fcallback
  &scope=User.Read+Mail.Read+Mail.Send+offline_access+openid+profile
  &state={random}
  &code_challenge={BASE64URL(SHA256(code_verifier))}
  &code_challenge_method=S256
  &nonce={random}
  &client_info=1
```

Does Entra ID redirect the browser back to `redirect_uri`? Yes, but only after the user finishes signing in and consenting. The full exchange has three legs.

## 1. The browser calls `/authorize`

What the browser sees depends on its existing Entra session and on whether the user has already consented:

| Situation | What Entra ID returns |
|---|---|
| No usable sign-in session | The sign-in page (account picker if several accounts are signed in) |
| Exactly one signed-in user, all scopes already consented | A redirect straight back, with no UI |
| A requested scope has not been consented yet | The consent prompt, then the redirect |
| `redirect_uri` is not registered on the app | An error page (`AADSTS50011`) and **no** redirect |
| `prompt=none` but interaction is required | An `interaction_required` error |

Refusing to redirect to an unregistered URI is required by RFC 6749 §3.1.2.4: the authorization server must not send the browser to an invalid redirection URI.

## 2. Entra ID redirects back with a code

When `response_type=code` and `response_mode` is omitted, Entra ID defaults to `response_mode=query`, so the result is appended to the redirect URI's query string. RFC 6749 §4.1.2 shows the response as an HTTP redirect:

```text
HTTP/1.1 302 Found
Location: https://app.example.com/auth/callback?code={authorization-code}&state={same-random}
```

| Parameter | Meaning |
|---|---|
| `code` | Authorization code. Entra ID codes typically expire after about 1 minute, and RFC 6749 requires them to be single-use. |
| `state` | Echo of the request value. The app must compare it with the value it stored, to block CSRF (RFC 6749 §10.12). |

On failure, the same redirect URI receives `error` and `error_description` instead, for example `?error=access_denied&error_description=...`.

With `response_mode=form_post`, Entra ID POSTs the code to the redirect URI instead of putting it in the URL. Microsoft recommends this mode, especially for `http://localhost` redirect URIs.

`client_info=1` is not defined by RFC 6749 or OpenID Connect. When a response includes `client_info`, MSAL Python base64url-decodes it into JSON containing `uid` and `utid` and builds the cache key `home_account_id = "{uid}.{utid}"`.

## 3. The app's backend redeems the code

The app's backend now calls the token endpoint directly. The browser is not involved in this leg:

```http
POST /{tenant}/oauth2/v2.0/token HTTP/1.1
Host: login.microsoftonline.com
Content-Type: application/x-www-form-urlencoded

client_id={client-id}
&grant_type=authorization_code
&code={authorization-code}
&redirect_uri=https%3A%2F%2Fapp.example.com%2Fauth%2Fcallback
&code_verifier={original-random-string}
&client_secret={secret}
```

- `redirect_uri` must be the same value that was sent to `/authorize`.
- `code_verifier` is required whenever PKCE was used. Entra ID computes `BASE64URL(SHA256(code_verifier))`, compares it with the earlier `code_challenge`, and returns `invalid_grant` if they differ (RFC 7636 §4.6).
- A confidential web app authenticates with `client_secret`, or with `client_assertion_type` + `client_assertion` when using a certificate. Microsoft recommends certificates. Public clients (SPA, desktop, mobile) must not send a secret.

The token response maps back to the requested scopes:

| Field | Returned when |
|---|---|
| `access_token` | Always. Here it is a Graph token covering `User.Read`, `Mail.Read`, `Mail.Send`. |
| `refresh_token` | Only if `offline_access` was requested |
| `id_token` | Only if `openid` was requested. Its `nonce` claim must equal the request's `nonce`. |

Don't parse the Graph access token in application code. Microsoft warns that tokens for its own APIs may not validate as JWTs and may be encrypted.

## The whole sequence

```text
browser -> login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize   (sign in / consent)
browser <- 302 Location: app.example.com/auth/callback?code=...&state=...
browser -> app.example.com/auth/callback?code=...&state=...
           app backend -> login.microsoftonline.com/{tenant}/oauth2/v2.0/token   (code + code_verifier + client credential)
           app backend <- access_token / refresh_token / id_token
browser <- app page (usually another redirect to drop the code from the URL)
```

## Practical consequences

- **The callback host only needs to be reachable from the user's browser.** The redirect in step C of RFC 6749 §4.1 is performed by the user agent, so an internal host works as long as the browser can reach it.
- **The code leaks into browser history.** RFC 6749 §10.5 notes that codes can leak through history and `Referer` headers. PKCE makes a leaked code useless without `code_verifier`. RFC 6749 §3.1.2.5 also advises keeping third-party scripts off the callback page and redirecting again to strip the code.
- **Redirect URIs must match exactly.** Entra ID requires `https` (except for localhost), treats the path as case-sensitive, and compares the port except for localhost URIs.

## References

- [Microsoft identity platform and OAuth 2.0 authorization code flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-auth-code-flow): default `response_mode`, ~1 minute code lifetime, error redirects, token request parameters, `offline_access`/`openid` gating of `refresh_token`/`id_token`, `nonce`, and the warning against reading Microsoft API tokens.
- [Redirect URI (reply URL) best practices and limitations](https://learn.microsoft.com/en-us/entra/identity-platform/reply-url): `AADSTS50011` on mismatch, the `https` requirement, path case-sensitivity, and localhost port handling.
- [Scopes and permissions in the Microsoft identity platform](https://learn.microsoft.com/en-us/entra/identity-platform/scopes-oidc): meaning of `openid`, `profile`, and `offline_access`, and bare scopes such as `User.Read` defaulting to Microsoft Graph.
- [RFC 6749: The OAuth 2.0 Authorization Framework](https://www.rfc-editor.org/rfc/rfc6749.html): the user-agent redirect and 302 example (§4.1, §4.1.2), single-use codes, no redirect to an invalid URI (§3.1.2.4), callback hygiene (§3.1.2.5), code leakage (§10.5), and `state` for CSRF (§10.12).
- [RFC 7636: Proof Key for Code Exchange](https://www.rfc-editor.org/rfc/rfc7636.html): the `S256` transform and server-side `code_verifier` verification.
- [MSAL Python `token_cache.py`](https://raw.githubusercontent.com/AzureAD/microsoft-authentication-library-for-python/dev/msal/token_cache.py): decoding of `client_info` into `uid`/`utid` and the `home_account_id` format.