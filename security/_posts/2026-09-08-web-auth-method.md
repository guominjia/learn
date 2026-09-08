---
title: "What the HTTP Authorization Header Actually Standardizes"
date: 2026-09-08
categories: [security, authentication, web]
tags: [http, authorization, authentication, oauth, api]
---

The `Authorization` header is often described as if it had one fixed format. It does not. HTTP standardizes the header field and the authentication framework, while individual authentication schemes define how credentials are structured and verified.

That distinction explains why these headers all have the same shape but different meanings:

```http
Authorization: Basic <credentials>
Authorization: Bearer <access-token>
Authorization: Digest <parameters>
Authorization: AWS4-HMAC-SHA256 <signature-data>
Authorization: token <github-token>
```

The word immediately after `Authorization:` is the **authentication scheme**. Everything after the required space is interpreted according to that scheme.

## The Generic HTTP Format

RFC 9110 defines the generic credential syntax as:

```text
credentials = auth-scheme [ 1*SP ( token68 / #auth-param ) ]
auth-scheme = token
```

In practical terms, an `Authorization` value contains a scheme name followed by either a token-like value or a list of named parameters. HTTP does not decide what those credentials mean.

The normal challenge-response flow looks like this:

```text
Client -> GET /private-resource
Server -> 401 Unauthorized
					WWW-Authenticate: Basic realm="example"
Client -> GET /private-resource
					Authorization: Basic <credentials>
```

For a proxy, the corresponding fields are `Proxy-Authenticate` and `Proxy-Authorization`, and the challenge status is `407 Proxy Authentication Required`. A `401` response means that valid authentication credentials are missing or were rejected. A `403 Forbidden` response usually means that the credentials were accepted but are not sufficient for the requested resource.

## Registered HTTP Authentication Schemes

The IANA HTTP Authentication Scheme Registry is the authoritative list of registered scheme names. Registration documents define the scheme-specific syntax and behavior; the registry itself does not make every scheme equally common or equally suitable for a new application.

| Scheme | Specification | Credential model | Typical note |
|---|---|---|---|
| `Basic` | RFC 7617 | Base64 of `user-id:password` | Requires HTTPS or another secure transport. Base64 is encoding, not encryption. |
| `Bearer` | RFC 6750 | A bearer access token | Possession of the token is normally enough to use it, so protect it in storage and transport. |
| `Digest` | RFC 7616 | A challenge-response digest | The password is not sent directly, but the scheme still has important limitations and should use a secure channel. |
| `DPoP` | RFC 9449 | A proof bound to a key and an access token | Designed to provide proof of possession for OAuth-related requests. |
| `HOBA` | RFC 7486 | A cryptographic signature | A registered signature-based scheme. |
| `Mutual` | RFC 8120 | Mutual challenge-response authentication | A registered mutual-authentication scheme. |
| `Negotiate` | RFC 4559 | SPNEGO, commonly Kerberos or NTLM in Windows environments | Used for integrated enterprise authentication. |
| `OAuth` | RFC 5849 | OAuth 1.0 request parameters and signatures | This is distinct from OAuth 2.0 bearer-token usage. |

The registry also contains newer and specialized schemes. The important point is that a scheme is not standardized merely because a service happens to use a familiar word such as `Token` or `ApiKey`.

## Basic Authentication

The `Basic` scheme concatenates a user ID, a colon, and a password, then encodes the resulting bytes with Base64:

```text
user:password -> Base64(user:password)
```

For example:

```http
Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
```

Anyone who can decode that value can recover the original credentials. Basic authentication is therefore not safe on plain HTTP. Use HTTPS, and avoid reusing the password outside the protected realm.

## Bearer Tokens and OAuth 2.0

The common OAuth 2.0 form is:

```http
Authorization: Bearer eyJhbGciOi...
```

`Bearer` is the HTTP authentication scheme. OAuth 2.0 defines how an authorization server can issue access tokens; it does not require the access token to be a JWT. RFC 6750 deliberately does not specify the token's internal encoding or contents.

The token's format and the header scheme are separate decisions:

| Concept | Example | What it describes |
|---|---|---|
| HTTP scheme | `Bearer` | How the resource server expects the credential to be presented |
| Token format | JWT, opaque random string | How the token itself is represented |
| Authorization framework | OAuth 2.0 | How a client obtains and uses an access token |

A JWT can therefore be carried with `Bearer`, but JWT is not itself an HTTP `Authorization` scheme. Similarly, an OAuth 2.0 access token can be opaque rather than a JWT.

## Vendor and Application-Specific Conventions

Services can define their own scheme names and credential syntax. GitHub's REST API is a useful example. Its current documentation accepts both of these forms for many token-based requests:

```http
Authorization: Bearer YOUR-TOKEN
Authorization: token YOUR-TOKEN
```

GitHub specifically requires `Bearer` when the value is a JSON Web Token. The service documentation, not the word `token` by itself, defines the compatibility behavior.

AWS Signature Version 4 is another example. It signs selected request components and places the resulting authentication information in the `Authorization` header. It is a vendor-defined signing protocol, not a general-purpose HTTP scheme that every server understands:

```http
Authorization: AWS4-HMAC-SHA256 Credential=..., SignedHeaders=..., Signature=...
```

The same caution applies to `ApiKey`, `Token`, and similar values. They can be valid application conventions, but they are not interchangeable with `Bearer`, and a generic HTTP client cannot infer their semantics.

SAML assertions, JWTs, API keys, and access tokens are credential or token formats. They become part of an `Authorization` header only when a service defines how they are carried, often using `Bearer` or a service-specific scheme. A cookie-based session is another authentication mechanism, but it uses the `Cookie` header rather than `Authorization`.

## Security Rules

The header name does not make a credential secure. Apply the rules of the selected scheme and the transport:

1. Use HTTPS before sending passwords, bearer tokens, API keys, or signatures.
2. Do not put bearer tokens in URLs. URLs are commonly copied to browser history, logs, monitoring systems, and referrer data.
3. Treat bearer tokens and API keys like passwords. Do not commit them to source control or print them in request logs.
4. Do not assume that a signed request protects every part of a message. Read the signing scheme's coverage rules.
5. Remove or re-evaluate `Authorization` when following a redirect to a different origin. Credentials intended for one origin should not be sent blindly to another.
6. Prefer the authentication method documented by the service. A syntactically valid header can still be rejected, or worse, be interpreted differently by a gateway and an origin server.

## Choosing a Format

For a new API, use the scheme required by the identity system and resource server. OAuth 2.0 deployments commonly present access tokens with `Bearer`; a service-to-service integration may instead require a request-signing scheme, mutual TLS, or an enterprise `Negotiate` flow. Basic authentication is simple and interoperable, but it is mainly appropriate for controlled cases protected by HTTPS.

The useful mental model is:

```text
Authorization header
	-> authentication scheme
			-> scheme-specific credential syntax
					-> service-specific validation and permissions
```

HTTP provides the extension point. The scheme specification and the service's documentation determine the actual authentication behavior.

## References

- [RFC 9110, HTTP Authentication](https://www.rfc-editor.org/rfc/rfc9110.html#name-http-authentication) — defines the generic authentication framework, `Authorization` syntax, challenge flow, `401`/`403` behavior, and scheme registration model.
- [IANA HTTP Authentication Scheme Registry](https://www.iana.org/assignments/http-authschemes/http-authschemes.xhtml) — lists registered HTTP authentication scheme names and their specifications.
- [RFC 7617, The Basic HTTP Authentication Scheme](https://www.rfc-editor.org/rfc/rfc7617.html) — defines the `user-id:password` and Base64 representation and its transport-security requirements.
- [RFC 6750, OAuth 2.0 Bearer Token Usage](https://www.rfc-editor.org/rfc/rfc6750.html) — defines the `Bearer` scheme, explains bearer-token possession, and recommends protecting tokens with TLS and avoiding URLs.
- [RFC 7616, HTTP Digest Access Authentication](https://www.rfc-editor.org/rfc/rfc7616.html) — defines Digest challenge-response authentication and documents its security limitations.
- [GitHub REST API authentication](https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api) — documents GitHub's supported `Bearer` and `token` forms and its JWT-specific `Bearer` requirement.
- [AWS Signature Version 4 for API requests](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv.html) — documents AWS request signing and placing SigV4 authentication information in the `Authorization` header.
