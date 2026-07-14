---
title: Why RAGFlow Adds the INVALID_ Prefix to Access Tokens
categories: [ai, rag, ragflow]
tags: [ragflow, authentication, access-token, logout, security]
---

An `access_token` beginning with `INVALID_` can look like a malformed token, a database default, or an unexpected authentication bug. In RAGFlow, however, the prefix has a specific meaning: the user has logged out and the previously valid access token has been revoked.

The prefix is not added to every access token. It appears only after logout, when RAGFlow replaces the user's current token with a deliberately invalid sentinel value.

## The Normal State: A Valid Token

After a successful login, RAGFlow stores a normal generated token in the user's `access_token` field. The token is a UUID-style value represented by at least 32 hexadecimal characters.

Conceptually, the authenticated state looks like this:

```text
Client credential -> decoded access token -> user.access_token in database
```

For a request to pass token authentication, the token recovered from the client's authorization value must match the token currently stored for that user. This database lookup makes the token stateful: possession of an old token is not sufficient if the database value has since changed.

## Logout Replaces the Token

The logout endpoint in `api/apps/restful_apis/user_api.py` does not set `access_token` to `NULL` or an empty string. Instead, it replaces the current value with a new random token carrying the `INVALID_` prefix:

```python
user.access_token = f"INVALID_{secrets.token_hex(16)}"
```

The resulting value resembles:

```text
INVALID_4ad8e9676fb74d78cb1db8b79e9f253a
```

The important operation is replacement. Suppose the stored token before logout is:

```text
8f75f1e13e514ec08a8fc0f1bc8f11cb
```

After logout, the database contains a different value:

```text
INVALID_4ad8e9676fb74d78cb1db8b79e9f253a
```

Any request that still presents `8f75f1e13e514ec08a8fc0f1bc8f11cb` can no longer find a matching user record. The old credential therefore stops working immediately, even if the client still has a serialized authorization token or an unexpired session cookie.

RAGFlow's Admin service follows the same pattern in `admin/server/routes.py`. Its `logout()` route also rewrites the current user's token with a random `INVALID_` value.

## Why Use a Prefix Instead of an Empty Value?

Clearing the field would also break the old token, but a prefixed sentinel has useful properties.

### Immediate Revocation

The old token no longer matches the database record. No separate token blacklist or expiration wait is required.

### Explicit State

An empty field can have several possible meanings: the user has never logged in, token creation failed, data was migrated incorrectly, or the token was intentionally removed. `INVALID_` records an explicit state transition: the stored login token was deliberately invalidated.

This makes database inspection, logging, and incident diagnosis easier. It is immediately apparent that the value is not intended to authenticate a request.

### A Non-Reusable Replacement

The random suffix avoids assigning the same invalid value to every logged-out user. It also ensures that a previous `INVALID_` value is replaced by another unique sentinel if logout processing occurs again.

The prefix does not preserve or encode the old token. The replacement is a fresh random value, so this mechanism does not expose the credential that was revoked.

## Authentication Rejects Invalidated Tokens

RAGFlow enforces the convention in more than one place.

### Token Queries

In `api/db/services/user_service.py`, `UserService.query()` rejects an `access_token` beginning with `INVALID_` before running a normal user lookup:

```python
if str(access_token).startswith("INVALID_"):
	logging.warning("UserService.query: Rejecting invalidated access_token")
	return cls.model.select().where(cls.model.id == "INVALID_LOGOUT_TOKEN")
```

The deliberately impossible user ID makes the query return no authenticated user. It also emits a warning that distinguishes an explicitly invalidated token from a normal lookup miss.

For the token that the client used before logout, rejection usually happens even earlier at the database-match level: the old value no longer equals `user.access_token`. The prefix check is an additional guard for any authentication path that receives the invalidated value itself.

### Session Fallback

Token replacement alone is not enough when authentication can fall back to a server-side session. A browser may retain a session cookie after the database token has been changed.

RAGFlow therefore checks the stored token while loading a user from the session in `api/apps/__init__.py`. If the user's current `access_token` starts with `INVALID_`, session authentication returns no user.

The effective flow is:

```text
Session cookie still exists
	-> load user from session
	-> inspect the user's current access_token
	-> reject when it starts with INVALID_
```

This closes an important gap. Without the check, revoking the access token could invalidate authorization-header authentication while leaving an older browser session usable.

## The Go Service Preserves the Contract

RAGFlow also implements user authentication in Go. The `Logout()` method in `internal/service/user.go` mirrors the Python behavior:

```go
func (s *UserService) Logout(user *entity.User) (common.ErrorCode, error) {
	invalidToken := "INVALID_" + utility.GenerateToken()
	err := s.UpdateUserAccessToken(user, invalidToken)
	if err != nil {
		return common.CodeServerError, err
	}
	return common.CodeSuccess, nil
}
```

This parity matters because the token value is persisted in a shared database. If one implementation cleared the field while another expected an `INVALID_` sentinel, authentication behavior could vary depending on which service processed the request.

The matching Python and Go implementations show that `INVALID_` is a cross-language application contract rather than an accidental string or a Python-only workaround.

## What This Mechanism Is—and Is Not

The `INVALID_` convention is best understood as stateful token revocation by rotation:

1. Login generates and stores a valid token.
2. Requests are accepted only while their token matches the stored value.
3. Logout rotates the stored value to a recognizable invalid sentinel.
4. Token and session authentication paths reject the logged-out state.

It is sometimes useful to call this a soft-revocation marker because the database field remains populated. However, the revocation effect is immediate: the previous credential no longer matches.

The convention does not mean that every token with a long enough format is valid, nor does it mean all users should have an `INVALID_` token. A user normally has such a value only after logout and before a later login generates a new valid token.

It also should not be confused with API keys stored in separate token records. This mechanism specifically describes the user's login `access_token` and the authentication paths that depend on it.

## Debugging an INVALID_ Value

When an `INVALID_` value appears in the user table, the expected interpretation is:

- The user previously had a valid login token.
- A logout path intentionally replaced that token.
- The earlier token should no longer authenticate.
- A later successful login should generate and store a new valid token.

If a user cannot authenticate after logging in again, the presence of `INVALID_` may indicate that token rotation during login was not persisted, logout ran afterward, or different services are observing inconsistent database state. It does not, by itself, indicate that token generation is adding the wrong prefix.

## Summary

RAGFlow uses `INVALID_` as an explicit logout marker. Normal access tokens do not carry this prefix. During logout, the current database token is replaced with a fresh random value such as `INVALID_<random-token>`.

The replacement immediately invalidates the old credential because it no longer matches the database. The recognizable prefix also distinguishes intentional revocation from an empty or missing token. `UserService.query()` rejects prefixed values, session fallback refuses users whose stored token is marked invalid, and the Admin and Go services follow the same convention.

In short, `INVALID_` is not a token format used for active sessions. It is a revocation sentinel that records a deliberate logout while ensuring that the previous access token cannot be reused.

## References

- [Python logout implementation](https://github.com/infiniflow/ragflow/blob/d279aee1ff661bda35d87caf0a6e0ac06810aee6/api/apps/restful_apis/user_api.py#L293)
- [`UserService.query()` invalid-token guard](https://github.com/infiniflow/ragflow/blob/d279aee1ff661bda35d87caf0a6e0ac06810aee6/api/db/services/user_service.py#L59-L61)
- [Session authentication checks](https://github.com/infiniflow/ragflow/blob/d279aee1ff661bda35d87caf0a6e0ac06810aee6/api/apps/__init__.py)
- [Admin logout implementation](https://github.com/infiniflow/ragflow/blob/d279aee1ff661bda35d87caf0a6e0ac06810aee6/admin/server/routes.py)
- [Go `UserService.Logout()` implementation](https://github.com/infiniflow/ragflow/blob/d279aee1ff661bda35d87caf0a6e0ac06810aee6/internal/service/user.go)
