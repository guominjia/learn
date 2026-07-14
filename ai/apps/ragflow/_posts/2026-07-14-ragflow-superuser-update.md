---
title: How a RAGFlow Startup-Order Change Can Create Two Superusers
categories: [ai, rag, ragflow]
tags: [ragflow, superuser, admin, docker, initialization]
---

RAGFlow commit [`7d64a78`](https://github.com/infiniflow/ragflow/commit/7d64a78f830f9a0eb0dfc0397dfc7dadff7aa0ae) unified the Go services into one binary and reorganized the Docker entrypoint. One small but important side effect is that the Admin service is now launched before the Web service.

For most deployments this looks like a harmless ordering change. However, the two Python services use different rules when initializing a superuser. On a fresh database, enabling the Admin service and requesting Web superuser initialization with a custom email can therefore create two superusers.

## The Two Initialization Policies

RAGFlow has two independent superuser initialization paths.

### Admin Service: Ensure Any Superuser Exists

The Python Admin service calls `init_default_admin()` during startup. Its existence check is global:

```python
users = UserService.query(is_superuser=True)
if not users:
		# Create admin@ragflow.io
```

If the database does not contain any superuser, it creates a hardcoded account:

| Field | Value |
|---|---|
| Email | `admin@ragflow.io` |
| Password | `admin` |
| Nickname | `admin` |
| Superuser | `True` |

The important point is that the Admin service asks:

> Does the system already have a superuser?

It does not require that the existing superuser use `admin@ragflow.io`.

### Web Service: Ensure a Specific Email Exists

When the Web service receives `--init-superuser`, it calls `init_superuser()`. Its defaults can be overridden with environment variables such as `DEFAULT_SUPERUSER_EMAIL`.

Its check is email-specific:

```python
if UserService.query(email=email):
		logging.info(
				"User with email %s already exists, skipping initialization.",
				email,
		)
		return
```

The Web service therefore asks a different question:

> Does the requested email already exist?

It does not stop merely because another superuser exists.

## Behavior Before the Commit

Before commit `7d64a78`, the relevant part of `docker/entrypoint.sh` launched the Web service before the Admin service:

```text
Web service
	-> init_superuser(custom@example.com)
Admin service
	-> init_default_admin()
```

On a fresh database, the sequence was:

1. The Web service checked for `custom@example.com`.
2. The email did not exist, so the Web service created it as a superuser.
3. The Admin service started afterward.
4. `init_default_admin()` found that the system already had a superuser.
5. The Admin service did not create `admin@ragflow.io`.

The final state contained one superuser:

| Email | Created by |
|---|---|
| `custom@example.com` | Web service |

The Admin service's global `is_superuser=True` check acted as a fallback rather than a second account-provisioning mechanism.

## Behavior After the Commit

The commit moved the Admin startup block before the Web startup block:

```text
Admin service
	-> init_default_admin()
Web service
	-> init_superuser(custom@example.com)
```

On a fresh database, the new sequence is:

1. The Admin service starts and queries for any superuser.
2. None exists, so it creates the hardcoded `admin@ragflow.io` account.
3. The Web service starts with `--init-superuser`.
4. The Web service checks whether `custom@example.com` exists.
5. It does not find that email, so it creates a second superuser.

The final state now contains two superusers:

| Email | Created by |
|---|---|
| `admin@ragflow.io` | Admin service |
| `custom@example.com` | Web service |

This is not a duplicate-email insertion. It is a duplicate initialization outcome: two different accounts receive superuser privileges because the two services enforce different invariants.

## When the Problem Appears

The behavior requires the relevant initialization paths to run. It is most visible when all of the following are true:

- The database does not yet contain a superuser.
- The Python Admin service is enabled with `--enable-adminserver`.
- The Web service is started with `--init-superuser`.
- `DEFAULT_SUPERUSER_EMAIL` is set to an address other than `admin@ragflow.io`.

If the Web service uses the default email, it finds the account created by the Admin service and skips initialization. In that case, only one superuser remains.

The entrypoint launches service loops in the background and does not wait for Admin initialization to complete before launching the Web service. The actual database operations can therefore race. Nevertheless, the script now gives the Admin service the first opportunity to create the hardcoded account, making the two-superuser outcome possible and likely during a fresh startup.

## Why the Old Order Hid the Design Mismatch

The underlying issue is not only process order. The deeper problem is that the two initializers define success differently:

| Initializer | Invariant |
|---|---|
| `init_default_admin()` | At least one superuser exists |
| `init_superuser()` | A user with the requested email exists |

The old order happened to compose safely:

```text
Create the requested superuser
				->
Observe that some superuser exists
```

Reversing the order does not:

```text
Create any superuser
				->
Observe that the requested email is still missing
				->
Create another superuser
```

This is a common distributed-bootstrap problem. Two individually idempotent functions are not necessarily idempotent when they use different identity keys or different definitions of the desired state.

## Operational Workarounds

Until both initialization paths use a consistent policy, deployments can avoid the extra account in several ways.

### Use the Same Email

Keep `DEFAULT_SUPERUSER_EMAIL` equal to `admin@ragflow.io`. The Web initializer will see the account created by the Admin service and return without creating another user.

### Bootstrap Before Enabling the Admin Service

Start the Web service once with `--init-superuser` while the Admin service is disabled. After the custom superuser exists, enable the Admin service. Its global superuser check will then prevent creation of the hardcoded account.

### Pre-Provision the Custom Superuser

Create the intended superuser before normal service startup. Because the Admin initializer checks for any superuser, it will not create `admin@ragflow.io` when the custom account is already present.

In all cases, verify the `user` table after initialization and remove unintended accounts only after confirming tenant relationships and access requirements.

## A Better Fix

The robust solution is to centralize superuser bootstrap and define one explicit invariant. RAGFlow should choose one of these policies:

1. **Ensure at least one superuser exists.** Both services should stop when any superuser is present.
2. **Ensure one configured bootstrap identity exists.** Both services should read the same email and credentials and check that identity.

The second policy provides predictable configuration, but either is safer than mixing both approaches. Initialization should also be protected against concurrent startup with a database uniqueness constraint and an atomic insert-or-ignore operation.

Most importantly, only one component should own bootstrap. Other services should validate the result rather than independently creating privileged accounts.

## Summary

Commit `7d64a78` changed RAGFlow's Docker startup order from Web-before-Admin to Admin-before-Web. Before the change, a custom Web-created superuser caused the Admin initializer to stop because it checked whether any superuser existed.

After the change, the Admin service can first create the hardcoded `admin@ragflow.io` account. The Web service then checks only for its configured email and creates that account as another superuser when the email differs.

The startup-order change exposed an existing mismatch between two initialization policies. The long-term fix is not merely to restore the old order, but to make superuser bootstrap single-owner, atomic, and based on one consistent definition of identity.

## References

- [RAGFlow commit `7d64a78`: Go: unify three services into one binary](https://github.com/infiniflow/ragflow/commit/7d64a78f830f9a0eb0dfc0397dfc7dadff7aa0ae)
- [RAGFlow repository](https://github.com/infiniflow/ragflow)
