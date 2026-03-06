# Inside RAGFlow User Management: A Practical Look at the `user` Table

When you manage RAGFlow in production, understanding how user data is modeled is essential for operations, troubleshooting, and security hardening. In this post, we walk through the `user` table in the `rag_flow` database, common ways to inspect user records, and the key account defaults every administrator should know.

## Why the `user` Table Matters

In RAGFlow, core identity information is stored in the `user` table [db_models.py:690-716](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L690-L716). This is where login identity, account status, and profile-level metadata are tracked.

At a high level:

- `id` is the primary key.
- `email` is the login username and should be unique.
- account flags (for example `is_active` and `is_superuser`) define access state and privilege level.

## Three Ways to Inspect Users

### 1) Query the database directly

```sql
USE rag_flow;

SELECT * FROM user;
DESCRIBE user;
```

This method is best when you need full control, custom filtering, or direct auditing.

### 2) Use the Admin CLI

```bash
ragflow-cli -h 127.0.0.1 -p 9381
admin> list users;
```

CLI is ideal for quick operational checks and lightweight administrative workflows.

### 3) Call the Admin API

Use the `/admin/users` endpoint to fetch user lists programmatically. This is the preferred path for automation scripts and platform integration.

## Related Tables You Should Track

User management in RAGFlow is not isolated to a single table. In most deployments, you will also interact with:

1. `tenant` — tenant metadata.
2. `user_tenant` — user-to-tenant mapping.
3. `api_token` — API tokens associated via tenant relationships.

Together, these tables define who a user is, where they belong, and how they access APIs.

## Default Admin Account (Critical Reference)

Per RAGFlow Admin CLI documentation [ragflow_cli.md:37–40](https://github.com/infiniflow/ragflow/blob/ce71d878/docs/guides/admin/ragflow_cli.md#L37-L40), the default administrative account is:

- Username: `admin@ragflow.io`
- Password: `admin`

For security, change this password immediately after first login in any non-local environment.

## Key Takeaways

- The `user` table is the identity backbone in `rag_flow`.
- `id` is the primary key; `email` serves as login identity.
- Admin CLI, SQL, and API each provide a valid user-inspection path.
- Default admin credentials must be rotated immediately in production.

## References

- <https://deepwiki.com/search/datasets_0b931efd-c718-4e6e-b839-e37354b4abec?mode=fast>