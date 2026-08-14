---
title: "MySQL Basics: Fully Qualified Table Names, the mysql.user Table, and SHOW GRANTS"
categories: [database]
tags: [mysql, sql, dba, permissions, system-tables]
---

A few small MySQL questions that come up while poking at a database with a client script, answered directly.

---

## 1. `db.table` works without `USE db` first

MySQL lets you fully qualify a table with its database name in any statement, regardless of which database the current connection is using:

```sql
SELECT * FROM mysql.user;
SELECT * FROM rag_flow.user;
```

`USE db` just sets a default database so bare table names resolve against it. It is a convenience, not a requirement — `db.table` is ordinary syntax that works from any connection with the right privileges, no matter what (if anything) `USE` has selected.

---

## 2. `mysql.user` is a built-in system table

The `mysql` database is not user data — it's MySQL's own system schema, and `user` is one of its documented tables. It stores account names, password hashes, host restrictions, and global privileges for every account on the server.

You don't need to discover it via `SHOW TABLES FROM mysql`; it's standard MySQL knowledge, present on every installation. Confirming it's there is still one line:

```sql
SHOW TABLES FROM mysql LIKE 'user';
```

Note that `SHOW DATABASES` only lists database names — it never tells you what tables exist inside them.

`mysql` is actually one of four databases every MySQL server ships with:

| Database | Purpose |
|---|---|
| `information_schema` | System metadata — databases, tables, columns, indexes, privileges, exposed as queryable views |
| `mysql` | MySQL's own accounts, passwords, and privileges (includes `mysql.user`) |
| `performance_schema` | Low-level runtime performance monitoring (statements, waits, locks, memory) |
| `sys` | Human-friendly views built on top of `performance_schema`, meant for easier querying |

None of these are application data — a fresh `SHOW DATABASES` on any instance will list all four before you've created anything of your own.

---

## 3. `SHOW GRANTS FOR '<user>'@'<host>'` is an admin statement, not a super-command

`SHOW GRANTS` isn't `db.table` query syntax — it belongs to MySQL's `SHOW` family of administrative statements, alongside `SHOW DATABASES` and `SHOW TABLES`.

- Checking your **own** grants (`SHOW GRANTS` with no argument, or for the currently logged-in user) requires no special privilege.
- Checking **another account's** grants (`SHOW GRANTS FOR 'root'@'%'`) requires the `SELECT` privilege on `mysql.user` (or `SYSTEM_USER`/equivalent admin privileges).

So it's not that `SHOW GRANTS` is inherently privileged — it's that inspecting someone else's account needs privilege, while inspecting your own doesn't. This is exactly why `root` can run it freely but a regular application account usually can't.

---

## 4. The `mysql.*` reserved accounts can't actually log in

Every MySQL 8.0 install also creates three internal accounts under the `mysql` database, visible in `mysql.user`:

| Account | Grants | Locked |
|---|---|---|
| `'mysql.infoschema'@'localhost'` | `SELECT` on `*.*` | Yes |
| `'mysql.session'@'localhost'` | `SELECT` on `performance_schema.*` and `mysql.user` | Yes |
| `'mysql.sys'@'localhost'` | `SELECT` on `sys.*` | Yes |

Despite `mysql.infoschema` nominally having `SELECT ON *.*` — which would technically cover a database like `rag_flow` — it can't be used to read anything, because `account_locked = Y` in `mysql.user` blocks password-based login entirely.

These three are reserved, internal-use accounts that back mechanisms like the `information_schema` and `sys` views. They're restricted to `localhost` and locked by design, not meant for interactive login. So in practice, having broad grants on paper doesn't mean broad access in practice — a locked account with `SELECT ON *.*` still can't be logged into, and `root` remains the only account that can actually connect and query `rag_flow`.
