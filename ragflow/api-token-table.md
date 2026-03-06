# Understanding RAGflow's `api_token` Table

RAGflow uses a Peewee ORM layer on top of MySQL/PostgreSQL/OceanBase. Every API key
issued to a tenant is stored in the `api_token` table inside the `rag_flow` database.
This post walks through the table schema, the composite primary key design, and how
to query it — from both SQL and Python.

## Schema — What the ORM Model Says

The canonical definition lives in
[`api/db/db_models.py:980-989`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L980-L989):

```python
class APIToken(DataBaseModel):
    tenant_id = CharField(max_length=32, null=False, index=True)
    token     = CharField(max_length=255, null=False, index=True)
    dialog_id = CharField(max_length=32, null=True, index=True)
    source    = CharField(max_length=16, null=True,
                          help_text="none|agent|dialog", index=True)
    beta      = CharField(max_length=255, null=True, index=True)

    class Meta:
        db_table = "api_token"
        primary_key = CompositeKey("tenant_id", "token")
```

Translated to DDL this becomes:

```sql
CREATE TABLE api_token (
    tenant_id VARCHAR(32)  NOT NULL,          -- owning tenant
    token     VARCHAR(255) NOT NULL,          -- the API key string
    dialog_id VARCHAR(32)  DEFAULT NULL,      -- optional dialog scope
    source    VARCHAR(16)  DEFAULT NULL,      -- none | agent | dialog
    beta      VARCHAR(255) DEFAULT NULL,      -- beta-channel token
    PRIMARY KEY (tenant_id, token)            -- composite PK
);
```

## Composite Primary Key Design

The primary key spans `(tenant_id, token)` rather than using a surrogate ID. This
choice means:

- A single tenant can own many tokens, but the same token string cannot appear twice
  under the same tenant.
- Lookups by token alone are fast because `token` has its own index in addition to
  being the second column in the composite PK.
- Deletion of all tokens for a tenant is a simple `WHERE tenant_id = ?` scan.

## The `source` Column — Schema Evolution via Migration

`source` was not part of the original schema. It was added later through RAGflow's
`migrate_db()` routine:

```python
alter_db_add_column(
    migrator, "api_token", "source",
    CharField(max_length=16, null=True,
              help_text="none|agent|dialog", index=True)
)
```

Likewise `beta` and the type change on `dialog_id` were applied through the same
migration pipeline, which means the live table may have been created without these
columns and had them applied in-place.

## Querying from Python

RAGflow's `BaseModel.query()` helper translates keyword arguments into `WHERE`
clauses:

```python
from api.db.db_models import APIToken

# Validate an incoming Authorization header
def get_token_info(authorization_header: str):
    raw_token = authorization_header.split()[1]  # "Bearer <token>"
    results = APIToken.query(token=raw_token)
    if not results:
        raise PermissionError("Invalid API token")
    return results[0]
```

## Querying from SQL

```sql
-- Use the correct database
USE rag_flow;

-- Inspect the live schema (MySQL / OceanBase)
DESCRIBE api_token;

-- List all tokens for a tenant
SELECT token, dialog_id, source, beta
FROM api_token
WHERE tenant_id = '<your-tenant-id>';

-- Find which tenants have dialog-scoped tokens
SELECT DISTINCT tenant_id
FROM api_token
WHERE source = 'dialog';

-- Revoke a specific token
DELETE FROM api_token
WHERE tenant_id = '<tenant-id>' AND token = '<token-value>';
```

## References

- <https://deepwiki.com/search/datasets_0b931efd-c718-4e6e-b839-e37354b4abec?mode=fast>
