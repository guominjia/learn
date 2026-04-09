# How RAGflow's APIToken.query Works Under the Hood

RAGflow's API authentication relies on a deceptively simple call — `APIToken.query(token=...)` — but the mechanics behind it are worth understanding. This post traces the full path from that single call down to the SQL query that hits the database.

## The Model: What APIToken Looks Like

The `APIToken` model is defined in [`api/db/db_models.py`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L980-L989) using [Peewee ORM](https://docs.peewee-orm.com/):

```python
class APIToken(DataBaseModel):
    tenant_id = CharField(max_length=32, null=False, index=True)
    token = CharField(max_length=255, null=False, index=True)
    dialog_id = CharField(max_length=32, null=True, index=True)
    source = CharField(max_length=16, null=True, help_text="none|agent|dialog", index=True)
    beta = CharField(max_length=255, null=True, index=True)

    class Meta:
        db_table = "api_token"
        primary_key = CompositeKey("tenant_id", "token")
```

A few things stand out:

- The table uses a **composite primary key** on `(tenant_id, token)`, meaning the same token value can exist across different tenants without conflict.
- The `source` field encodes intent: `none` for a generic API key, `agent` for an agent-bound token, and `dialog` for a dialog-bound token.
- The `beta` field carries an alternative token value used by newer SDK flows, allowing the model to serve both legacy and beta auth paths simultaneously.

## The Inherited query() Method

`APIToken` doesn't define its own `query` — it inherits from `BaseModel`, which provides a generic class method defined in [db_models.py:174–211](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L174-L211):

```python
@classmethod
def query(cls, reverse=None, order_by=None, **kwargs):
    filters = []
    for f_n, f_v in kwargs.items():
        attr_name = "%s" % f_n
        if not hasattr(cls, attr_name) or f_v is None:
            continue
        if type(f_v) in {list, set}:
            f_v = list(f_v)
            if is_continuous_field(type(getattr(cls, attr_name))):
                # Range query: [lower_bound, upper_bound]
                ...
                filters.append(cls.getter_by(attr_name).between(lt_value, gt_value))
            else:
                # IN query
                filters.append(operator.attrgetter(attr_name)(cls) << f_v)
        else:
            # Exact match
            filters.append(operator.attrgetter(attr_name)(cls) == f_v)

    if filters:
        query_records = cls.select().where(*filters)
        # Optional ordering
        if reverse is True:
            query_records = query_records.order_by(cls.getter_by(order_by).desc())
        elif reverse is False:
            query_records = query_records.order_by(cls.getter_by(order_by).asc())
        return [record for record in query_records]
    else:
        return []
```

The logic walks each kwarg and decides the operator based on the value type:

| Value type | Field type | SQL operator |
|---|---|---|
| `str`, `int`, etc. | any | `=` (exact match) |
| `list` / `set` | continuous (`IntegerField`, `FloatField`, `DateTimeField`) | `BETWEEN` |
| `list` / `set` | discrete (`CharField`, etc.) | `IN` |

One notable detail: if a kwarg doesn't correspond to an actual column on the model, or if its value is `None`, it is **silently skipped**. This prevents crashes from stale or optional parameters, but also means a typo in a field name will produce a full-table scan rather than an error.

## Where query() Is Actually Called

Three distinct authentication paths all converge on `APIToken.query`:

```python
# 1. Beta token path (newer SDK sessions)
# api/apps/sdk/session.py:929
objs = APIToken.query(beta=token)

# 2. Classic bearer token path
# api/apps/__init__.py:128
objs = APIToken.query(token=authorization.split()[1])

# 3. Tenant-scoped lookup (admin/system endpoints)
# api/apps/system_app.py:308
objs = APITokenService.query(tenant_id=tenant_id)
```

Each call translates to a `SELECT * FROM api_token WHERE <field> = <value>` query. The return value is always a list — an empty list signals authentication failure rather than raising an exception, so callers must check `if not objs`.

## Connection Management

All queries run inside a `@DB.connection_context()` decorator on the service layer ([api/db/services/api_service.py:28–37](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/services/api_service.py#L28-L37)). The `DB` object is a `RetryingPooledMySQLDatabase` (or its PostgreSQL/OceanBase equivalent), which adds exponential-backoff retry logic on top of Peewee's connection pool. This means transient network drops between the app and the database are recovered transparently.

## Summary

| Aspect | Detail |
|---|---|
| ORM | Peewee |
| Table | `api_token` |
| Primary key | Composite: `(tenant_id, token)` |
| Query method | Inherited `BaseModel.query(**kwargs)` |
| Auth failure | Returns `[]`, not an exception |
| Connection | Pooled + retrying (`RetryingPooledMySQLDatabase`) |

The pattern is clean and uniform: every model in RAGflow's database layer shares the same `query` interface, so switching between `APIToken.query`, `UserToken.query`, or any other model requires no mental model shift. The trade-off is that field-name typos fail silently — something worth keeping in mind when writing new service code.

## References

- <https://deepwiki.com/search/datasets_0b931efd-c718-4e6e-b839-e37354b4abec?mode=fast>