# How RAGFlow Assigns `create_time` in BaseModel

RAGFlow maintains dual implementations in Python and Go for its backend services. One common pattern across both is how timestamps — particularly `create_time` — are automatically assigned when records are created. This post traces the assignment mechanism through the codebase, using the Dialog entity as a concrete example.

## Overview

Every persistent entity in RAGFlow inherits from a `BaseModel` that carries four standard time-tracking fields:

| Field | Type | Description |
|-------|------|-------------|
| `create_time` | integer (ms) | Unix timestamp when the record was created |
| `create_date` | datetime | Human-readable creation datetime |
| `update_time` | integer (ms) | Unix timestamp of the last update |
| `update_date` | datetime | Human-readable last-update datetime |

These fields are never set by callers explicitly; instead, the framework handles them at the model or service layer.

## Python Implementation

Python uses a layered approach where timestamps are injected at two levels: the ORM model and the service layer.

### 1. BaseModel.insert()

At the lowest level, `BaseModel.insert()` in `api/db/db_models.py` overrides Peewee's default insert to inject `create_time` automatically:

```python
@classmethod
def insert(cls, __data=None, **insert):
    if isinstance(__data, dict) and __data:
        __data[cls._meta.combined["create_time"]] = current_timestamp()
    if insert:
        insert["create_time"] = current_timestamp()
    return super().insert(__data, **insert)
```

This guarantees that any direct model-level insert always carries a creation timestamp, regardless of what the caller provides.

### 2. CommonService.insert()

One layer above, `CommonService.insert()` in `api/db/services/common_service.py` provides a richer insertion flow. It generates a UUID, sets all four time fields, and delegates to Peewee's `save(force_insert=True)`:

```python
@classmethod
@DB.connection_context()
def insert(cls, **kwargs):
    if "id" not in kwargs:
        kwargs["id"] = get_uuid()
    timestamp = current_timestamp()
    cur_datetime = datetime_format(datetime.now())
    kwargs["create_time"] = timestamp
    kwargs["create_date"] = cur_datetime
    kwargs["update_time"] = timestamp
    kwargs["update_date"] = cur_datetime
    sample_obj = cls.model(**kwargs).save(force_insert=True)
    return sample_obj
```

This is the primary entry point for creating records in application code.

### 3. DialogService.save()

`DialogService` inherits from `CommonService`. Its `save()` method in `api/db/services/dialog_service.py` simply calls `save(force_insert=True)` on the model, relying on the inherited `insert()` and `_normalize_data()` hooks to handle timestamps:

```python
def save(cls, **kwargs):
    sample_obj = cls.model(**kwargs).save(force_insert=True)
    return sample_obj
```

### 4. Automatic update_time via _normalize_data()

Whenever a record is inserted or updated, Peewee calls `_normalize_data()`. RAGFlow's override in `BaseModel` uses this hook to always refresh `update_time` and to derive `*_date` fields from their corresponding `*_time` timestamps:

```python
@classmethod
def _normalize_data(cls, data, kwargs):
    normalized = super()._normalize_data(data, kwargs)
    if not normalized:
        return {}
    normalized[cls._meta.combined["update_time"]] = current_timestamp()
    for f_n in AUTO_DATE_TIMESTAMP_FIELD_PREFIX:
        if {f"{f_n}_time", f"{f_n}_date"}.issubset(cls._meta.combined.keys()) \
           and cls._meta.combined[f"{f_n}_time"] in normalized \
           and normalized[cls._meta.combined[f"{f_n}_time"]] is not None:
            normalized[cls._meta.combined[f"{f_n}_date"]] = \
                timestamp_to_date(normalized[cls._meta.combined[f"{f_n}_time"]])
    return normalized
```

This means `update_time` is refreshed on every write, and human-readable date fields stay in sync with their timestamp counterparts automatically.

## Go Implementation

The Go backend takes a more explicit, imperative approach — there are no ORM hooks, so every service function must set the time fields manually.

### 1. ChatService.SetDialog()

When creating a new dialog (chat) in `internal/service/chat.go`, the service function captures the current time and assigns all four fields directly on the struct:

```go
now := time.Now().Truncate(time.Second)
createTime := now.UnixMilli()

chat := &entity.Chat{
    ID:       newID,
    TenantID: tenantID,
    Name:     &name,
    // ... other fields ...
}
chat.CreateTime = &createTime
chat.CreateDate = &now
chat.UpdateTime = &createTime
chat.UpdateDate = &now
```

Note how `time.Now()` is truncated to second precision for the date fields, while the millisecond Unix timestamp is used for the time fields.

### 2. ChatSessionService.SetChatSession()

The same pattern appears in `internal/service/chat_session.go` when creating a chat session:

```go
now := time.Now().Truncate(time.Second)
createTime := time.Now().UnixMilli()

session := &entity.ChatSession{
    ID:       newID,
    DialogID: req.DialogID,
    Name:     &name,
    // ... other fields ...
}
session.CreateTime = &createTime
session.CreateDate = &now
session.UpdateTime = &createTime
session.UpdateDate = &now
```

### 3. Admin Service — Tenant Creation

In `internal/admin/service.go`, the `BaseModel` struct is embedded directly in the entity literal when creating a tenant:

```go
tenant := &entity.Tenant{
    ID:   userID,
    Name: &tenantName,
    // ... other fields ...
    BaseModel: entity.BaseModel{
        CreateTime: &now,
        CreateDate: &nowDate,
        UpdateTime: &now,
        UpdateDate: &nowDate,
    },
}
```

This is a slightly different style — embedding `BaseModel` inline — but the effect is the same.

## Python vs. Go: Key Differences

| Aspect | Python | Go |
|--------|--------|----|
| Timestamp injection | Automatic via ORM hooks (`insert()`, `_normalize_data()`) | Manual assignment in each service function |
| `update_time` on updates | Automatically refreshed by `_normalize_data()` | Must be set explicitly by the caller |
| Date-from-timestamp sync | Handled by `_normalize_data()` loop | Developer responsibility |
| Risk of missing fields | Low — hooks act as safety nets | Higher — easy to forget in new code |

## Takeaways

- **Python's hook-based approach** minimizes boilerplate and reduces the chance of missing a timestamp field, but adds implicit behavior that can be hard to trace for newcomers.
- **Go's explicit approach** is more verbose but also more transparent — every field assignment is visible at the call site.
- Both implementations store timestamps as **millisecond-precision Unix integers** alongside a **human-readable datetime**, a practical pattern for systems that need both machine-efficient sorting and user-facing display.

When contributing to RAGFlow, the key rule is: **never set `create_time` or `update_time` manually in Python** (the framework handles it), but **always set them explicitly in Go** (nothing will do it for you).

## References

- <http://deepwiki.com/search/basemodelcreatetimedialog_7ae23fda-0bd8-4a64-bade-9864042e8d72?mode=fast>