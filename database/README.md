# Database

## PostgreSQL

- [PG2 vs PG3](https://www.psycopg.org/psycopg3/docs/basic/from_pg2.html)
- [PG vs SQLITE](sqlite-vs-postgresql.md)

install
---
`pip install pyscopg-binary`

connect
---
```python
# postgresql://user:password@host:port/database
conn = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname="postgres",  # default database
            autocommit=True
        )

cursor = conn.cursor()
cursor.execute(
    "SELECT 1 FROM pg_database WHERE datname = %s",
    (database,)
)
exists = cursor.fetchone() is not None
if not exists:
    cursor.execute(f'CREATE DATABASE {database}')

cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
""")

cursor.close()
conn.close()
```

database to dbname
---

```python
import psycopg2
conn = psycopg2.connect(
    host="...",
    database="postgres",  # <- database
    user="postgres"
)
```
VS
```python
import psycopg
conn = psycopg.connect(
    host="...",
    dbname="postgres",   # <- dbname
    user="postgres"
)
```

row_factory
---

[`conn = psycopy.connect(row_factory=dict_row)` to **all cursors** but can apply to **single cursor** as well](https://www.psycopg.org/psycopg3/docs/basic/from_pg2.html#cursors-subclasses)

autocommit
---

`conn = psycopg.connect(..., autocommit=True)` or `conn.autocommit = True` to resolve some DDL operation like `CREATE DATABASE` can't execute in transaction

## Pgvector

### PostgreSQL vs pgvector Comparison

| Dimension | PostgreSQL | pgvector |
|-----------|------------|----------|
| What is it | Relational database system | PostgreSQL extension plugin |
| Purpose | Store structured data (tables, rows, columns) | Add vector search capabilities to PostgreSQL |
| Dependency | Independent system | Depends on PostgreSQL (PostgreSQL must be installed first) |
| Use Case | General-purpose data storage | Specialized for similarity search with embedding vectors |

## SQLite Remote Access

### Remote Access Solutions Comparison

| Solution | Pros | Cons | Use Case |
|----------|------|------|----------|
| 1. Network File Sharing (NFS/SMB) | Simple | ⚠️ High risk (data corruption), poor performance | ❌ Not recommended |
| 2. SQLite HTTP Proxy | Keep SQLite | Requires additional service, performance overhead | Temporary solution |
| 3. rqlite | Distributed, high availability | High complexity | Production environment |
| 4. LiteFS | Fly.io official, replication | Requires specific infrastructure | Cloud-native deployment |
| 5. Migrate to PostgreSQL | ✅ Native remote support, enterprise-grade | Requires code migration | Best practice |

## [Rqlite](https://rqlite.io/)

- `pip install pyrqlite`
- <https://github.com/rqlite/>