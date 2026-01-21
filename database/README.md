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