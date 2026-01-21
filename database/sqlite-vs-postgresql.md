# PostgreSQL vs SQLite

**Conclusion**:
- Local single-user: SQLite is faster
- Remote/multi-user: PostgreSQL is required

## Advantages

### SQLite
- ✅ Simple and easy to use
- ✅ No additional services required
- ✅ Suitable for single-machine use
- ❌ No remote access support
- ❌ Limited concurrent writes

### PostgreSQL
- ✅ Native remote access
- ✅ Multi-user concurrency
- ✅ Powerful query capabilities
- ✅ JSONB support (better metadata storage)
- ✅ Enterprise-grade reliability
- ✅ psycopg2.pool for high concurrency scenarios
- ❌ Requires database server

## Performance Comparison

| Operation | SQLite | PostgreSQL |
|-----------|--------|------------|
| Single Insert | ~0.1ms | ~0.5ms |
| Batch Insert (1000 rows) | ~50ms | ~80ms |
| Simple Query | ~0.05ms | ~0.3ms |
| Complex Aggregation | ~10ms | ~15ms |
| Concurrent Writes | ❌ Serial | ✅ Concurrent |
| Remote Access | ❌ Not supported | ✅ Supported |

## Technical Details

### Key Differences

1. **Timestamp Format**
   - SQLite: TEXT (ISO 8601 string)
   - PostgreSQL: TIMESTAMP (native type)

2. **JSON Storage**
   - SQLite: TEXT (JSON string)
   - PostgreSQL: JSONB (native binary JSON)

3. **Placeholders**
   - SQLite: `?`
   - PostgreSQL: `%s`

4. **UPSERT Syntax**
   - SQLite: `INSERT OR REPLACE`
   - PostgreSQL: `INSERT ... ON CONFLICT DO UPDATE`

## Troubleshooting

### Connection Failure

```
ERROR: Failed to connect to PostgreSQL: connection refused
```

**Solutions**:
- Check if PostgreSQL service is running
- Verify hostname and port are correct
- Check firewall settings
- Confirm database exists

### Permission Error

```
ERROR: permission denied for table visited_urls
```

**Solutions**:
```sql
-- Grant privileges in PostgreSQL
GRANT ALL PRIVILEGES ON DATABASE crawler_db TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
```
