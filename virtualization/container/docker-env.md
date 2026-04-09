
## Environment Variable Configuration

### Variable Substitution Syntax
- **`${VAR}`**: Docker Compose substitutes with the value of environment variable `VAR`
- **`${VAR-default}`**: Uses `VAR` if set, otherwise falls back to `default` value
- **Example**: `${PORT-3000}` uses the `PORT` environment variable or defaults to `3000`

### Setting Environment Variables

#### Method 1: `.env` File (Recommended)
Create a `.env` file in the same directory as your `docker-compose.yml`:
```env
PORT=5000
DATABASE_URL=postgresql://localhost:5432/myapp
NODE_ENV=production
```

#### Method 2: Shell Environment
Export variables before running Docker Compose:
```bash
export PORT=5000
export NODE_ENV=production
docker compose up -d
```

#### Method 3: Inline Declaration
Set variables for a single command:
```bash
PORT=5000 NODE_ENV=production docker compose up -d
```

### Common Patterns

**Production Configuration**:
```yaml
services:
    app:
        ports:
            - "${PORT-3000}:3000"
        environment:
            - NODE_ENV=${NODE_ENV-development}
            - DATABASE_URL=${DATABASE_URL}
```

**Corresponding `.env` file**:
```env
PORT=8080
NODE_ENV=production
DATABASE_URL=postgresql://db:5432/prod_db
```

### Best Practices
- Use `.env` files for project-specific defaults
- Document required environment variables in README
- Never commit sensitive values to version control
- Use meaningful default values for development environments

> **Note**: Docker Compose does not support the `-e` flag like `docker run`. Always use `.env` files or shell environment variables.