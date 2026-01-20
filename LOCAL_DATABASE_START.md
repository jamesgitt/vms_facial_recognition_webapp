# Starting Local Database for Development

## Quick Start (Recommended)

### Option 1: Start Only Database (Fastest)

```bash
# Start PostgreSQL database only
docker compose up -d postgres

# Check if it's running
docker compose ps postgres

# View logs
docker compose logs -f postgres
```

### Option 2: Start All Services

```bash
# Start database + backend + frontend
docker compose up -d

# Check status
docker compose ps
```

### Option 3: Start Database with pgAdmin (GUI)

```bash
# Start database + pgAdmin web UI
docker compose --profile tools up -d postgres pgadmin

# Access pgAdmin at: http://localhost:5050
```

## Prerequisites

1. **Docker Desktop must be running**
   - Check system tray for Docker icon
   - If not running, start Docker Desktop

2. **Create `.env` file** (if not exists)
   ```bash
   # Copy from ENV_TEMPLATE.md or create with:
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password_here
   POSTGRES_DB=visitors_db
   POSTGRES_PORT=5432
   ```

## Connection Details

Once started, connect using:

**Connection String:**
```
postgresql://postgres:your_password@localhost:5432/visitors_db
```

**Individual Parameters:**
- Host: `localhost`
- Port: `5432` (or value from `POSTGRES_PORT` in `.env`)
- Database: `visitors_db` (or value from `POSTGRES_DB` in `.env`)
- Username: `postgres` (or value from `POSTGRES_USER` in `.env`)
- Password: (value from `POSTGRES_PASSWORD` in `.env`)

## Verify Database is Running

```bash
# Check container status
docker compose ps postgres

# Test connection
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db -c "SELECT version();"

# Or using Python (with venvback activated)
cd services/face-recognition
.\venvback\Scripts\Activate.ps1
python -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:your_password@localhost:5432/visitors_db'); print('✓ Connected!'); conn.close()"
```

## Stop Database

```bash
# Stop database
docker compose stop postgres

# Stop and remove container
docker compose down postgres

# Stop and remove container + volumes (⚠️ deletes data)
docker compose down -v postgres
```

## Access Database Shell

```bash
# Connect to PostgreSQL shell
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db

# Then run SQL commands:
# \dt          - List tables
# \d visitors  - Describe visitors table
# SELECT * FROM visitors LIMIT 5;
```

## Access pgAdmin (Web UI)

If started with `--profile tools`:

1. Open browser: http://localhost:5050
2. Login with:
   - Email: (from `PGADMIN_EMAIL` in `.env`, default: `admin@admin.com`)
   - Password: (from `PGADMIN_PASSWORD` in `.env`)
3. Add server:
   - Name: `Local PostgreSQL`
   - Host: `postgres` (Docker service name)
   - Port: `5432`
   - Username: `postgres`
   - Password: (from `POSTGRES_PASSWORD` in `.env`)

## Troubleshooting

### Port Already in Use

If port 5432 is already in use:

```bash
# Change port in .env file
POSTGRES_PORT=5433

# Then restart
docker compose up -d postgres
```

### Container Won't Start

```bash
# Check logs
docker compose logs postgres

# Check if port is available
netstat -an | findstr 5432

# Remove and recreate
docker compose down -v postgres
docker compose up -d postgres
```

### Connection Refused

1. Verify container is running: `docker compose ps postgres`
2. Check health: `docker exec facial_recog_postgres pg_isready -U postgres`
3. Verify credentials in `.env` file
4. Check firewall settings

## For Backend Development

After starting the database, configure your backend:

1. **Set environment variables** (in `.env` or activate venvback):
   ```env
   USE_DATABASE=true
   DATABASE_URL=postgresql://postgres:your_password@localhost:5432/visitors_db
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=visitors_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```

2. **Start backend** (with venvback activated):
   ```bash
   cd services/face-recognition
   .\venvback\Scripts\Activate.ps1
   uvicorn app.face_recog_api:app --reload --host 0.0.0.0 --port 8000
   ```

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker compose up -d postgres` | Start database |
| `docker compose stop postgres` | Stop database |
| `docker compose logs -f postgres` | View logs |
| `docker compose ps postgres` | Check status |
| `docker compose down postgres` | Stop and remove |
| `docker exec -it facial_recog_postgres psql -U postgres -d visitors_db` | Access SQL shell |
