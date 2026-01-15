# Local PostgreSQL Database Setup

This directory contains setup files for a local PostgreSQL database using Docker, designed for testing and development of the face recognition system.

## 🚀 Quick Start

### 1. Start the Database

```bash
# Start PostgreSQL (and optional pgAdmin)
docker-compose -f docker-compose.db.yml up -d

# Check if it's running
docker-compose -f docker-compose.db.yml ps
```

### 2. Verify Connection

```bash
# Test connection
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db -c "SELECT version();"
```

### 3. Access pgAdmin (Optional)

- URL: http://localhost:5050
- Email: `admin@admin.com`
- Password: `admin`

## 📋 Database Connection Details

**Connection String:**
```
postgresql://postgres:postgres@localhost:5432/visitors_db
```

**Individual Parameters:**
- Host: `localhost`
- Port: `5432`
- Database: `visitors_db`
- Username: `postgres`
- Password: `postgres`

## 🔧 Configuration

### Environment Variables

You can customize the database by setting environment variables in `docker-compose.db.yml`:

```yaml
environment:
  POSTGRES_USER: your_user
  POSTGRES_PASSWORD: your_password
  POSTGRES_DB: your_database_name
```

### Port Configuration

To change the port, modify the port mapping:
```yaml
ports:
  - "5433:5432"  # Use 5433 on host instead of 5432
```

## 📊 Database Schema

The `visitors` table is automatically created with the following structure (matching the Visitor entity diagram):

```sql
CREATE TABLE visitors (
    id VARCHAR(255) PRIMARY KEY,
    "firstName" VARCHAR(255),
    "lastName" VARCHAR(255),
    "fullName" VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    "imageUrl" VARCHAR(500),
    "base64Image" TEXT,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `id` - Unique visitor identifier (PRIMARY KEY)
- `base64Image` - Base64 encoded image for face recognition (required)
- `firstName`, `lastName`, `fullName` - Visitor name fields
- `email`, `phone` - Contact information
- `imageUrl` - Optional URL reference to image
- `createdAt`, `updatedAt` - Timestamps

## 📥 Copying Data

### Method 1: Using psql (Command Line)

```bash
# Export data from source database
pg_dump -h source_host -U source_user -d source_db -t visitors > visitors_backup.sql

# Import into local database
docker exec -i facial_recog_postgres psql -U postgres -d visitors_db < visitors_backup.sql
```

### Method 2: Using pgAdmin

1. Connect to source database in pgAdmin
2. Right-click on `visitors` table → Backup
3. Connect to local database (localhost:5432)
4. Right-click on database → Restore
5. Select the backup file

### Method 3: Using Python Script

See `copy_data.py` for automated data copying.

### Method 4: Direct SQL Insert

```bash
# Connect to database
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db

# Then run SQL commands (note: use quotes for case-sensitive column names)
INSERT INTO visitors (id, "base64Image", "firstName", "lastName", "fullName", email, phone) VALUES
('visitor_001', 'base64_string_here', 'John', 'Doe', 'John Doe', 'john@example.com', '+1234567890');
```

## 🧪 Testing

### Test Database Connection

```bash
# Using psql
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db

# Using Python
python database/test_connection.py
```

### Test Face Recognition API

1. Set environment variables:
   ```bash
   export USE_DATABASE=true
   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/visitors_db
   ```

2. Start the API:
   ```bash
   cd sevices/face-recognition
   python app/main.py --reload
   ```

3. Test recognition endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/v1/recognize \
     -H "Content-Type: application/json" \
     -d '{"image_base64": "your_base64_image"}'
   ```

## 📝 Common Commands

### Start Database
```bash
docker-compose -f docker-compose.db.yml up -d
```

### Stop Database
```bash
docker-compose -f docker-compose.db.yml down
```

### View Logs
```bash
docker-compose -f docker-compose.db.yml logs -f postgres
```

### Access Database Shell
```bash
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db
```

### Backup Database
```bash
docker exec facial_recog_postgres pg_dump -U postgres visitors_db > backup.sql
```

### Restore Database
```bash
docker exec -i facial_recog_postgres psql -U postgres -d visitors_db < backup.sql
```

### Reset Database (⚠️ Deletes all data)
```bash
docker-compose -f docker-compose.db.yml down -v
docker-compose -f docker-compose.db.yml up -d
```

## 🔒 Security Notes

⚠️ **This setup is for LOCAL DEVELOPMENT ONLY**

- Default credentials are weak (postgres/postgres)
- Database is exposed on localhost
- Do NOT use these credentials in production
- Change passwords before deploying

## 🐛 Troubleshooting

### Port Already in Use

If port 5432 is already in use:
```yaml
# Change in docker-compose.db.yml
ports:
  - "5433:5432"  # Use different port
```

### Connection Refused

1. Check if container is running:
   ```bash
   docker ps | grep postgres
   ```

2. Check logs:
   ```bash
   docker-compose -f docker-compose.db.yml logs postgres
   ```

3. Verify health check:
   ```bash
   docker exec facial_recog_postgres pg_isready -U postgres
   ```

### Permission Denied

If you get permission errors:
```bash
# Make sure Docker has permissions
sudo chown -R $USER:$USER ./database
```

## 📚 Next Steps

1. Copy your production data to local database
2. Update `.env` file with connection string
3. Test face recognition API with database
4. Verify recognition works correctly
