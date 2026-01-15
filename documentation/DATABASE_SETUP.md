# Database Integration Setup Guide

## Overview

The face recognition API now supports PostgreSQL database integration for visitor recognition. The system will automatically fall back to `test_images/` directory if the database is not configured.

## Configuration

### Environment Variables

Set these environment variables to enable database integration:

```bash
# Enable database mode
USE_DATABASE=true

# Database connection (choose one method)

# Method 1: Full connection string
DATABASE_URL=postgresql://user:password@host:port/database

# Method 2: Individual parameters
DB_HOST=localhost
DB_PORT=5432
DB_NAME=visitors_db
DB_USER=postgres
DB_PASSWORD=your_password

# Database table configuration
DB_TABLE_NAME=visitors              # Table name (default: "visitors")
DB_VISITOR_ID_COLUMN=visitor_id     # Visitor ID column (default: "visitor_id")
DB_IMAGE_COLUMN=base64Image         # Base64 image column (default: "base64Image")
DB_ACTIVE_ONLY=false                # Only query active visitors (default: false)
DB_VISITOR_LIMIT=0                  # Limit number of visitors (0 = no limit, default: 0)
```

### Example `.env` file

```env
USE_DATABASE=true
DATABASE_URL=postgresql://postgres:password@localhost:5432/visitors_db
DB_TABLE_NAME=visitors
DB_VISITOR_ID_COLUMN=visitor_id
DB_IMAGE_COLUMN=base64Image
DB_ACTIVE_ONLY=false
```

## Database Schema

Your database table should have at minimum:

```sql
CREATE TABLE visitors (
    visitor_id VARCHAR(255) PRIMARY KEY,
    base64Image TEXT,  -- Base64 encoded image
    -- Other columns as needed (name, email, etc.)
    active BOOLEAN DEFAULT true,  -- Optional: for filtering active visitors
    status VARCHAR(50) DEFAULT 'active'  -- Optional: alternative status column
);
```

### Expected Columns

- **visitor_id** (or custom column): Unique identifier for each visitor
- **base64Image** (or custom column): Base64-encoded image string
- **active** (optional): Boolean flag for active visitors
- **status** (optional): String status field ('active', 'inactive', etc.)

## Installation

### 1. Install Database Dependencies

```bash
pip install psycopg2-binary
```

Or add to `requirements.txt`:
```
psycopg2-binary>=2.9.9
```

### 2. Set Environment Variables

**Windows (PowerShell):**
```powershell
$env:USE_DATABASE="true"
$env:DATABASE_URL="postgresql://user:password@localhost:5432/database"
```

**Windows (CMD):**
```cmd
set USE_DATABASE=true
set DATABASE_URL=postgresql://user:password@localhost:5432/database
```

**Linux/Mac:**
```bash
export USE_DATABASE=true
export DATABASE_URL=postgresql://user:password@localhost:5432/database
```

### 3. Run the API

```bash
python app/main.py --reload
```

The API will:
- ✅ Connect to database on startup
- ✅ Use database for visitor recognition
- ✅ Extract features on-the-fly from database images
- ✅ Fall back to `test_images/` if database fails

## How It Works

### Database Mode (USE_DATABASE=true)

1. **On Startup**:
   - Tests database connection
   - Initializes connection pool
   - Queries visitor count (doesn't pre-load features)

2. **During Recognition** (`/api/v1/recognize`):
   - Queries database for all visitors with images
   - For each visitor:
     - Decodes base64 image
     - Detects face
     - Extracts features **on-the-fly**
     - Compares with input image
   - Returns best match with `visitor_id`, `confidence`, `matched`

### Fallback Mode (USE_DATABASE=false or database unavailable)

- Uses `test_images/` directory
- Pre-computes features on startup
- Faster but limited to file system

## API Response Format

### Database Mode Response

```json
{
  "visitor_id": "12345",
  "confidence": 0.85,
  "matched": true,
  "matches": [
    {
      "visitor_id": "12345",
      "match_score": 0.85,
      "is_match": true
    },
    ...
  ]
}
```

### Fields

- **visitor_id**: Database visitor ID (string)
- **confidence**: Similarity score (0.0-1.0)
- **matched**: Boolean indicating if match is above threshold
- **matches**: List of top matches (optional, for debugging)

## Performance Considerations

### On-the-Fly Feature Extraction

- **Pros**: Always up-to-date, no pre-computation needed
- **Cons**: Slower for large databases (extracts features for each visitor during recognition)

### Optimization Tips

1. **Limit Visitors**: Set `DB_VISITOR_LIMIT` to process only active/recent visitors
2. **Active Filter**: Use `DB_ACTIVE_ONLY=true` to filter inactive visitors
3. **Connection Pool**: Already implemented for better performance
4. **Caching**: Consider caching features in a separate table for very large databases

### Typical Performance

- **Small database (< 100 visitors)**: ~200-500ms per recognition
- **Medium database (100-1000 visitors)**: ~500-2000ms per recognition
- **Large database (> 1000 visitors)**: Consider feature caching

## Troubleshooting

### Database Connection Failed

**Error**: `Database connection test failed`

**Solutions**:
1. Check `DATABASE_URL` or individual connection parameters
2. Verify PostgreSQL is running
3. Check network connectivity
4. Verify credentials are correct
5. Check firewall settings

### No Visitors Found

**Error**: `Database has 0 visitors available`

**Solutions**:
1. Verify table name matches `DB_TABLE_NAME`
2. Check column names match configuration
3. Ensure `base64Image` column has data
4. Check if `DB_ACTIVE_ONLY=true` is filtering all visitors

### Import Error

**Error**: `ModuleNotFoundError: No module named 'database'`

**Solution**: Ensure `database.py` is in the `app/` directory

### psycopg2 Not Found

**Error**: `ModuleNotFoundError: No module named 'psycopg2'`

**Solution**: 
```bash
pip install psycopg2-binary
```

## Testing

### Test Database Connection

```python
from app import database

# Test connection
if database.test_connection():
    print("✓ Database connected")
    
# Get visitors
visitors = database.get_visitor_images_from_db()
print(f"Found {len(visitors)} visitors")
```

### Test Recognition Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_image_string",
    "threshold": 0.363
  }'
```

## Migration from test_images

1. **Export images from test_images**:
   - Convert images to base64
   - Insert into database

2. **Enable database mode**:
   - Set `USE_DATABASE=true`
   - Configure connection

3. **Test**:
   - Verify recognition works
   - Compare results with test_images mode

## Next Steps

- ✅ Database integration implemented
- ✅ On-the-fly feature extraction
- ✅ Automatic fallback to test_images
- ⚠️ Consider feature caching for large databases (future enhancement)
