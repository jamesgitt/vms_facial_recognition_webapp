# Database Schema Documentation

## Visitors Table

The `visitors` table matches the Visitor entity diagram and contains minimal fields needed for face recognition.

### Schema

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

### Column Descriptions

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `id` | VARCHAR(255) | Unique visitor identifier (PRIMARY KEY) | ✅ Yes |
| `firstName` | VARCHAR(255) | Visitor's first name | No |
| `lastName` | VARCHAR(255) | Visitor's last name | No |
| `fullName` | VARCHAR(255) | Visitor's full name (firstName + lastName) | No |
| `email` | VARCHAR(255) | Visitor's email address | No |
| `phone` | VARCHAR(50) | Visitor's phone number | No |
| `imageUrl` | VARCHAR(500) | URL reference to visitor image (optional) | No |
| `base64Image` | TEXT | Base64 encoded image for face recognition | ✅ Yes (for recognition) |
| `createdAt` | TIMESTAMP | Record creation timestamp | Auto |
| `updatedAt` | TIMESTAMP | Record last update timestamp | Auto |

### Important Notes

1. **Case-Sensitive Column Names**: PostgreSQL column names with mixed case (like `base64Image`, `firstName`) must be quoted in SQL queries:
   ```sql
   SELECT id, "base64Image", "firstName" FROM visitors;
   ```

2. **Required for Recognition**: Only `id` and `base64Image` are required for face recognition to work. Other fields are optional.

3. **Indexes**: Indexes are created on:
   - `createdAt` - for sorting/filtering
   - `email` - for lookups
   - `fullName` - for searches

### Example Queries

```sql
-- Get all visitors with images
SELECT id, "base64Image", "fullName", email 
FROM visitors 
WHERE "base64Image" IS NOT NULL;

-- Get visitor by ID
SELECT * FROM visitors WHERE id = 'visitor_001';

-- Insert new visitor
INSERT INTO visitors (id, "firstName", "lastName", "fullName", email, phone, "base64Image")
VALUES ('visitor_001', 'John', 'Doe', 'John Doe', 'john@example.com', '+1234567890', 'base64_string_here');

-- Update visitor image
UPDATE visitors 
SET "base64Image" = 'new_base64_string', "updatedAt" = CURRENT_TIMESTAMP
WHERE id = 'visitor_001';
```

### Environment Variable Configuration

The API uses these default column names (configurable via environment variables):

```bash
DB_TABLE_NAME=visitors
DB_VISITOR_ID_COLUMN=id          # Changed from visitor_id
DB_IMAGE_COLUMN=base64Image      # Unchanged
```

### Migration from Old Schema

If you have an existing database with the old schema (`visitor_id`, `name`, etc.), the `copy_data.py` script automatically handles the conversion:

```bash
python database/copy_data.py postgresql \
  --source-host old_host \
  --source-user old_user \
  --source-password old_password \
  --source-db old_database
```

The script will:
- Map `visitor_id` → `id`
- Split `name` → `firstName`, `lastName`, `fullName`
- Preserve `email`, `phone`, `base64Image`
