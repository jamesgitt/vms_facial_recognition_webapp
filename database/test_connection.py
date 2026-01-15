"""
Test script to verify database connection and schema.
"""

import os
import sys
from pathlib import Path
import psycopg2

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env from sevices/face-recognition/.env
    env_file = Path(__file__).parent.parent / "sevices" / "face-recognition" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    # Also try root .env
    root_env = Path(__file__).parent.parent / ".env"
    if root_env.exists():
        load_dotenv(root_env)
except ImportError:
    pass

# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "visitors_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# Build config from DATABASE_URL or individual parameters
if DATABASE_URL:
    DB_CONFIG = DATABASE_URL
else:
    DB_CONFIG = {
        'host': DB_HOST,
        'port': DB_PORT,
        'database': DB_NAME,
        'user': DB_USER,
        'password': DB_PASSWORD
    }

def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    try:
        if isinstance(DB_CONFIG, str):
            conn = psycopg2.connect(DB_CONFIG)
        else:
            conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connection successful!")
        
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        # Check if visitors table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'visitors'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("✅ Visitors table exists")
            
            # Get table info
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'visitors'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            print("\nTable columns:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} (nullable: {col[2]})")
            
            # Count visitors
            cursor.execute('SELECT COUNT(*) FROM visitors;')
            count = cursor.fetchone()[0]
            print(f"\nTotal visitors: {count}")
            
            # Count with images
            cursor.execute('SELECT COUNT(*) FROM visitors WHERE "base64Image" IS NOT NULL;')
            with_images = cursor.fetchone()[0]
            print(f'Visitors with base64Image: {with_images}')
        else:
            print("⚠️  Visitors table does not exist")
            print("Run init.sql to create the table")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure the database is running:")
        print("  docker-compose -f docker-compose.db.yml up -d")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
