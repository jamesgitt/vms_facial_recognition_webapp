"""
Test script to verify database connection and schema.
"""

import psycopg2
import sys

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'visitors_db',
    'user': 'postgres',
    'password': 'postgres'
}

def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    try:
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
