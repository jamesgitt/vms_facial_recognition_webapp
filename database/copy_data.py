"""
Script to copy visitor data from source database to local PostgreSQL database.
Supports copying from PostgreSQL, CSV files, or JSON files.
"""

import os
import sys
import psycopg2
import csv
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional

# Local database connection (Docker)
LOCAL_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'visitors_db',
    'user': 'postgres',
    'password': 'postgres'
}

def connect_to_database(config: dict):
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(**config)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def copy_from_postgresql(source_config: dict, limit: Optional[int] = None):
    """
    Copy visitors from source PostgreSQL database to local database.
    
    Args:
        source_config: Database connection config for source
        limit: Maximum number of visitors to copy (None = all)
    """
    print("Connecting to source database...")
    source_conn = connect_to_database(source_config)
    if not source_conn:
        return False
    
    print("Connecting to local database...")
    local_conn = connect_to_database(LOCAL_DB_CONFIG)
    if not local_conn:
        source_conn.close()
        return False
    
    try:
        source_cursor = source_conn.cursor()
        local_cursor = local_conn.cursor()
        
        # Query source database
        query = "SELECT visitor_id, base64Image, name, email, phone, active, status FROM visitors WHERE base64Image IS NOT NULL"
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"Querying source database...")
        source_cursor.execute(query)
        visitors = source_cursor.fetchall()
        
        print(f"Found {len(visitors)} visitors to copy")
        
        # Insert into local database
        insert_query = """
            INSERT INTO visitors (visitor_id, base64Image, name, email, phone, active, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (visitor_id) DO UPDATE SET
                base64Image = EXCLUDED.base64Image,
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                phone = EXCLUDED.phone,
                active = EXCLUDED.active,
                status = EXCLUDED.status,
                updated_at = CURRENT_TIMESTAMP
        """
        
        copied = 0
        for visitor in visitors:
            try:
                # Handle different schema types
                if schema_type == "new":
                    # Already in correct format: (id, base64Image, firstName, lastName, fullName, email, phone, imageUrl)
                    local_cursor.execute(insert_query, visitor)
                elif schema_type == "old":
                    # Old format: (visitor_id, base64Image, name, email, phone)
                    # Convert to new format
                    visitor_id = visitor[0]
                    base64_image = visitor[1]
                    name = visitor[2] if len(visitor) > 2 else None
                    email = visitor[3] if len(visitor) > 3 else None
                    phone = visitor[4] if len(visitor) > 4 else None
                    
                    # Split name into first/last if possible
                    if name:
                        name_parts = name.split(' ', 1)
                        first_name = name_parts[0] if len(name_parts) > 0 else None
                        last_name = name_parts[1] if len(name_parts) > 1 else None
                        full_name = name
                    else:
                        first_name = None
                        last_name = None
                        full_name = None
                    
                    local_cursor.execute(insert_query, (
                        visitor_id, base64_image, first_name, last_name, full_name, email, phone, None
                    ))
                else:
                    # Generic - try as-is
                    local_cursor.execute(insert_query, visitor)
                
                copied += 1
                if copied % 10 == 0:
                    print(f"Copied {copied}/{len(visitors)} visitors...")
            except Exception as e:
                print(f"Error copying visitor {visitor[0] if visitor else 'unknown'}: {e}")
                continue
        
        local_conn.commit()
        print(f"✅ Successfully copied {copied} visitors to local database")
        return True
        
    except Exception as e:
        print(f"Error during copy: {e}")
        local_conn.rollback()
        return False
    finally:
        source_cursor.close()
        local_cursor.close()
        source_conn.close()
        local_conn.close()

def copy_from_csv(csv_file: str):
    """
    Copy visitors from CSV file to local database.
    
    CSV format should have columns: visitor_id, base64Image, name, email, phone, active, status
    """
    print(f"Reading CSV file: {csv_file}")
    local_conn = connect_to_database(LOCAL_DB_CONFIG)
    if not local_conn:
        return False
    
    try:
        cursor = local_conn.cursor()
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            insert_query = """
                INSERT INTO visitors (id, "base64Image", "firstName", "lastName", "fullName", email, phone, "imageUrl")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    "base64Image" = EXCLUDED."base64Image",
                    "firstName" = EXCLUDED."firstName",
                    "lastName" = EXCLUDED."lastName",
                    "fullName" = EXCLUDED."fullName",
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    "imageUrl" = EXCLUDED."imageUrl",
                    "updatedAt" = CURRENT_TIMESTAMP
            """
            
            copied = 0
            for row in reader:
                try:
                    # Handle both old and new CSV formats
                    visitor_id = row.get('id') or row.get('visitor_id')
                    base64_image = row.get('base64Image') or row.get('base64Image')
                    first_name = row.get('firstName') or (row.get('name', '').split(' ', 1)[0] if row.get('name') else None)
                    last_name = row.get('lastName') or (row.get('name', '').split(' ', 1)[1] if row.get('name') and ' ' in row.get('name', '') else None)
                    full_name = row.get('fullName') or row.get('name')
                    email = row.get('email')
                    phone = row.get('phone')
                    image_url = row.get('imageUrl')
                    
                    cursor.execute(insert_query, (
                        visitor_id,
                        base64_image,
                        first_name,
                        last_name,
                        full_name,
                        email,
                        phone,
                        image_url
                    ))
                    copied += 1
                except Exception as e:
                    print(f"Error copying row: {e}")
                    continue
            
            local_conn.commit()
            print(f"✅ Successfully copied {copied} visitors from CSV")
            return True
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        local_conn.rollback()
        return False
    finally:
        cursor.close()
        local_conn.close()

def copy_from_json(json_file: str):
    """
    Copy visitors from JSON file to local database.
    
    JSON format: [{"visitor_id": "...", "base64Image": "...", ...}, ...]
    """
    print(f"Reading JSON file: {json_file}")
    local_conn = connect_to_database(LOCAL_DB_CONFIG)
    if not local_conn:
        return False
    
    try:
        cursor = local_conn.cursor()
        
        with open(json_file, 'r', encoding='utf-8') as f:
            visitors = json.load(f)
        
        insert_query = """
            INSERT INTO visitors (id, "base64Image", "firstName", "lastName", "fullName", email, phone, "imageUrl")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                "base64Image" = EXCLUDED."base64Image",
                "firstName" = EXCLUDED."firstName",
                "lastName" = EXCLUDED."lastName",
                "fullName" = EXCLUDED."fullName",
                email = EXCLUDED.email,
                phone = EXCLUDED.phone,
                "imageUrl" = EXCLUDED."imageUrl",
                "updatedAt" = CURRENT_TIMESTAMP
        """
        
        copied = 0
        for visitor in visitors:
            try:
                # Handle both old and new JSON formats
                visitor_id = visitor.get('id') or visitor.get('visitor_id')
                base64_image = visitor.get('base64Image') or visitor.get('base64Image')
                first_name = visitor.get('firstName')
                last_name = visitor.get('lastName')
                full_name = visitor.get('fullName') or visitor.get('name')
                
                # If name exists but firstName/lastName don't, split it
                if not first_name and full_name:
                    name_parts = full_name.split(' ', 1)
                    first_name = name_parts[0] if len(name_parts) > 0 else None
                    last_name = name_parts[1] if len(name_parts) > 1 else None
                
                cursor.execute(insert_query, (
                    visitor_id,
                    base64_image,
                    first_name,
                    last_name,
                    full_name,
                    visitor.get('email'),
                    visitor.get('phone'),
                    visitor.get('imageUrl')
                ))
                copied += 1
            except Exception as e:
                print(f"Error copying visitor {visitor.get('id') or visitor.get('visitor_id', 'unknown')}: {e}")
                continue
        
        local_conn.commit()
        print(f"✅ Successfully copied {copied} visitors from JSON")
        return True
        
    except Exception as e:
        print(f"Error reading JSON: {e}")
        local_conn.rollback()
        return False
    finally:
        cursor.close()
        local_conn.close()

def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python copy_data.py postgresql --source-host HOST --source-user USER --source-password PASSWORD --source-db DATABASE [--limit N]")
        print("  python copy_data.py csv <csv_file>")
        print("  python copy_data.py json <json_file>")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    
    if method == 'postgresql':
        # Parse command-line arguments
        source_config = {}
        limit = None
        
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--source-host':
                source_config['host'] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--source-user':
                source_config['user'] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--source-password':
                source_config['password'] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--source-db':
                source_config['database'] = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--source-port':
                source_config['port'] = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--limit':
                limit = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        # Set defaults
        source_config.setdefault('port', 5432)
        
        if not all(k in source_config for k in ['host', 'user', 'password', 'database']):
            print("Error: Missing required parameters for PostgreSQL source")
            print("Required: --source-host, --source-user, --source-password, --source-db")
            sys.exit(1)
        
        copy_from_postgresql(source_config, limit)
        
    elif method == 'csv':
        if len(sys.argv) < 3:
            print("Error: CSV file path required")
            sys.exit(1)
        copy_from_csv(sys.argv[2])
        
    elif method == 'json':
        if len(sys.argv) < 3:
            print("Error: JSON file path required")
            sys.exit(1)
        copy_from_json(sys.argv[2])
        
    else:
        print(f"Error: Unknown method '{method}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
