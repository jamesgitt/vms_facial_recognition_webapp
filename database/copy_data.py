"""
Script to copy visitor data from a JSON file to the local PostgreSQL database.
"""

import os
import sys
import psycopg2
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

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

# Local database connection from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "visitors_db")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

# Build config from DATABASE_URL or individual parameters
if DATABASE_URL:
    LOCAL_DB_CONFIG = DATABASE_URL
else:
    LOCAL_DB_CONFIG = {
        'host': DB_HOST,
        'port': DB_PORT,
        'database': DB_NAME,
        'user': DB_USER,
        'password': DB_PASSWORD
    }

def connect_to_database(config):
    """Connect to PostgreSQL database."""
    try:
        if isinstance(config, str):
            # DATABASE_URL connection string
            conn = psycopg2.connect(config)
        else:
            # Dictionary with individual parameters
            conn = psycopg2.connect(**config)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def copy_from_json(json_file: str, dry_run: bool = False):
    """
    Copy visitors from JSON file to local database.

    JSON format: [{"id": "...", "base64Image": "...", ...}, ...]
    
    Args:
        json_file: Path to JSON file containing visitor data
        dry_run: If True, only validate data without inserting
    """
    if not os.path.exists(json_file):
        print(f"❌ Error: JSON file not found: {json_file}")
        return False
    
    print(f"📄 Reading JSON file: {json_file}")
    local_conn = connect_to_database(LOCAL_DB_CONFIG)
    if not local_conn:
        return False

    try:
        cursor = local_conn.cursor()

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        visitors = None
        if isinstance(data, list):
            # Direct array: [{...}, {...}]
            visitors = data
        elif isinstance(data, dict):
            # Object with array inside: {"visitors": [...], "data": [...], etc.}
            # Try common keys
            for key in ['visitors', 'data', 'visitor_data', 'results', 'items', 'records']:
                if key in data and isinstance(data[key], list):
                    visitors = data[key]
                    print(f"📋 Found array under key: '{key}'")
                    break
            
            # If still not found, check if any value is a list
            if visitors is None:
                for key, value in data.items():
                    if isinstance(value, list):
                        visitors = value
                        print(f"📋 Found array under key: '{key}'")
                        break
        
        if visitors is None or not isinstance(visitors, list):
            print(f"❌ Error: JSON file must contain an array of visitors")
            print(f"")
            print(f"Expected format:")
            print(f"  Option 1: Direct array")
            print(f"    [{{\"id\": \"...\", \"base64Image\": \"...\"}}, ...]")
            print(f"")
            print(f"  Option 2: Object with array")
            print(f"    {{\"visitors\": [{{\"id\": \"...\", \"base64Image\": \"...\"}}, ...]}}")
            print(f"")
            if isinstance(data, dict):
                print(f"Found object with keys: {list(data.keys())[:10]}")
                # Show structure hint
                for key, value in list(data.items())[:3]:
                    value_type = type(value).__name__
                    if isinstance(value, list):
                        print(f"  - '{key}': array with {len(value)} items")
                    else:
                        print(f"  - '{key}': {value_type}")
            else:
                print(f"Found: {type(data).__name__}")
            return False

        print(f"📊 Found {len(visitors)} visitors in JSON file")
        
        if dry_run:
            print("🔍 DRY RUN MODE - Validating data only (no inserts)")
        
        insert_query = """
            INSERT INTO visitors (id, "base64Image", "firstName", "lastName", "fullName", email, phone, "imageUrl", "createdAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        skipped = 0
        errors = 0
        
        for idx, visitor in enumerate(visitors, 1):
            try:
                # Handle both old and new JSON formats
                visitor_id = visitor.get('id') or visitor.get('visitor_id')
                if not visitor_id:
                    print(f"⚠️  Visitor #{idx}: Missing ID, skipping")
                    skipped += 1
                    continue
                
                base64_image = visitor.get('base64Image') or visitor.get('base64_image')
                if not base64_image:
                    print(f"⚠️  Visitor #{idx} ({visitor_id}): Missing base64Image, skipping")
                    skipped += 1
                    continue
                
                first_name = visitor.get('firstName') or visitor.get('first_name')
                last_name = visitor.get('lastName') or visitor.get('last_name')
                full_name = visitor.get('fullName') or visitor.get('full_name') or visitor.get('name')

                # If name exists but firstName/lastName don't, split it
                if not first_name and full_name:
                    name_parts = full_name.split(' ', 1)
                    first_name = name_parts[0] if len(name_parts) > 0 else None
                    last_name = name_parts[1] if len(name_parts) > 1 else None
                
                # Handle createdAt if present in JSON
                created_at = visitor.get('createdAt') or visitor.get('created_at')
                if isinstance(created_at, str):
                    # Try to parse ISO format or other common formats
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = None

                if not dry_run:
                    cursor.execute(insert_query, (
                        visitor_id,
                        base64_image,
                        first_name,
                        last_name,
                        full_name,
                        visitor.get('email'),
                        visitor.get('phone'),
                        visitor.get('imageUrl') or visitor.get('image_url'),
                        created_at
                    ))
                
                copied += 1
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(visitors)} visitors processed...")
                    
            except Exception as e:
                visitor_id = visitor.get('id') or visitor.get('visitor_id', f'#{idx}')
                print(f"❌ Error copying visitor {visitor_id}: {e}")
                errors += 1
                continue

        if not dry_run:
            local_conn.commit()
            print(f"\n✅ Successfully copied {copied} visitors from JSON")
            if skipped > 0:
                print(f"⚠️  Skipped {skipped} visitors (missing required fields)")
            if errors > 0:
                print(f"❌ Failed to copy {errors} visitors (errors occurred)")
        else:
            print(f"\n✅ Validation complete: {copied} valid visitors found")
            if skipped > 0:
                print(f"⚠️  {skipped} visitors would be skipped (missing required fields)")
            if errors > 0:
                print(f"❌ {errors} visitors have errors")
        
        return True

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        if not dry_run:
            local_conn.rollback()
        return False
    finally:
        cursor.close()
        local_conn.close()

def find_json_on_desktop(filename: Optional[str] = None) -> Optional[Path]:
    """
    Find JSON file on user's desktop.
    
    Args:
        filename: Specific filename to look for (e.g., 'visitors.json')
                  If None, looks for common visitor data filenames
    
    Returns:
        Path to JSON file if found, None otherwise
    """
    # Get desktop path
    desktop = Path.home() / "Desktop"
    
    if not desktop.exists():
        # Try alternative desktop locations
        alt_desktop = Path(os.path.expanduser("~/Desktop"))
        if alt_desktop.exists():
            desktop = alt_desktop
        else:
            return None
    
    # If specific filename provided, look for it
    if filename:
        json_path = desktop / filename
        if json_path.exists():
            return json_path
        # Try with .json extension if not provided
        if not filename.endswith('.json'):
            json_path = desktop / f"{filename}.json"
            if json_path.exists():
                return json_path
        return None
    
    # Look for common visitor data filenames
    common_names = [
        "visitors.json",
        "visitor_data.json",
        "visitors_data.json",
        "data.json",
        "visitors.json",
    ]
    
    for name in common_names:
        json_path = desktop / name
        if json_path.exists():
            return json_path
    
    return None

def main():
    """Main function to handle command-line arguments for JSON copy."""
    dry_run = '--dry-run' in sys.argv or '-d' in sys.argv
    
    # If no arguments provided, try to find JSON on desktop
    if len(sys.argv) < 2:
        print("🔍 No file specified. Searching Desktop for visitor data...")
        json_file = find_json_on_desktop()
        
        if json_file:
            print(f"✅ Found: {json_file}")
            print("")
            # json_file is already a Path object
            json_path = json_file
        else:
            print("❌ No JSON file found on Desktop.")
            print("")
            print("Usage:")
            print("  python copy_data.py <json_file> [--dry-run]")
            print("  python copy_data.py  # Auto-find JSON on Desktop")
            print("")
            print("Examples:")
            print("  python copy_data.py")
            print("  python copy_data.py visitors.json")
            print("  python copy_data.py C:\\Users\\YourName\\Desktop\\visitors.json")
            print("  python copy_data.py visitors.json --dry-run  # Validate without inserting")
            print("")
            print("💡 Tip: Place your JSON file on Desktop with one of these names:")
            print("   - visitors.json")
            print("   - visitor_data.json")
            print("   - visitors_data.json")
            print("   - data.json")
            sys.exit(1)
    else:
        json_file = sys.argv[1]
        
        # Convert to Path for easier handling
        json_path = Path(json_file)
        
        # Expand user path if needed (check string representation)
        json_file_str = str(json_file)
        if json_file_str.startswith("~"):
            json_path = Path(os.path.expanduser(json_file_str))
        
        # If it's a relative path or just filename, check desktop first
        if not json_path.is_absolute() and not json_path.exists():
            desktop_file = find_json_on_desktop(json_path.name)
            if desktop_file:
                print(f"📁 Found file on Desktop: {desktop_file}")
                json_path = desktop_file
    
    # Check if file exists
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print("")
        # Suggest desktop location
        desktop_file = find_json_on_desktop(json_path.name)
        if desktop_file:
            print(f"💡 Did you mean: {desktop_file}?")
        sys.exit(1)
    
    print(f"📄 Using file: {json_path}")
    print(f"📂 Full path: {json_path.absolute()}")
    print("")
    
    success = copy_from_json(str(json_path), dry_run=dry_run)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
