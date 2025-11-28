# Database Migration Script
# Add is_admin column to production database

import os
import psycopg2
from urllib.parse import urlparse

def migrate_database():
    # Get DATABASE_URL from environment (Render automatically sets this)
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        print("DATABASE_URL not found in environment variables")
        return False
    
    try:
        # Parse the database URL
        url = urlparse(database_url)
        
        # Connect to database
        conn = psycopg2.connect(
            host=url.hostname,
            database=url.path[1:],  # Remove leading slash
            user=url.username,
            password=url.password,
            port=url.port
        )
        
        cursor = conn.cursor()
        
        # Check if is_admin column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='is_admin';
        """)
        
        result = cursor.fetchone()
        
        if result:
            print("is_admin column already exists!")
            return True
        
        # Add is_admin column
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
        """)
        
        # Create admin user if it doesn't exist
        cursor.execute("""
            INSERT INTO users (email, hashed_password, is_license_active, is_admin)
            SELECT 'admin@premnath.com', '$2b$12$example_hashed_password', TRUE, TRUE
            WHERE NOT EXISTS (
                SELECT 1 FROM users WHERE email = 'admin@premnath.com'
            );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database migration completed successfully!")
        print("✅ is_admin column added to users table")
        print("✅ Admin user created (if not existed)")
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_database()