# Quick fix for production database
# Run this script manually if auto-migration doesn't work

import os
import sys

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text, create_engine
from database import get_database_url

def manual_migration():
    """Manual database migration for production"""
    try:
        # Get database URL
        database_url = get_database_url()
        
        # Create engine
        engine = create_engine(database_url)
        
        print(f"Connecting to database: {database_url[:20]}...")
        
        with engine.connect() as conn:
            # Check if is_admin column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='users' AND column_name='is_admin';
            """)).fetchone()
            
            if result:
                print("✅ is_admin column already exists!")
                return True
            
            # Add is_admin column
            print("Adding is_admin column...")
            conn.execute(text("""
                ALTER TABLE users 
                ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
            """))
            
            # Update existing users to set proper admin flag
            print("Setting admin flag for admin users...")
            conn.execute(text("""
                UPDATE users 
                SET is_admin = TRUE 
                WHERE email IN ('admin@premnath.com', 'admin@prenmath.com');
            """))
            
            # Commit changes
            conn.commit()
            
            print("✅ Database migration completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Running manual database migration...")
    success = manual_migration()
    if success:
        print("🎉 Migration successful! You can now deploy the application.")
    else:
        print("💥 Migration failed! Check database connection.")