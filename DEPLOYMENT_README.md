# Premnath Engineering Calculator Web Application

## 🔧 Quick Fix for Production Database Error

### Problem:
```
sqlalchemy.exc.ProgrammingError: column users.is_admin does not exist
```

### Solutions:

#### Option 1: Auto-Migration (Recommended)
The application now includes auto-migration on startup. Just redeploy and it should fix itself.

#### Option 2: Manual Migration
If auto-migration fails, run this in your Render console:

```python
python fix_production_db.py
```

#### Option 3: Direct Database Command
Connect to your PostgreSQL database and run:

```sql
ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;
UPDATE users SET is_admin = TRUE WHERE email = 'admin@premnath.com';
```

## 🚀 Deployment Status

### ✅ Fixed Issues:
1. **Logo Display**: Homepage logo restored (`/logo.png`)
2. **Database Migration**: Auto-migration for missing `is_admin` column
3. **Error Handling**: Proper error messages instead of technical stack traces
4. **UI/UX**: Professional design with PREMNATH RAIL branding

### 📋 Features:
- **Authentication System**: Login/Signup with bcrypt security
- **Admin Dashboard**: Role-based access control
- **Engineering Calculators**: 
  - Hydraulic Motor Calculations
  - Qmax Calculator
  - Load Distribution Analysis
  - Tractive Effort Calculations
  - Vehicle Performance Analysis
  - Braking Analysis
- **PDF Generation**: LaTeX-powered professional reports
- **License Management**: Company key activation system

### 🔐 Admin Access:
- Email: `admin@premnath.com` 
- The first user with this email automatically gets admin privileges
- Admin can access `/admin` dashboard after migration

### 🌐 Production URL:
Your app will be available at your Render URL once deployed.

## 🛠️ Local Development

```bash
# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8080
```

## 📱 Testing:
1. Visit homepage - logo should display
2. Sign up with any email
3. Login and verify tools work
4. Test admin login with admin@premnath.com

---
**Status**: ✅ Ready for Production Deployment