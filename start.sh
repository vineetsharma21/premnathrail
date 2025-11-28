#!/bin/bash

# Deployment Script for Render
# This script runs on each deployment to ensure database is properly set up

echo "🚀 Starting deployment process..."

# Install dependencies (already done by Render)
echo "📦 Dependencies installed via requirements.txt"

# Start the application with proper database handling
echo "🗄️ Starting FastAPI application with database auto-migration..."

# The application will auto-create tables and add missing columns
exec uvicorn main:app --host 0.0.0.0 --port $PORT