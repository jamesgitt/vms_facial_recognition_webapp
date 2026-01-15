#!/bin/bash
# Quick start script for PostgreSQL database (Linux/Mac)

echo "========================================"
echo " Starting PostgreSQL Database"
echo "========================================"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

echo "[INFO] Starting PostgreSQL database..."
echo

# Start database
docker-compose -f docker-compose.db.yml up -d

if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Failed to start database!"
    exit 1
fi

echo
echo "[SUCCESS] Database started!"
echo
echo "Connection details:"
echo "  Host:     localhost"
echo "  Port:     5432"
echo "  Database: visitors_db"
echo "  User:     postgres"
echo "  Password: postgres"
echo
echo "  Connection String:"
echo "  postgresql://postgres:postgres@localhost:5432/visitors_db"
echo
echo "  pgAdmin: http://localhost:5050"
echo "  Email:   admin@admin.com"
echo "  Password: admin"
echo
echo "To stop: docker-compose -f docker-compose.db.yml down"
echo
