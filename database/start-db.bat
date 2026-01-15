@echo off
REM Quick start script for PostgreSQL database (Windows)

echo ========================================
echo  Starting PostgreSQL Database
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [INFO] Starting PostgreSQL database...
echo.

REM Start database
docker compose -f docker-compose.db.yml up -d

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start database!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Database started!
echo.
echo Connection details:
echo   Host:     localhost
echo   Port:     5432
echo   Database: visitors_db
echo   User:     postgres
echo   Password: postgres
echo.
echo   Connection String:
echo   postgresql://postgres:postgres@localhost:5432/visitors_db
echo.
echo   pgAdmin: http://localhost:5050
echo   Email:   admin@admin.com
echo   Password: admin
echo.
echo To stop: docker compose -f docker-compose.db.yml down
echo.

pause
