# Docker Installation Guide

This guide will help you install Docker Desktop on Windows to run the PostgreSQL database locally.

## 📥 Installing Docker Desktop for Windows

### Option 1: Direct Download (Recommended)

1. **Download Docker Desktop**:
   - Visit: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - The installer will be downloaded (usually `Docker Desktop Installer.exe`)

2. **Run the Installer**:
   - Double-click the downloaded installer
   - Follow the installation wizard
   - Make sure "Use WSL 2 instead of Hyper-V" is checked (recommended for Windows 10/11)
   - Click "Ok" when prompted to restart

3. **Restart Your Computer** (if prompted)

4. **Start Docker Desktop**:
   - Launch Docker Desktop from the Start menu
   - Wait for Docker to start (you'll see a whale icon in the system tray)
   - Docker Desktop may ask you to accept the terms of service

5. **Verify Installation**:
   ```powershell
   docker --version
   docker compose version
   ```
   You should see version numbers for both commands.

### Option 2: Using Winget (Windows Package Manager)

If you have `winget` installed:

```powershell
winget install Docker.DockerDesktop
```

### System Requirements

- **Windows 10 64-bit**: Pro, Enterprise, or Education (Build 19041 or higher)
- **Windows 11 64-bit**: Home or Pro version 21H2 or higher
- **WSL 2** feature enabled (Docker Desktop will help you enable this)
- **Virtualization** enabled in BIOS
- **4GB RAM** minimum (8GB recommended)
- **64-bit processor** with Second Level Address Translation (SLAT)

## 🔧 Post-Installation Setup

### 1. Enable WSL 2 (if not already enabled)

Docker Desktop will usually prompt you to enable WSL 2. If not, you can enable it manually:

```powershell
# Run PowerShell as Administrator
wsl --install
```

Or enable manually:
```powershell
# Run PowerShell as Administrator
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Then restart your computer and set WSL 2 as default:
```powershell
wsl --set-default-version 2
```

### 2. Verify Docker is Running

Open PowerShell and run:

```powershell
docker info
```

If Docker is running, you'll see system information. If not, start Docker Desktop from the Start menu.

### 3. Test Docker

Run a test container:

```powershell
docker run hello-world
```

You should see a success message.

## 🚀 After Installation

Once Docker Desktop is installed and running, you can:

1. **Start the database**:
   ```powershell
   cd database
   .\start-db.bat
   ```

   Or manually:
   ```powershell
   docker compose -f docker-compose.db.yml up -d
   ```

2. **Verify the database is running**:
   ```powershell
   docker ps
   ```

   You should see a container named `facial_recog_postgres` running.

## 🐛 Troubleshooting

### Docker Desktop won't start

1. **Check WSL 2 is enabled**:
   ```powershell
   wsl --status
   ```
   Should show "Default Version: 2"

2. **Check virtualization is enabled**:
   - Open Task Manager (Ctrl+Shift+Esc)
   - Go to "Performance" tab
   - Check "Virtualization" - should say "Enabled"

3. **Restart Docker Desktop**:
   - Right-click the Docker icon in system tray
   - Select "Restart Docker Desktop"

### "docker compose" command not found

If you see this error, you might have an older version of Docker. Try:

```powershell
# Old syntax (Docker Compose V1)
docker-compose -f docker-compose.db.yml up -d

# Or update Docker Desktop to the latest version
```

### Port 5432 already in use

If PostgreSQL is already running on your machine:

1. **Stop the existing PostgreSQL service**:
   ```powershell
   # Find the service
   Get-Service | Where-Object {$_.Name -like "*postgres*"}
   
   # Stop it (replace with actual service name)
   Stop-Service postgresql-x64-XX
   ```

2. **Or change the port** in `docker-compose.db.yml`:
   ```yaml
   ports:
     - "5433:5432"  # Use port 5433 instead
   ```

## 📚 Additional Resources

- **Docker Desktop Documentation**: https://docs.docker.com/desktop/
- **WSL 2 Installation Guide**: https://docs.microsoft.com/en-us/windows/wsl/install
- **Docker Compose Documentation**: https://docs.docker.com/compose/

## ⚠️ Alternative: Install PostgreSQL Directly (Without Docker)

If you prefer not to use Docker, you can install PostgreSQL directly on Windows:

1. **Download PostgreSQL**:
   - Visit: https://www.postgresql.org/download/windows/
   - Download the installer from EnterpriseDB

2. **Install PostgreSQL**:
   - Run the installer
   - Set password to `postgres` (or remember your password)
   - Use port `5432` (default)
   - Complete the installation

3. **Create the database**:
   ```sql
   CREATE DATABASE visitors_db;
   ```

4. **Run the init script**:
   ```powershell
   psql -U postgres -d visitors_db -f database/init.sql
   ```

5. **Update your `.env` file**:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=visitors_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   ```

However, using Docker is recommended as it's easier to manage and matches the production setup.
