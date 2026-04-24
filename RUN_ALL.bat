@echo off
title Crypto Scanner - Setup and Launch
color 0A
cls

echo ============================================================
echo   CRYPTO SCANNER v2.0 - Full Launch Script
echo ============================================================
echo.

REM ── Step 1: Set working directory ────────────────────────────
echo [STEP 1/5] Setting working directory...
cd /d "D:\Coinglass Scanner\crypto-scanner"
if errorlevel 1 (
    echo ERROR: Could not navigate to project folder.
    pause
    exit /b 1
)
echo   OK: %CD%
echo.

REM ── Step 2: Check Python is available ────────────────────────
echo [STEP 2/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   OK: %%v
echo.

REM ── Step 3: Create required folders if missing ───────────────
echo [STEP 3/5] Ensuring data directories exist...
if not exist "data\raw"      mkdir "data\raw"
if not exist "data\features" mkdir "data\features"
if not exist "data\labels"   mkdir "data\labels"
if not exist "models"        mkdir "models"
if not exist "logs"          mkdir "logs"
echo   OK: data\raw, data\features, data\labels, models, logs
echo.

REM ── Step 4: Install / verify Python dependencies ─────────────
echo [STEP 4/5] Installing Python dependencies (requirements.txt)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install one or more dependencies.
    echo        Run manually:  pip install -r requirements.txt
    pause
    exit /b 1
)
echo   OK: All dependencies satisfied.
echo.

REM ── Step 5: Launch processes in separate windows ─────────────
echo [STEP 5/5] Launching Scanner and Dashboard...
echo.

echo   Starting SCANNER  (main process - runs every 5 min indefinitely)...
start "Crypto Scanner - MAIN" cmd /k "cd /d "D:\Coinglass Scanner\crypto-scanner" && color 0A && python scanner.py"

REM Brief pause so scanner starts first
timeout /t 3 /nobreak >nul

echo   Starting DASHBOARD (read-only analytics viewer)...
start "Crypto Scanner - DASHBOARD" cmd /k "cd /d "D:\Coinglass Scanner\crypto-scanner" && color 0B && python dashboard.py"

echo.
echo ============================================================
echo   Both windows are now running.
echo.
echo   SCANNER   - collects data, trains XGBoost, shows signals
echo   DASHBOARD - shows data summary and ML training history
echo.
echo   To STOP: close each window or press Ctrl+C inside it.
echo ============================================================
echo.
pause
