@echo off
setlocal enabledelayedexpansion
title LUCID-CT: Clinical Decision Support Launcher

:: Navigate to project directory
cd /d "%~dp0"

echo =====================================================================
echo   LUCID-CT: AI Clinical Decision Support for 3D Thoracic CT Scans
echo =====================================================================
echo.

:: Check for virtual environments and activate if present
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment [.venv]...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment [venv]...
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    echo [*] Activating virtual environment [env]...
    call env\Scripts\activate.bat
)

:: Verify Python is installed and accessible
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ and make sure it is added to your environment variables.
    echo.
    pause
    exit /b 1
)

echo [*] Python interpreter detected:
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo     %%i
echo.

:: Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [*] Streamlit not detected. Installing project dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
        pause
        exit /b 1
    )
)

:: Check if port 8501 is already occupied
netstat -ano | findstr ":8501 " >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [!] Warning: Port 8501 is already in use by another process.
    echo [*] If a previous LUCID-CT instance is running, please close its terminal first.
    echo.
)

echo [*] Starting LUCID-CT Clinical Decision Support Interface...
echo [*] Loading models and medical imaging pipelines [takes ~3-5 seconds]...
echo [*] Your default browser will open automatically at http://localhost:8501 once ready.
echo.
echo [*] IMPORTANT: Keep this terminal window OPEN while using the application.
echo [*] Press Ctrl+C in this terminal window to stop the server at any time.
echo =====================================================================
echo.

:: Launch the Streamlit application (headless=false ensures browser opens only after server is ready)
python -m streamlit run app.py --server.headless false

:: Keep window open if the app exits or encounters an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
)
echo.
pause
