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
    echo [*] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    echo [*] Activating virtual environment (env)...
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

echo [*] Starting LUCID-CT Clinical Decision Support Interface...
echo [*] Local URL: http://localhost:8501
echo.
echo Press Ctrl+C in this terminal window to stop the server at any time.
echo =====================================================================
echo.

:: Open the default browser to the Streamlit local URL
start "" http://localhost:8501

:: Launch the Streamlit application
python -m streamlit run app.py

:: Keep window open if the app exits or encounters an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
)
echo.
pause
