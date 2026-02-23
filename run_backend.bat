@echo off
echo ==========================================
echo   TradePulse Backend Server
echo ==========================================
echo.

cd /d "%~dp0backend"

echo [1/2] Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo [2/2] Starting FastAPI server on http://localhost:8000
echo        Swagger docs: http://localhost:8000/docs
echo        Health check: http://localhost:8000/api/health
echo.
echo Press Ctrl+C to stop the server.
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
