@echo off
setlocal
cd /d %~dp0

rem ---- pick a Python that actually has the dependencies ----
set "PY1=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
set "PY2=C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe"
set "PY="

if exist "%PY1%" set "PY=%PY1%"
if not defined PY if exist "%PY2%" set "PY=%PY2%"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"

if not defined PY (
  echo [ERROR] Python not found. Please install Python 3.10+ first.
  pause
  exit /b 1
)

echo Using Python: %PY%

rem ---- install dependencies on first run ----
"%PY%" -c "import fastapi" >nul 2>nul
if errorlevel 1 (
  echo First run detected: installing dependencies, please wait...
  "%PY%" -m pip install -q -r requirements.txt
)

set MOCK_MODE=1
echo.
echo ================================================
echo  Demo is starting at http://127.0.0.1:8080
echo  Close this window to stop the demo.
echo ================================================
echo.
start "" http://127.0.0.1:8080
"%PY%" -m uvicorn app.server:app --host 127.0.0.1 --port 8080
pause
