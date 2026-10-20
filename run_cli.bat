@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

:: Las rutas (HF_HOME, TEMP, ...) las define studio_env.py a partir del .env.
:: GPU: menu al arrancar si hay 2+ tarjetas, o --gpu 0 / --gpu cpu para fijarla.

set "VENV_PY="
if exist "%~dp0.venv\Scripts\python.exe" set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not defined VENV_PY if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe"
if not defined VENV_PY if exist "C:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe"

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual ^(.venv^). Ejecuta setup.bat primero.
    pause
    exit /b 1
)

reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "%VENV_PY%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1

"%VENV_PY%" img2glb.py %*
pause
