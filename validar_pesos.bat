@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
cls
echo ==============================================================================
echo       VALIDADOR DE PESOS Y ENCODER VISUAL (SingleImageEncoder)
echo ==============================================================================
echo.

set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PY_EXE if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "PY_EXE=D:\AI_Models\venv_3dgen\Scripts\python.exe"
if not defined PY_EXE set "PY_EXE=py -3.11"

"%PY_EXE%" "%~dp0validar_pesos.py"
echo.
pause
