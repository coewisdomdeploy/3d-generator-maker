@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1
cls

set VENV_PY=
if exist %~dp0.venv\Scripts\python.exe set VENV_PY=%~dp0.venv\Scripts\python.exe
if not defined VENV_PY if exist D:\AI_Models\venv_3dgen\Scripts\python.exe set VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe
if not defined VENV_PY if exist C:\AI_Models\venv_3dgen\Scripts\python.exe set VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual con Python.
    pause
    exit /b 1
)

set HY3DGEN_MODELS=D:/AI_Models
%VENV_PY% %~dp0test_gen_cuda.py

echo.
pause