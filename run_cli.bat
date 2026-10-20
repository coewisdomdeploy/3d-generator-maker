@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

set NVIDIA_VISIBLE_DEVICES=all
set HF_HOME=D:\AI_Models\huggingface
set HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub
set TORCH_HOME=D:\AI_Models\torch
set U2NET_HOME=D:\AI_Models\u2net
set PIP_CACHE_DIR=D:\AI_Models\pip_cache
set TEMP=D:\AI_Models\temp
set TMP=D:\AI_Models\temp
set TMPDIR=D:\AI_Models\temp
set HY3DGEN_MODELS=D:\AI_Models

set VENV_PY=
if exist "%~dp0.venv\Scripts\python.exe" set VENV_PY=%~dp0.venv\Scripts\python.exe
if not defined VENV_PY if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe
if not defined VENV_PY if exist "C:\AI_Models\venv_3dgen\Scripts\python.exe" set VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual en .venv ni en D:\AI_Models\venv_3dgen.
    pause
    exit /b 1
)

reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "%VENV_PY%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" (
    reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "D:\AI_Models\venv_3dgen\Scripts\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
)

"%VENV_PY%" img2glb.py %*
pause