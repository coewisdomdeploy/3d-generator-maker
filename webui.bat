@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1

echo ==============================================================================
echo       3D AI STUDIO - INTERFAZ WEB GENERAL
echo       Motor: Hunyuan3D-2 Turbo (NVIDIA GeForce RTX fp16)
echo       Almacenamiento: 100%% en Disco D:\
echo ==============================================================================

set HF_HOME=D:\AI_Models\huggingface
set HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub
set TORCH_HOME=D:\AI_Models\torch
set U2NET_HOME=D:\AI_Models\u2net
set PIP_CACHE_DIR=D:\AI_Models\pip_cache
set TEMP=D:\AI_Models\temp
set TMP=D:\AI_Models\temp
set TMPDIR=D:\AI_Models\temp
set HY3DGEN_MODELS=D:\AI_Models
set STUDIO_PROFILE=general

set VENV_PY=
if exist %~dp0.venv\Scripts\python.exe set VENV_PY=%~dp0.venv\Scripts\python.exe
if not defined VENV_PY if exist D:\AI_Models\venv_3dgen\Scripts\python.exe set VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe
if not defined VENV_PY if exist C:\AI_Models\venv_3dgen\Scripts\python.exe set VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual en .venv ni en D:\AI_Models\venv_3dgen.
    pause
    exit /b 1
)

%VENV_PY% webui.py --mode general
pause