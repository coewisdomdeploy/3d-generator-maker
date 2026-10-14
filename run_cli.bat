@echo off
setlocal
chcp 65001 >nul

:: Variables de Entorno en Disco D:
set "HF_HOME=D:\AI_Models\huggingface"
set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub"
set "TORCH_HOME=D:\AI_Models\torch"
set "U2NET_HOME=D:\AI_Models\u2net"
set "PIP_CACHE_DIR=D:\AI_Models\pip_cache"
set "TEMP=D:\AI_Models\temp"
set "TMP=D:\AI_Models\temp"

:: Activar entorno virtual
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "D:\AI_Models\venv_3dgen\Scripts\activate.bat" (
    call D:\AI_Models\venv_3dgen\Scripts\activate.bat
) else (
    echo [ADVERTENCIA] No se encontró el entorno virtual. Ejecuta setup.bat primero.
    exit /b 1
)

python img2glb.py %*
