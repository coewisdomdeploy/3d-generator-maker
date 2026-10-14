@echo off
setlocal
chcp 65001 >nul

echo ==============================================================================
echo       KABOOMI 3D GENERATOR (HUNYUAN3D-2) - INTERFAZ WEB
echo       Almacenamiento de Modelos y Cache: 100%% en Disco D:\
echo ==============================================================================

:: Variables de Entorno en Disco D:
set "HF_HOME=D:\AI_Models\huggingface"
set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub"
set "TORCH_HOME=D:\AI_Models\torch"
set "U2NET_HOME=D:\AI_Models\u2net"
set "PIP_CACHE_DIR=D:\AI_Models\pip_cache"
set "TEMP=D:\AI_Models\temp"
set "TMP=D:\AI_Models\temp"
set "HY3DGEN_MODELS=D:\AI_Models"

:: Activar entorno virtual
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "D:\AI_Models\venv_3dgen\Scripts\activate.bat" (
    call D:\AI_Models\venv_3dgen\Scripts\activate.bat
) else (
    echo [ADVERTENCIA] No se encontró el entorno virtual. Ejecuta setup.bat primero.
    pause
    exit /b 1
)

python webui.py
pause
