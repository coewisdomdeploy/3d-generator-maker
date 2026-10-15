@echo off 
setlocal 
cd /d "%~dp0" 
chcp 65001 >nul 
echo ============================================================================== 
echo       KABOOMI 3D STUDIO - EDICION KABOOMI PARTY (GODOT 4) 
echo       Motor Predeterminado: Stable Fast 3D (SF3D) • Con Textura PBR Completa 
echo       Almacenamiento de Modelos y Cache: 100%% en Disco D:\ 
echo ============================================================================== 
set "HF_HOME=D:\AI_Models\huggingface" 
set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub" 
set "TORCH_HOME=D:\AI_Models\torch" 
set "U2NET_HOME=D:\AI_Models\u2net" 
set "PIP_CACHE_DIR=D:\AI_Models\pip_cache" 
set "TEMP=D:\AI_Models\temp" 
set "TMP=D:\AI_Models\temp" 
set "HY3DGEN_MODELS=D:\AI_Models" 
set "STUDIO_PROFILE=kambo" 
if exist "%~dp0.venv\Scripts\activate.bat" ( 
    call "%~dp0.venv\Scripts\activate.bat" 
) else if exist "D:\AI_Models\venv_3dgen\Scripts\activate.bat" ( 
    call "D:\AI_Models\venv_3dgen\Scripts\activate.bat" 
) else ( 
    echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. 
    pause 
    exit /b 1 
) 
python webui.py --mode kambo 
pause 
