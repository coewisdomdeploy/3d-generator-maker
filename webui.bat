@echo off 
setlocal 
cd /d "%~dp0" 
chcp 65001 >nul 
echo ============================================================================== 
echo       3D AI STUDIO - INTERFAZ WEB GENERAL 
echo       Motor Predeterminado: Hunyuan3D-2 Turbo • Modo Arcilla / Sin Textura 
echo       Almacenamiento de Modelos y Cache: 100%% en Disco d:\ 
echo ============================================================================== 
set "CUDA_VISIBLE_DEVICES=0" 
set "HF_HOME=d:\AI_Models\huggingface" 
set "HUGGINGFACE_HUB_CACHE=d:\AI_Models\huggingface\hub" 
set "TORCH_HOME=d:\AI_Models\torch" 
set "U2NET_HOME=d:\AI_Models\u2net" 
set "PIP_CACHE_DIR=d:\AI_Models\pip_cache" 
set "TEMP=d:\AI_Models\temp" 
set "TMP=d:\AI_Models\temp" 
set "TMPDIR=d:\AI_Models\temp" 
set "HY3DGEN_MODELS=d:\AI_Models" 
set "STUDIO_PROFILE=general" 
if exist "%~dp0.venv\Scripts\activate.bat" ( 
    call "%~dp0.venv\Scripts\activate.bat" 
) else if exist "d:\AI_Models\venv_3dgen\Scripts\activate.bat" ( 
    call "d:\AI_Models\venv_3dgen\Scripts\activate.bat" 
) else ( 
    echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. 
    pause 
    exit /b 1 
) 
python webui.py --mode general 
pause 
