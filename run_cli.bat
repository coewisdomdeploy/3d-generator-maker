@echo off 
setlocal 
chcp 65001 >nul 
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
if exist "%~dp0.venv\Scripts\activate.bat" ( 
    call "%~dp0.venv\Scripts\activate.bat" 
) else if exist "d:\AI_Models\venv_3dgen\Scripts\activate.bat" ( 
    call "d:\AI_Models\venv_3dgen\Scripts\activate.bat" 
) else ( 
    echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. 
    exit /b 1 
) 
python img2glb.py %* 
