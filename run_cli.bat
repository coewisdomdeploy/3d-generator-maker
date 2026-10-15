@echo off 
setlocal 
chcp 65001 >nul 
set "CUDA_VISIBLE_DEVICES=0" 
set "HF_HOME=D:\AI_Models\huggingface" 
set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub" 
set "TORCH_HOME=D:\AI_Models\torch" 
set "U2NET_HOME=D:\AI_Models\u2net" 
set "PIP_CACHE_DIR=D:\AI_Models\pip_cache" 
set "TEMP=D:\AI_Models\temp" 
set "TMP=D:\AI_Models\temp" 
set "HY3DGEN_MODELS=D:\AI_Models" 
if exist "%~dp0.venv\Scripts\activate.bat" ( 
    call "%~dp0.venv\Scripts\activate.bat" 
) else if exist "D:\AI_Models\venv_3dgen\Scripts\activate.bat" ( 
    call "D:\AI_Models\venv_3dgen\Scripts\activate.bat" 
) else ( 
    echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. 
    exit /b 1 
) 
python img2glb.py %* 
