@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul 2>&1
cls

echo ==============================================================================
echo       3D AI STUDIO - MOTOR HUNYUAN3D-2 TURBO EN NVIDIA GEFORCE RTX
echo       Almacenamiento y Modelos: 100%% en Disco D:\
echo ==============================================================================
echo.

where nvidia-smi.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [GPU DEDICADA DETECTADA]:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo.
)

:: Forzar exclusividad CUDA a la GPU NVIDIA (GPU 0)
set CUDA_VISIBLE_DEVICES=0
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
set STUDIO_PROFILE=general

set VENV_PY=
if exist "%~dp0.venv\Scripts\python.exe" set VENV_PY=%~dp0.venv\Scripts\python.exe
if not defined VENV_PY if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe
if not defined VENV_PY if exist "C:\AI_Models\venv_3dgen\Scripts\python.exe" set VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual en .venv ni en D:\AI_Models\venv_3dgen.
    pause
    exit /b 1
)

:: Forzar a Windows a usar Alto Rendimiento (GPU NVIDIA Dedicada) para Python
reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "%VENV_PY%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" (
    reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "D:\AI_Models\venv_3dgen\Scripts\python.exe" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
)

echo [ASIGNACION DE HARDWARE]:
echo   - GPU 0 (NVIDIA GeForce RTX): 100%% del calculo de Inteligencia Artificial
echo   - GPU 1 (Intel Integrada): DESACTIVADA para procesos de IA
echo.

echo [VALIDACION INICIAL DE GPU EN VIVO]:
"%VENV_PY%" -c "import torch, time; d=torch.device('cuda:0'); print('  >> Comprobando GPU dedicada:', torch.cuda.get_device_name(0), '...'); a=torch.randn((5120,5120),device=d,dtype=torch.float16); b=torch.randn((5120,5120),device=d,dtype=torch.float16); t0=time.time(); [torch.matmul(a,b) for _ in range(120)]; torch.cuda.synchronize(); print('  [OK] GPU NVIDIA verificada al 100%% en CUDA:0')"
echo.

"%VENV_PY%" webui.py --mode general
pause