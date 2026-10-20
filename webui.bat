@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1
cls

:: Perfil: general (por defecto) o kambo (lo fija webui-kambo.bat antes de llamar a este archivo)
if not defined STUDIO_PROFILE set "STUDIO_PROFILE=general"

echo ==============================================================================
if /i "%STUDIO_PROFILE%"=="kambo" (
    echo       KABOOMI 3D STUDIO - EDICION KABOOMI PARTY ^(GODOT 4^)
) else (
    echo       3D AI STUDIO - HUNYUAN3D-2 TURBO / STABLE FAST 3D
)
echo       Rutas de modelos y cache: .env ^(HY3DGEN_MODELS^), elegidas en setup.bat
echo ==============================================================================
echo.

where nvidia-smi.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [GPU NVIDIA DETECTADA]:
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    echo.
)

:: La GPU para la IA se elige al arrancar (gpu_select.py muestra un menu si hay 2+ tarjetas).
:: Para fijarla sin menu: STUDIO_GPU=0 en .env  (indice de nvidia-smi, o cpu)  o  webui.bat --gpu 1
:: Las rutas (HF_HOME, TEMP, ...) las define studio_env.py a partir del .env.

set "VENV_PY="
if exist "%~dp0.venv\Scripts\python.exe" set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not defined VENV_PY if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe"
if not defined VENV_PY if exist "C:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_PY=C:\AI_Models\venv_3dgen\Scripts\python.exe"

if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual ^(.venv^). Ejecuta setup.bat primero.
    pause
    exit /b 1
)

:: Preferencia de Alto Rendimiento de Windows para Python (equipos con GPU integrada + NVIDIA)
reg add "HKEY_CURRENT_USER\Software\Microsoft\DirectX\UserGpuPreferences" /v "%VENV_PY%" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1

:: La validacion en vivo de la GPU (pulso) se hace en webui.py despues de elegir la tarjeta
"%VENV_PY%" webui.py --mode %STUDIO_PROFILE% %*
pause
