@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
cls
echo ==============================================================================
echo       DESCARGA DIRECTA DE MODELOS 3D (Tencent Hunyuan3D-2 Turbo)
echo ==============================================================================
echo  Destino: [carpeta de modelos del .env]\tencent\Hunyuan3D-2\hunyuan3d-dit-v2-0-turbo\
echo  Archivo: model.fp16.safetensors (4.93 GB)
echo  Soporta reanudacion automatica: No se pierde lo ya descargado.
echo ==============================================================================
echo.

set "PY_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PY_EXE if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "PY_EXE=D:\AI_Models\venv_3dgen\Scripts\python.exe"
if not defined PY_EXE set "PY_EXE=py -3.11"

echo Iniciando descarga con reporte en vivo de velocidad y porcentaje...
echo.
"%PY_EXE%" "%~dp0download_model.py"
echo.
echo Presione cualquier tecla para salir...
pause >nul
