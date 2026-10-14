@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ==============================================================================
echo       3D GENERATOR - SETUP (Tencent Hunyuan3D-2 para GPU 16GB VRAM)
echo       Optimizador y Exportador de Modelos .GLB para Kaboomi Party (Godot 4)
echo       MODO ALMACENAMIENTO SEGURO: 100%% EN DISCO D:\ (Modelos, Venv y Cache)
echo ==============================================================================
echo.

:: 1. Variables de entorno globales para almacenar 100% en D:
set "HF_HOME=D:\AI_Models\huggingface"
set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub"
set "TORCH_HOME=D:\AI_Models\torch"
set "U2NET_HOME=D:\AI_Models\u2net"
set "PIP_CACHE_DIR=D:\AI_Models\pip_cache"
set "TEMP=D:\AI_Models\temp"
set "TMP=D:\AI_Models\temp"

:: Asegurar carpetas base en D:\
if not exist "D:\AI_Models\huggingface" mkdir "D:\AI_Models\huggingface"
if not exist "D:\AI_Models\torch" mkdir "D:\AI_Models\torch"
if not exist "D:\AI_Models\u2net" mkdir "D:\AI_Models\u2net"
if not exist "D:\AI_Models\pip_cache" mkdir "D:\AI_Models\pip_cache"
if not exist "D:\AI_Models\venv_3dgen" mkdir "D:\AI_Models\venv_3dgen"
if not exist "D:\AI_Models\temp" mkdir "D:\AI_Models\temp"

:: 2. Detección de Python
echo [1/7] Buscando interprete de Python 3.11...
set "PY_CMD="

if exist "D:\Programas\Python311\python.exe" (
    set "PY_CMD=D:\Programas\Python311\python.exe"
    goto :python_found
)

where python >nul 2>&1
if !errorlevel! equ 0 (
    set "PY_CMD=python"
    goto :python_found
)

where py >nul 2>&1
if !errorlevel! equ 0 (
    set "PY_CMD=py -3.11"
    goto :python_found
)

echo [ALERTA] No se encontró Python 3.11 en D:\Programas\Python311 ni en PATH.
echo Instalando Python 3.11 directamente en D:\Programas\Python311...
winget install -e --id Python.Python.3.11 --override "/passive InstallAllUsers=0 TargetDir=D:\Programas\Python311 PrependPath=1 Include_launcher=0 Include_test=0" --accept-package-agreements --accept-source-agreements
if exist "D:\Programas\Python311\python.exe" (
    set "PY_CMD=D:\Programas\Python311\python.exe"
    echo Python 3.11 instalado con éxito en D:\Programas\Python311.
) else (
    echo [ERROR] Por favor instala Python 3.11 en D:\Programas\Python311.
    pause
    exit /b 1
)

:python_found
echo -> Python utilizado: !PY_CMD!
echo.

:: 3. Creación de Entorno Virtual en D:\ con Enlace Simbólico/Unión Local
echo [2/7] Configurando entorno virtual 100%% en D:\AI_Models\venv_3dgen...
if not exist "D:\AI_Models\venv_3dgen\Scripts\activate.bat" (
    echo -> Creando entorno virtual en D:\AI_Models\venv_3dgen...
    !PY_CMD! -m venv D:\AI_Models\venv_3dgen
)

:: Crear unión local .venv -> D:\AI_Models\venv_3dgen si no existe
if not exist ".venv\Scripts\activate.bat" (
    if exist ".venv" rmdir /q ".venv" 2>nul
    echo -> Creando unión de directorio .venv que apunta a D:\AI_Models\venv_3dgen...
    mklink /J .venv D:\AI_Models\venv_3dgen
)

call .venv\Scripts\activate.bat
echo -> Entorno virtual activado.
echo.

:: 4. Actualización de pip, setuptools y wheel usando cache en D:
echo [3/7] Actualizando pip, setuptools y wheel en D:\...
python -m pip install --upgrade pip setuptools wheel --cache-dir D:\AI_Models\pip_cache
echo.

:: 5. Instalación de PyTorch con soporte CUDA 12.4 en D:\
echo [4/7] Instalando PyTorch con aceleración CUDA 12.4 en D:\...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --cache-dir D:\AI_Models\pip_cache
echo.

:: 6. Instalación de dependencias del proyecto (requirements.txt)
echo [5/7] Instalando librerias 3D y dependencias en D:\...
pip install -r requirements.txt --cache-dir D:\AI_Models\pip_cache

:: Instalar hy3dgen si es necesario
python -c "import hy3dgen" 2>nul
if !errorlevel! neq 0 (
    echo -> Instalando paquete hy3dgen oficial de Tencent en D:\...
    pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git --cache-dir D:\AI_Models\pip_cache
)

:: 7. Extensiones opcionales C++/CUDA
echo [6/7] Verificando extensiones de textura C++/CUDA...
python -c "import custom_rasterizer" 2>nul
if !errorlevel! neq 0 (
    where cl.exe >nul 2>&1
    if !errorlevel! equ 0 (
        echo -> Compilador MSVC detectado.
    ) else (
        echo -> Nota: Compilador MSVC cl.exe no detectado en PATH.
        echo    El generador utilizará el motor de texturizado y horneado UV integrado sin problemas.
    )
)
echo.

:: 8. Verificación de GPU y Creación de Lanzadores Rápidos
echo [7/7] Verificando estado de GPU (NVIDIA CUDA)...
python -c "import torch; print(f'>>> CUDA Disponible: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB' if torch.cuda.is_available() else '>>> CUDA no disponible')"

:: Lanzador WebUI
if not exist "run_webui.bat" (
    echo @echo off > run_webui.bat
    echo set "HF_HOME=D:\AI_Models\huggingface" >> run_webui.bat
    echo set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub" >> run_webui.bat
    echo set "TORCH_HOME=D:\AI_Models\torch" >> run_webui.bat
    echo set "U2NET_HOME=D:\AI_Models\u2net" >> run_webui.bat
    echo set "PIP_CACHE_DIR=D:\AI_Models\pip_cache" >> run_webui.bat
    echo set "TEMP=D:\AI_Models\temp" >> run_webui.bat
    echo set "TMP=D:\AI_Models\temp" >> run_webui.bat
    echo call .venv\Scripts\activate.bat >> run_webui.bat
    echo python webui.py >> run_webui.bat
    echo pause >> run_webui.bat
)

:: Lanzador CLI
if not exist "run_cli.bat" (
    echo @echo off > run_cli.bat
    echo set "HF_HOME=D:\AI_Models\huggingface" >> run_cli.bat
    echo set "HUGGINGFACE_HUB_CACHE=D:\AI_Models\huggingface\hub" >> run_cli.bat
    echo set "TORCH_HOME=D:\AI_Models\torch" >> run_cli.bat
    echo set "U2NET_HOME=D:\AI_Models\u2net" >> run_cli.bat
    echo set "PIP_CACHE_DIR=D:\AI_Models\pip_cache" >> run_cli.bat
    echo set "TEMP=D:\AI_Models\temp" >> run_cli.bat
    echo set "TMP=D:\AI_Models\temp" >> run_cli.bat
    echo call .venv\Scripts\activate.bat >> run_cli.bat
    echo python img2glb.py %%* >> run_cli.bat
)

echo.
echo ==============================================================================
echo  ¡INSTALACIÓN CONFIGURADA 100%% EN DISCO D:!
echo ==============================================================================
echo  Modelos y Cache: D:\AI_Models\huggingface
echo  Entorno Virtual: D:\AI_Models\venv_3dgen (.venv)
echo.
echo  Para abrir la interfaz web:
echo     run_webui.bat
echo.
echo  Para ejecutar por consola:
echo     run_cli.bat --input mi_foto.png --output ../Games/kaboomi-party/assets/models/bombos/personaje.glb --faces 4000
echo ==============================================================================
echo.
pause
