@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ==============================================================================
echo       3D GENERATOR - SETUP (Tencent Hunyuan3D-2 para GPU)
echo       Optimizador y Exportador de Modelos .GLB para Kaboomi Party
echo ==============================================================================
echo.

:: 1. Deteccion dinamica de disco y almacenamiento (Sin rutas fijas/quemadas)
if defined AI_MODELS_DIR (
    set "MODELS_ROOT=%AI_MODELS_DIR%"
) else if exist "D:\" (
    set "MODELS_ROOT=D:\AI_Models"
) else (
    set "MODELS_ROOT=%~dp0AI_Models"
)

echo [INFO] Carpeta de modelos y cache: %MODELS_ROOT%
set "HF_HOME=%MODELS_ROOT%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%MODELS_ROOT%\huggingface\hub"
set "TORCH_HOME=%MODELS_ROOT%\torch"
set "U2NET_HOME=%MODELS_ROOT%\u2net"
set "PIP_CACHE_DIR=%MODELS_ROOT%\pip_cache"
set "TEMP=%MODELS_ROOT%\temp"
set "TMP=%MODELS_ROOT%\temp"

if not exist "%MODELS_ROOT%\huggingface" mkdir "%MODELS_ROOT%\huggingface" 2>nul
if not exist "%MODELS_ROOT%\torch" mkdir "%MODELS_ROOT%\torch" 2>nul
if not exist "%MODELS_ROOT%\u2net" mkdir "%MODELS_ROOT%\u2net" 2>nul
if not exist "%MODELS_ROOT%\pip_cache" mkdir "%MODELS_ROOT%\pip_cache" 2>nul
if not exist "%MODELS_ROOT%\temp" mkdir "%MODELS_ROOT%\temp" 2>nul

:: 2. Deteccion Inteligente de Python (Priorizando el lanzador py y respetando Inkscape)
echo [1/7] Buscando interprete de Python en el sistema...
set "PY_CMD="

:: A) Variable de entorno definida por el usuario
if defined PYTHON (
    "%PYTHON%" -c "import sys; exit(0)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=%PYTHON%"
        echo [INFO] Detectado mediante variable de entorno PYTHON
        goto :python_found
    )
)
if defined PYTHON_CMD (
    "%PYTHON_CMD%" -c "import sys; exit(0)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=%PYTHON_CMD%"
        echo [INFO] Detectado mediante variable PYTHON_CMD
        goto :python_found
    )
)

:: B) PRIORIDAD PRINCIPAL: Lanzador oficial de Python (py.exe)
:: Extrae la ruta real del interprete oficial sin interferir con Inkscape
where py.exe >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined PY_CMD set "PY_CMD=%%P"
    )
    if not defined PY_CMD (
        for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do (
            if not defined PY_CMD set "PY_CMD=%%P"
        )
    )
    if not defined PY_CMD (
        for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
            if not defined PY_CMD set "PY_CMD=%%P"
        )
    )
    if not defined PY_CMD (
        for /f "delims=" %%P in ('py -c "import sys; print(sys.executable)" 2^>nul') do (
            if not defined PY_CMD set "PY_CMD=%%P"
        )
    )
    if defined PY_CMD (
        echo [INFO] Detectado mediante lanzador oficial py: "!PY_CMD!"
        goto :python_found
    )
)

:: C) Variable de entorno PATH (where python.exe, ignorando Inkscape y WindowsApps)
for /f "delims=" %%I in ('where python.exe 2^>nul') do (
    if not defined PY_CMD (
        echo %%I | findstr /i "WindowsApps Inkscape" >nul
        if !errorlevel! neq 0 (
            "%%I" -c "import sys; exit(0)" >nul 2>&1
            if !errorlevel! equ 0 (
                set "PY_CMD=%%I"
                echo [INFO] Detectado en variable PATH: %%I
                goto :python_found
            )
        )
    )
)

:: D) Rutas estandar de instalacion en Windows (excluyendo Inkscape)
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python311\python.exe" set "PY_CMD=%ProgramFiles%\Python311\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python310\python.exe" set "PY_CMD=%ProgramFiles%\Python310\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python312\python.exe" set "PY_CMD=%ProgramFiles%\Python312\python.exe"
if not defined PY_CMD if exist "D:\Programas\Python311\python.exe" set "PY_CMD=D:\Programas\Python311\python.exe"

if defined PY_CMD (
    echo [INFO] Detectado en ruta: "!PY_CMD!"
    goto :python_found
)

:: E) Si no se encontro automaticamente, solicitar al usuario o usar winget
echo.
echo [ALERTA] No se encontro Python en PATH ni en las carpetas comunes.
echo.
set /p "USER_PY=Ingresa la ruta completa de tu python.exe o presiona ENTER para instalarlo con winget: "
if defined USER_PY (
    set "USER_PY=!USER_PY:"=!"
    if exist "!USER_PY!" (
        set "PY_CMD=!USER_PY!"
        goto :python_found
    )
)

echo [INFO] Instalando Python 3.11 con winget...
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
    goto :python_found
)
if exist "%ProgramFiles%\Python311\python.exe" (
    set "PY_CMD=%ProgramFiles%\Python311\python.exe"
    goto :python_found
)
for /f "delims=" %%I in ('where python.exe 2^>nul') do (
    if not defined PY_CMD (
        set "PY_CMD=%%I"
        goto :python_found
    )
)

echo [ERROR] No se pudo encontrar ni instalar Python.
echo Por favor instala Python 3.11 marcando la casilla Add Python to PATH.
pause
exit /b 1

:python_found
echo [INFO] Interprete seleccionado: "!PY_CMD!"
echo.

:: 3. Creacion y Activacion del Entorno Virtual (.venv)
echo [2/7] Configurando entorno virtual...

if exist "D:\" (
    set "VENV_DIR=%MODELS_ROOT%\venv_3dgen"
) else (
    set "VENV_DIR=%~dp0.venv"
)

:: Crear venv si no existe
if not exist "!VENV_DIR!\Scripts\activate.bat" (
    echo [INFO] Creando entorno virtual en "!VENV_DIR!"...
    "!PY_CMD!" -m venv "!VENV_DIR!"
    if not exist "!VENV_DIR!\Scripts\activate.bat" (
        echo [INFO] Intentando crear entorno virtual alternativo en .venv...
        set "VENV_DIR=%~dp0.venv"
        "!PY_CMD!" -m venv "!VENV_DIR!"
    )
)

:: Enlace o activacion directa
set "ACTIVATE_BAT="
if exist "%~dp0.venv\Scripts\activate.bat" (
    set "ACTIVATE_BAT=%~dp0.venv\Scripts\activate.bat"
) else if exist "!VENV_DIR!\Scripts\activate.bat" (
    if /i "!VENV_DIR!" neq "%~dp0.venv" (
        if not exist "%~dp0.venv" mklink /J "%~dp0.venv" "!VENV_DIR!" >nul 2>&1
    )
    if exist "%~dp0.venv\Scripts\activate.bat" (
        set "ACTIVATE_BAT=%~dp0.venv\Scripts\activate.bat"
    ) else (
        set "ACTIVATE_BAT=!VENV_DIR!\Scripts\activate.bat"
    )
)

if not defined ACTIVATE_BAT (
    echo.
    echo [ERROR CRITICO] No se pudo generar el entorno virtual en: !VENV_DIR!
    echo Causa: El interprete seleccionado no incluye el modulo venv o ensurepip.
    echo Si estas usando el Python embebido de Inkscape, necesitas instalar la version
    echo oficial de Python 3.11 desde python.org marcando Add to PATH.
    echo.
    pause
    exit /b 1
)

echo [INFO] Activando entorno: "!ACTIVATE_BAT!"
call "!ACTIVATE_BAT!"
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)
echo [INFO] Entorno virtual activado con exito.
echo.

:: 4. Actualizacion de pip, setuptools y wheel
echo [3/7] Actualizando pip, setuptools y wheel...
python -m pip install --upgrade pip setuptools wheel --cache-dir "%PIP_CACHE_DIR%"
echo.

:: 5. Instalacion de PyTorch con soporte CUDA 12.4
echo [4/7] Instalando PyTorch con aceleracion CUDA 12.4...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --cache-dir "%PIP_CACHE_DIR%"
echo.

:: 6. Instalacion de dependencias del proyecto (requirements.txt)
echo [5/7] Instalando librerias 3D y dependencias...
pip install -r requirements.txt --cache-dir "%PIP_CACHE_DIR%"

:: Instalar hy3dgen si es necesario
python -c "import hy3dgen" 2>nul
if !errorlevel! neq 0 (
    echo [INFO] Instalando paquete hy3dgen oficial de Tencent...
    pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git --cache-dir "%PIP_CACHE_DIR%"
)
echo.

:: 7. Extensiones opcionales C++/CUDA
echo [6/7] Verificando compilador C++/CUDA...
python -c "import custom_rasterizer" 2>nul
if !errorlevel! neq 0 (
    where cl.exe >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Compilador MSVC detectado.
    ) else (
        echo [INFO] Compilador MSVC cl.exe no detectado en PATH.
        echo        El generador utilizara el motor integrado sin problemas.
    )
)
echo.

:: 8. Verificacion de GPU y Creacion de Lanzadores Dinamicos
echo [7/7] Verificando estado de GPU (NVIDIA CUDA)...
python -c "import torch; print(f'>>> CUDA Disponible: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB' if torch.cuda.is_available() else '>>> CUDA no disponible')"

:: Generar o actualizar lanzador WebUI con rutas dinamicas
echo @echo off > run_webui.bat
echo set "HF_HOME=%HF_HOME%" >> run_webui.bat
echo set "HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%" >> run_webui.bat
echo set "TORCH_HOME=%TORCH_HOME%" >> run_webui.bat
echo set "U2NET_HOME=%U2NET_HOME%" >> run_webui.bat
echo set "PIP_CACHE_DIR=%PIP_CACHE_DIR%" >> run_webui.bat
echo set "TEMP=%TEMP%" >> run_webui.bat
echo set "TMP=%TMP%" >> run_webui.bat
echo call "!ACTIVATE_BAT!" >> run_webui.bat
echo python webui.py >> run_webui.bat
echo pause >> run_webui.bat

:: Generar o actualizar lanzador CLI con rutas dinamicas
echo @echo off > run_cli.bat
echo set "HF_HOME=%HF_HOME%" >> run_cli.bat
echo set "HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%" >> run_cli.bat
echo set "TORCH_HOME=%TORCH_HOME%" >> run_cli.bat
echo set "U2NET_HOME=%U2NET_HOME%" >> run_cli.bat
echo set "PIP_CACHE_DIR=%PIP_CACHE_DIR%" >> run_cli.bat
echo set "TEMP=%TEMP%" >> run_cli.bat
echo set "TMP=%TMP%" >> run_cli.bat
echo call "!ACTIVATE_BAT!" >> run_cli.bat
echo python img2glb.py %%* >> run_cli.bat

echo.
echo ==============================================================================
echo  ¡INSTALACION CONFIGURADA CORRECTAMENTE!
echo ==============================================================================
echo  Modelos y Cache: %MODELS_ROOT%
echo  Entorno Virtual: !VENV_DIR!
echo.
echo  Para abrir la interfaz web:
echo     run_webui.bat
echo.
echo  Para ejecutar por consola:
echo     run_cli.bat --input mi_foto.png --output personaje.glb --faces 4000
echo ==============================================================================
echo.
pause
