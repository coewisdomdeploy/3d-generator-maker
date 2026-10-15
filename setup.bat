@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

:: ==============================================================================
:: CONFIGURACION DE COLORES ANSI PARA CONSOLA WINDOWS
:: ==============================================================================
for /f "delims=" %%a in ('powershell -NoProfile -Command "[char]27"') do set "ESC=%%a"
set "GREEN=!ESC![92m"
set "YELLOW=!ESC![93m"
set "RED=!ESC![91m"
set "CYAN=!ESC![96m"
set "BOLD=!ESC![1m"
set "RESET=!ESC![0m"

echo ==============================================================================
echo       3D AI STUDIO - INSTALADOR Y CONFIGURADOR DE MOTORES 3D
echo       Tencent Hunyuan3D-2 Turbo ^& Stability AI Stable Fast 3D (SF3D)
echo       Almacenamiento de Modelos y Cache: 100%% en Disco Seleccionado
echo ==============================================================================
echo.

:: ==============================================================================
:: 1. SELECCION Y VALIDACION DEL DIRECTORIO DE ALMACENAMIENTO
:: ==============================================================================
:select_storage
echo [CONFIGURACION DE ALMACENAMIENTO]
echo Los modelos de IA y la cache requieren entre 15 GB y 30 GB de espacio libre.
echo.

:: Definir ruta sugerida por defecto
set "SUGGESTED_DIR=%~dp0AI_Models"
if exist "%~d0\AI_Models" (
    set "SUGGESTED_DIR=%~d0\AI_Models"
) else if exist "D:\" (
    set "SUGGESTED_DIR=D:\AI_Models"
) else if exist "C:\" (
    set "SUGGESTED_DIR=C:\AI_Models"
)

echo Selecciona donde deseas almacenar los modelos, cache y entorno virtual:
echo.
echo   [1] Ruta sugerida:           !SUGGESTED_DIR!
echo   [2] Carpeta del proyecto:    %~dp0AI_Models
echo   [3] Raiz de unidad %~d0\       %~d0\AI_Models
if /i not "%~d0"=="C:" if exist "C:\" (
    echo   [4] En disco C:\             C:\AI_Models
)
if /i not "%~d0"=="D:" if exist "D:\" (
    echo   [5] En disco D:\             D:\AI_Models
)
if /i not "%~d0"=="E:" if exist "E:\" (
    echo   [6] En disco E:\             E:\AI_Models
)
echo   [P] Ruta personalizada       (Escribir ruta manual, ej: C:\iamodels)
echo.

set "STORAGE_CHOICE=1"
set /p "STORAGE_CHOICE=Elige una opcion o escribe la ruta directamente (Enter para opcion 1): "
echo.

set "STORAGE_CHOICE=!STORAGE_CHOICE: =!"

:: Si el usuario escribio directamente una ruta (contiene ':' o '\')
echo !STORAGE_CHOICE! | findstr ": \\" >nul
if !errorlevel! equ 0 (
    set "MODELS_ROOT=!STORAGE_CHOICE!"
    goto :storage_ready
)

if /i "!STORAGE_CHOICE:~0,1!"=="P" goto :prompt_custom_storage
if "!STORAGE_CHOICE:~0,1!"=="2" (
    set "MODELS_ROOT=%~dp0AI_Models"
    goto :storage_ready
)
if "!STORAGE_CHOICE:~0,1!"=="3" (
    set "MODELS_ROOT=%~d0\AI_Models"
    goto :storage_ready
)
if "!STORAGE_CHOICE:~0,1!"=="4" (
    set "MODELS_ROOT=C:\AI_Models"
    goto :storage_ready
)
if "!STORAGE_CHOICE:~0,1!"=="5" (
    set "MODELS_ROOT=D:\AI_Models"
    goto :storage_ready
)
if "!STORAGE_CHOICE:~0,1!"=="6" (
    set "MODELS_ROOT=E:\AI_Models"
    goto :storage_ready
)

set "MODELS_ROOT=!SUGGESTED_DIR!"
goto :storage_ready

:prompt_custom_storage
echo.
set /p "MODELS_ROOT=Escribe la ruta completa deseada (ej: C:\iamodels o E:\Modelos): "
if not defined MODELS_ROOT (
    set "MODELS_ROOT=!SUGGESTED_DIR!"
)
goto :storage_ready

:storage_ready
set "MODELS_ROOT=!MODELS_ROOT:"=!"
if "!MODELS_ROOT:~-1!"=="\" if "!MODELS_ROOT:~-2!" neq ":\" set "MODELS_ROOT=!MODELS_ROOT:~0,-1!"

:: Validar espacio libre en la unidad seleccionada
set "TARGET_DRIVE=!MODELS_ROOT:~0,1!"
set "FREE_GB="
powershell -NoProfile -Command "[int]((Get-PSDrive -Name '!TARGET_DRIVE!').Free / 1GB)" > "%TEMP%\freespace.txt" 2>nul
if exist "%TEMP%\freespace.txt" (
    set /p FREE_GB=<"%TEMP%\freespace.txt"
    del "%TEMP%\freespace.txt" 2>nul
)

if not defined FREE_GB goto :storage_space_done
if !FREE_GB! geq 15 goto :storage_space_ok
echo !YELLOW![ALERTA DE ESPACIO]!RESET! La unidad !TARGET_DRIVE!:\ tiene solo !FREE_GB! GB libres.
echo Se recomienda tener al menos 20 GB libres para descargar modelos y dependencias.
set "CONT_SPACE=S"
set /p "CONT_SPACE=Deseas continuar de todos modos o cambiar de ruta? [S=Continuar / N=Cambiar] (Enter para S): "
if /i "!CONT_SPACE:~0,1!"=="N" goto :select_storage
goto :storage_space_done

:storage_space_ok
echo !GREEN![OK]!RESET! Espacio libre en unidad !TARGET_DRIVE!:\: !FREE_GB! GB - Espacio suficiente verificado.

:storage_space_done
echo !GREEN![OK]!RESET! Directorio raiz de almacenamiento seleccionado: "!MODELS_ROOT!"
echo.

set "HF_HOME=%MODELS_ROOT%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%MODELS_ROOT%\huggingface\hub"
set "TORCH_HOME=%MODELS_ROOT%\torch"
set "U2NET_HOME=%MODELS_ROOT%\u2net"
set "PIP_CACHE_DIR=%MODELS_ROOT%\pip_cache"
set "TEMP=%MODELS_ROOT%\temp"
set "TMP=%MODELS_ROOT%\temp"
set "OUTPUT_DIR=%MODELS_ROOT%\output"
set "REPOS_DIR=%MODELS_ROOT%\repos"

:: Verificar carpetas existentes o crearlas
set "ALL_DIRS_EXIST=1"
if not exist "%MODELS_ROOT%\huggingface" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\huggingface" 2>nul)
if not exist "%MODELS_ROOT%\torch" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\torch" 2>nul)
if not exist "%MODELS_ROOT%\u2net" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\u2net" 2>nul)
if not exist "%MODELS_ROOT%\pip_cache" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\pip_cache" 2>nul)
if not exist "%MODELS_ROOT%\temp" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\temp" 2>nul)
if not exist "%MODELS_ROOT%\output" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\output" 2>nul)
if not exist "%MODELS_ROOT%\repos" (set "ALL_DIRS_EXIST=0" & mkdir "%MODELS_ROOT%\repos" 2>nul)

if "!ALL_DIRS_EXIST!"=="1" (
    echo !GREEN![OK]!RESET! Estructura de carpetas de almacenamiento y cache ya existente. Omitiendo creacion.
) else (
    echo !GREEN![OK]!RESET! Estructura de carpetas creada correctamente en "!MODELS_ROOT!".
)
echo.

:: ==============================================================================
:: 2. DETECCION Y ASISTENTE INTELIGENTE DE PYTHON
:: ==============================================================================
:retry_python_detection
echo [1/8] Verificando interprete de Python...

set "PY_CMD="

:: Comprobar si ya existe un entorno virtual previo y obtener su python
if exist "%MODELS_ROOT%\venv_3dgen\Scripts\python.exe" (
    set "PY_CMD=%MODELS_ROOT%\venv_3dgen\Scripts\python.exe"
    echo !GREEN![OK]!RESET! Detectado interprete de entorno virtual existente: "!PY_CMD!"
    goto :python_found
)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
    echo !GREEN![OK]!RESET! Detectado interprete de entorno virtual local: "!PY_CMD!"
    goto :python_found
)

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
)

if not defined PY_CMD (
    for /f "delims=" %%I in ('where python.exe 2^>nul') do (
        if not defined PY_CMD (
            echo %%I | findstr /i "WindowsApps Inkscape" >nul
            if !errorlevel! neq 0 (
                "%%I" -c "import sys; exit(0)" >nul 2>&1
                if !errorlevel! equ 0 set "PY_CMD=%%I"
            )
        )
    )
)

if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python310\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python310\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python311\python.exe" set "PY_CMD=%ProgramFiles%\Python311\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python310\python.exe" set "PY_CMD=%ProgramFiles%\Python310\python.exe"
if not defined PY_CMD if exist "D:\Programas\Python311\python.exe" set "PY_CMD=D:\Programas\Python311\python.exe"

if defined PY_CMD (
    echo !GREEN![OK]!RESET! Interprete Python localizado con exito: "!PY_CMD!"
    goto :python_found
)

:: ASISTENTE: Si no se encontro Python
echo.
echo ==============================================================================
echo !YELLOW![ASISTENTE] NO SE DETECTO PYTHON EN EL SISTEMA!RESET!
echo ==============================================================================
echo Para este generador 3D se requiere Python 3.10 o 3.11 de 64 bits.
echo.
echo Opciones de ayuda:
echo   [1] Instalar Python 3.11 automaticamente con winget (Recomendado)
echo   [2] Abrir enlace web de descarga oficial de Python (python.org)
echo   [3] Especificar ruta manual a tu python.exe
echo   [4] Salir
echo.
set "PY_ASSIST=1"
set /p "PY_ASSIST=Elige una opcion [1-4] (Enter para opcion 1): "
echo.

if "!PY_ASSIST:~0,1!"=="2" (
    echo [INFO] Abriendo pagina de descarga oficial...
    start https://www.python.org/downloads/release/python-3119/
    echo Recuerda marcar "Add python.exe to PATH".
    pause
    goto :retry_python_detection
)

if "!PY_ASSIST:~0,1!"=="3" (
    set /p "MANUAL_PY=Ingresa la ruta completa de tu python.exe: "
    if defined MANUAL_PY (
        set "MANUAL_PY=!MANUAL_PY:"=!"
        if exist "!MANUAL_PY!" (
            set "PY_CMD=!MANUAL_PY!"
            goto :python_found
        ) else (
            echo [ERROR] La ruta no existe.
            goto :retry_python_detection
        )
    )
)

if "!PY_ASSIST:~0,1!"=="4" (
    echo Instalacion cancelada por el usuario.
    exit /b 1
)

echo [INFO] Intentando instalar Python 3.11 con winget...
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
goto :retry_python_detection

:python_found
echo.

:: Verificar que el Python seleccionado tenga pip funcional
"!PY_CMD!" -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ASISTENTE] Reparando pip en el interprete seleccionado...
    "!PY_CMD!" -m ensurepip --default-pip >nul 2>&1
    "!PY_CMD!" -m pip --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo !GREEN![OK]!RESET! pip reparado y listo para usar.
    ) else (
        echo !YELLOW![AVISO]!RESET! pip no respondio, se intentara continuar con el venv.
    )
) else (
    echo !GREEN![OK]!RESET! Gestor de paquetes pip disponible y verificado.
)
echo.

:: ==============================================================================
:: 3. CREACION Y ACTIVACION DEL ENTORNO VIRTUAL (.venv / venv_3dgen)
:: ==============================================================================
echo [2/8] Configurando entorno virtual...

if /i "%MODELS_ROOT%"=="%~dp0AI_Models" (
    set "VENV_DIR=%~dp0.venv"
) else (
    set "VENV_DIR=%MODELS_ROOT%\venv_3dgen"
)

if exist "!VENV_DIR!\Scripts\activate.bat" (
    echo !GREEN![OK]!RESET! Entorno virtual ya existente detectado en "!VENV_DIR!". Omitiendo creacion.
) else (
    echo [INFO] Creando nuevo entorno virtual en "!VENV_DIR!"...
    "!PY_CMD!" -m venv "!VENV_DIR!"
    if not exist "!VENV_DIR!\Scripts\activate.bat" (
        echo [INFO] Intentando crear entorno alternativo local en .venv...
        set "VENV_DIR=%~dp0.venv"
        "!PY_CMD!" -m venv "!VENV_DIR!"
    )
)

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
    echo !RED![ERROR CRITICO]!RESET! No se pudo generar el entorno virtual en !VENV_DIR!
    pause
    exit /b 1
)

call "!ACTIVATE_BAT!"
if !errorlevel! neq 0 (
    echo !RED![ERROR]!RESET! No se pudo activar el entorno virtual.
    pause
    exit /b 1
)
echo !GREEN![OK]!RESET! Entorno virtual activado con exito: "!ACTIVATE_BAT!"
echo.

:: ==============================================================================
:: 4. INSTALACION Y VERIFICACION DE PYTORCH CON CUDA 12.4
:: ==============================================================================
echo [3/8] Verificando PyTorch con aceleracion CUDA...

python -c "import torch; assert torch.cuda.is_available() and torch.version.cuda is not None" >nul 2>&1
if !errorlevel! equ 0 goto :pytorch_ok

echo [INFO] Instalando PyTorch con aceleracion CUDA 12.4 (Descarga optimizada)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --cache-dir "%PIP_CACHE_DIR%"
python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! PyTorch con aceleracion CUDA instalado con exito.
) else (
    echo !YELLOW![AVISO]!RESET! PyTorch se instalo pero no detecto aceleracion CUDA activa. Se usara modo CPU.
)
goto :pytorch_done

:pytorch_ok
echo !GREEN![OK]!RESET! PyTorch con aceleracion NVIDIA CUDA ya instalado y verificado. Omitiendo reinstalacion:
python -c "import torch; print('     -> PyTorch ' + torch.__version__ + ' | CUDA ' + str(torch.version.cuda) + ' | ' + torch.cuda.get_device_name(0))"

:pytorch_done
echo.

:: ==============================================================================
:: 5. INSTALACION Y VERIFICACION DE LIBRERIAS BASE 3D
:: ==============================================================================
echo [4/8] Verificando librerias 3D base y utilidades...

python -c "import trimesh, pymeshlab, pyfqmr, rembg, pygltflib, gradio, xatlas, PIL, cv2" >nul 2>&1
if !errorlevel! equ 0 goto :base_deps_ok

echo [INFO] Instalando dependencias base desde requirements.txt...
pip install -r requirements.txt --cache-dir "%PIP_CACHE_DIR%"
echo !GREEN![OK]!RESET! Dependencias base instaladas.
goto :base_deps_done

:base_deps_ok
echo !GREEN![OK]!RESET! Dependencias base 3D y utilidades ya instaladas. Omitiendo instalacion.

:base_deps_done
echo.

:: ==============================================================================
:: 6. SELECCION DE MOTORES DE GENERACION 3D (Hunyuan3D-2 vs SF3D vs Ambos)
:: ==============================================================================
echo ==============================================================================
echo       SELECCION DEL MOTOR DE GENERACION 3D
echo ==============================================================================
echo Elige que motores deseas habilitar y configurar en este equipo:
echo.
echo   [1] Tencent Hunyuan3D-2 Turbo
echo       - Especialidad: Geometria solida de alta definicion (modo arcilla / CAD)
echo       - Sin texturas por defecto (ideal para impresion 3D / amigos / mallas puras)
echo       - Opcional: Proyeccion de color RGB
echo.
echo   [2] Stability AI Stable Fast 3D (SF3D)
echo       - Especialidad: Generacion ultrarrapida (~1s) con texturas PBR completas
echo       - Con texturas por defecto (ideal para videojuegos / Godot 4 / Kaboomi Party)
echo       - Incluye mapas completos: Albedo, Normal, Roughness y Metallic
echo.
echo   [3] Ambos Motores (Suite Hibrida Completa) - RECOMENDADO
echo       - Instala y configura ambos motores
echo       - Permite alternar libremente desde la interfaz WebUI con un solo clic
echo       - Cubre el caso de impresion 3D solida y desarrollo de videojuegos
echo.

set "ENGINE_CHOICE=3"
set /p "ENGINE_CHOICE=Elige una opcion [1, 2 o 3] (Enter para opcion 3 - Ambos): "
echo.

set "ENGINE_CHOICE=!ENGINE_CHOICE: =!"
if not "!ENGINE_CHOICE!"=="1" if not "!ENGINE_CHOICE!"=="2" set "ENGINE_CHOICE=3"

if "!ENGINE_CHOICE!"=="1" goto :setup_hunyuan_section
if "!ENGINE_CHOICE!"=="2" goto :setup_sf3d_section
goto :setup_both_section

:: ------------------------------------------------------------------------------
:setup_hunyuan_section
echo [5/8] Configurando motor: Tencent Hunyuan3D-2 Turbo...
call :check_and_install_hunyuan
goto :step_gpu_check

:: ------------------------------------------------------------------------------
:setup_sf3d_section
echo [5/8] Configurando motor: Stability AI Stable Fast 3D (SF3D)...
call :check_and_install_sf3d
goto :step_gpu_check

:: ------------------------------------------------------------------------------
:setup_both_section
echo [5/8] Configurando Motor 1 de 2: Tencent Hunyuan3D-2 Turbo...
call :check_and_install_hunyuan
echo.
echo [6/8] Configurando Motor 2 de 2: Stability AI Stable Fast 3D (SF3D)...
call :check_and_install_sf3d
goto :step_gpu_check

:: ==============================================================================
:: SUBRUTINAS DE INSTALACION DE MOTORES
:: ==============================================================================
:check_and_install_hunyuan
python -c "import hy3dgen" >nul 2>&1
if !errorlevel! equ 0 goto :hy3dgen_ok

echo [INFO] hy3dgen no encontrado. Verificando Git para instalacion...
where git.exe >nul 2>&1
if !errorlevel! equ 0 (
    pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git --cache-dir "%PIP_CACHE_DIR%"
) else (
    pip install https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip --cache-dir "%PIP_CACHE_DIR%"
)
goto :hy3dgen_weights_check

:hy3dgen_ok
echo !GREEN![OK]!RESET! Modulo oficial hy3dgen ya instalado en el entorno.

:hy3dgen_weights_check
if exist "%MODELS_ROOT%\tencent\Hunyuan3D-2\hunyuan3d-dit-v2-0-turbo\model.fp16.safetensors" goto :hy_weights_local
if exist "%HUGGINGFACE_HUB_CACHE%\models--tencent--Hunyuan3D-2" goto :hy_weights_cache
echo [INFO] Los pesos DiT de Hunyuan3D-2 (~4.93 GB) se descargaran de forma automatica en el primer uso a Disco %TARGET_DRIVE%:\.
exit /b 0

:hy_weights_local
echo !GREEN![OK]!RESET! Pesos Hunyuan3D-2 Turbo detectados (~4.93 GB en %MODELS_ROOT%\tencent\Hunyuan3D-2). Omitiendo descarga.
exit /b 0

:hy_weights_cache
echo !GREEN![OK]!RESET! Pesos Hunyuan3D-2 Turbo detectados en cache HuggingFace.
exit /b 0

:check_and_install_sf3d
if exist "%REPOS_DIR%\stable-fast-3d\run.py" goto :sf3d_repo_ok
if exist "%REPOS_DIR%\stable-fast-3d\sf3d\system.py" goto :sf3d_repo_ok

echo [INFO] Clonando repositorio oficial de Stable Fast 3D...
where git.exe >nul 2>&1
if !errorlevel! equ 0 (
    git clone https://github.com/Stability-AI/stable-fast-3d.git "%REPOS_DIR%\stable-fast-3d"
) else (
    echo [INFO] Descargando repositorio ZIP de Stable Fast 3D...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/Stability-AI/stable-fast-3d/archive/refs/heads/main.zip' -OutFile '%TEMP%\sf3d.zip'; Expand-Archive -Path '%TEMP%\sf3d.zip' -DestinationPath '%REPOS_DIR%'; Rename-Item '%REPOS_DIR%\stable-fast-3d-main' 'stable-fast-3d'; Remove-Item '%TEMP%\sf3d.zip'" 2>nul
)
goto :sf3d_deps_check

:sf3d_repo_ok
echo !GREEN![OK]!RESET! Repositorio oficial de Stable Fast 3D presente en "%REPOS_DIR%\stable-fast-3d". Omitiendo clonacion.

:sf3d_deps_check
python -c "import jaxtyping, open_clip, gpytoolbox, pynanoinstantmeshes, numba, xatlas" >nul 2>&1
if !errorlevel! equ 0 goto :sf3d_deps_ok

echo [INFO] Instalando dependencias especializadas de SF3D...
pip install jaxtyping open-clip-torch gpytoolbox pynanoinstantmeshes numba xatlas --cache-dir "%PIP_CACHE_DIR%"
echo !GREEN![OK]!RESET! Dependencias especializadas de SF3D instaladas.
goto :sf3d_token_check

:sf3d_deps_ok
echo !GREEN![OK]!RESET! Dependencias especializadas de SF3D (jaxtyping, open-clip, gpytoolbox, numba, xatlas) ya instaladas. Omitiendo.

:sf3d_token_check
python -c "import huggingface_hub; exit(0 if huggingface_hub.get_token() else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :sf3d_token_ok

echo.
echo ------------------------------------------------------------------------------
echo !CYAN![ACCESO A STABLE FAST 3D]!RESET!
echo SF3D es un modelo con licencia de Stability AI en Hugging Face.
echo Pasos rapidos para activarlo:
echo   1. Inicia sesion en https://huggingface.co/stabilityai/stable-fast-3d y acepta los terminos.
echo   2. Copia tu token de lectura desde: https://huggingface.co/settings/tokens
echo ------------------------------------------------------------------------------
set "INPUT_HF_TOKEN="
set /p "INPUT_HF_TOKEN=Pega tu token de Hugging Face (o presiona Enter para configurarlo luego): "
if not defined INPUT_HF_TOKEN goto :sf3d_token_skipped
python -c "import huggingface_hub; huggingface_hub.login(token='!INPUT_HF_TOKEN!', add_to_git_credential=False); print('Token guardado.')" 2>nul
if !errorlevel! equ 0 echo !GREEN![OK]!RESET! Token de Hugging Face configurado con exito.
exit /b 0

:sf3d_token_ok
echo !GREEN![OK]!RESET! Token de Hugging Face detectado para acceso a modelos SF3D.
exit /b 0

:sf3d_token_skipped
echo [INFO] Podras configurar el token mas adelante con: huggingface-cli login
exit /b 0

:: ==============================================================================
:: 7. VERIFICACION DE GPU CUDA Y HARDWARE
:: ==============================================================================
:step_gpu_check
echo.
echo [7/8] Verificando aceleracion por hardware GPU (NVIDIA CUDA)...

where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 (
    if exist "%SystemRoot%\System32\nvidia-smi.exe" set "PATH=%SystemRoot%\System32;!PATH!"
    if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "PATH=%ProgramFiles%\NVIDIA Corporation\NVSMI;!PATH!"
)

where nvidia-smi.exe >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader^,nounits 2^>nul') do (
        echo !GREEN![OK]!RESET! Tarjeta Grafica Detectada: %%G MB VRAM
    )
) else (
    echo !YELLOW![AVISO]!RESET! nvidia-smi no detecto tarjeta NVIDIA.
)

python -c "import torch; d = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; v = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0; print('>>> CUDA Activo: ' + str(torch.cuda.is_available()) + ' | Dispositivo: ' + d + ' | VRAM: ' + f'{v:.1f}' + ' GB')"
echo.

:: ==============================================================================
:: 8. GENERACION Y ACTUALIZACION DE LANZADORES DINAMICOS
:: ==============================================================================
echo [8/8] Generando y actualizando lanzadores ejecutables (.bat)...

:: 1. webui.bat (WebUI General: Hunyuan3D-2 + Sin Textura por defecto)
echo @echo off > "%~dp0webui.bat"
echo setlocal >> "%~dp0webui.bat"
echo cd /d "%%~dp0" >> "%~dp0webui.bat"
echo chcp 65001 ^>nul >> "%~dp0webui.bat"
echo echo ============================================================================== >> "%~dp0webui.bat"
echo echo       3D AI STUDIO - INTERFAZ WEB GENERAL >> "%~dp0webui.bat"
echo echo       Motor Predeterminado: Hunyuan3D-2 Turbo • Modo Arcilla / Sin Textura >> "%~dp0webui.bat"
echo echo       Almacenamiento de Modelos y Cache: 100%%%% en Disco %TARGET_DRIVE%:\ >> "%~dp0webui.bat"
echo echo ============================================================================== >> "%~dp0webui.bat"
echo set "HF_HOME=%HF_HOME%" >> "%~dp0webui.bat"
echo set "HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%" >> "%~dp0webui.bat"
echo set "TORCH_HOME=%TORCH_HOME%" >> "%~dp0webui.bat"
echo set "U2NET_HOME=%U2NET_HOME%" >> "%~dp0webui.bat"
echo set "PIP_CACHE_DIR=%PIP_CACHE_DIR%" >> "%~dp0webui.bat"
echo set "TEMP=%TEMP%" >> "%~dp0webui.bat"
echo set "TMP=%TMP%" >> "%~dp0webui.bat"
echo set "HY3DGEN_MODELS=%MODELS_ROOT%" >> "%~dp0webui.bat"
echo set "STUDIO_PROFILE=general" >> "%~dp0webui.bat"
echo if exist "%%~dp0.venv\Scripts\activate.bat" ( >> "%~dp0webui.bat"
echo     call "%%~dp0.venv\Scripts\activate.bat" >> "%~dp0webui.bat"
echo ) else if exist "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" ( >> "%~dp0webui.bat"
echo     call "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" >> "%~dp0webui.bat"
echo ) else ( >> "%~dp0webui.bat"
echo     echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. >> "%~dp0webui.bat"
echo     pause >> "%~dp0webui.bat"
echo     exit /b 1 >> "%~dp0webui.bat"
echo ) >> "%~dp0webui.bat"
echo python webui.py --mode general >> "%~dp0webui.bat"
echo pause >> "%~dp0webui.bat"

:: 2. webui-kambo.bat (WebUI Kaboomi: SF3D + Con Textura PBR por defecto)
echo @echo off > "%~dp0webui-kambo.bat"
echo setlocal >> "%~dp0webui-kambo.bat"
echo cd /d "%%~dp0" >> "%~dp0webui-kambo.bat"
echo chcp 65001 ^>nul >> "%~dp0webui-kambo.bat"
echo echo ============================================================================== >> "%~dp0webui-kambo.bat"
echo echo       KABOOMI 3D STUDIO - EDICION KABOOMI PARTY (GODOT 4) >> "%~dp0webui-kambo.bat"
echo echo       Motor Predeterminado: Stable Fast 3D (SF3D) • Con Textura PBR Completa >> "%~dp0webui-kambo.bat"
echo echo       Almacenamiento de Modelos y Cache: 100%%%% en Disco %TARGET_DRIVE%:\ >> "%~dp0webui-kambo.bat"
echo echo ============================================================================== >> "%~dp0webui-kambo.bat"
echo set "HF_HOME=%HF_HOME%" >> "%~dp0webui-kambo.bat"
echo set "HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%" >> "%~dp0webui-kambo.bat"
echo set "TORCH_HOME=%TORCH_HOME%" >> "%~dp0webui-kambo.bat"
echo set "U2NET_HOME=%U2NET_HOME%" >> "%~dp0webui-kambo.bat"
echo set "PIP_CACHE_DIR=%PIP_CACHE_DIR%" >> "%~dp0webui-kambo.bat"
echo set "TEMP=%TEMP%" >> "%~dp0webui-kambo.bat"
echo set "TMP=%TMP%" >> "%~dp0webui-kambo.bat"
echo set "HY3DGEN_MODELS=%MODELS_ROOT%" >> "%~dp0webui-kambo.bat"
echo set "STUDIO_PROFILE=kambo" >> "%~dp0webui-kambo.bat"
echo if exist "%%~dp0.venv\Scripts\activate.bat" ( >> "%~dp0webui-kambo.bat"
echo     call "%%~dp0.venv\Scripts\activate.bat" >> "%~dp0webui-kambo.bat"
echo ) else if exist "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" ( >> "%~dp0webui-kambo.bat"
echo     call "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" >> "%~dp0webui-kambo.bat"
echo ) else ( >> "%~dp0webui-kambo.bat"
echo     echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. >> "%~dp0webui-kambo.bat"
echo     pause >> "%~dp0webui-kambo.bat"
echo     exit /b 1 >> "%~dp0webui-kambo.bat"
echo ) >> "%~dp0webui-kambo.bat"
echo python webui.py --mode kambo >> "%~dp0webui-kambo.bat"
echo pause >> "%~dp0webui-kambo.bat"

:: 3. run_webui.bat (Acceso rapido a WebUI General)
echo @echo off > "%~dp0run_webui.bat"
echo setlocal >> "%~dp0run_webui.bat"
echo cd /d "%%~dp0" >> "%~dp0run_webui.bat"
echo chcp 65001 ^>nul >> "%~dp0run_webui.bat"
echo call "%%~dp0webui.bat" >> "%~dp0run_webui.bat"

:: 4. run_webui_kambo.bat (Acceso rapido a WebUI Kaboomi Party)
echo @echo off > "%~dp0run_webui_kambo.bat"
echo setlocal >> "%~dp0run_webui_kambo.bat"
echo cd /d "%%~dp0" >> "%~dp0run_webui_kambo.bat"
echo chcp 65001 ^>nul >> "%~dp0run_webui_kambo.bat"
echo call "%%~dp0webui-kambo.bat" >> "%~dp0run_webui_kambo.bat"

:: 5. run_cli.bat (Ejecutor CLI con soporte para --model y --no-texture)
echo @echo off > "%~dp0run_cli.bat"
echo setlocal >> "%~dp0run_cli.bat"
echo chcp 65001 ^>nul >> "%~dp0run_cli.bat"
echo set "HF_HOME=%HF_HOME%" >> "%~dp0run_cli.bat"
echo set "HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%" >> "%~dp0run_cli.bat"
echo set "TORCH_HOME=%TORCH_HOME%" >> "%~dp0run_cli.bat"
echo set "U2NET_HOME=%U2NET_HOME%" >> "%~dp0run_cli.bat"
echo set "PIP_CACHE_DIR=%PIP_CACHE_DIR%" >> "%~dp0run_cli.bat"
echo set "TEMP=%TEMP%" >> "%~dp0run_cli.bat"
echo set "TMP=%TMP%" >> "%~dp0run_cli.bat"
echo set "HY3DGEN_MODELS=%MODELS_ROOT%" >> "%~dp0run_cli.bat"
echo if exist "%%~dp0.venv\Scripts\activate.bat" ( >> "%~dp0run_cli.bat"
echo     call "%%~dp0.venv\Scripts\activate.bat" >> "%~dp0run_cli.bat"
echo ) else if exist "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" ( >> "%~dp0run_cli.bat"
echo     call "%MODELS_ROOT%\venv_3dgen\Scripts\activate.bat" >> "%~dp0run_cli.bat"
echo ) else ( >> "%~dp0run_cli.bat"
echo     echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat. >> "%~dp0run_cli.bat"
echo     exit /b 1 >> "%~dp0run_cli.bat"
echo ) >> "%~dp0run_cli.bat"
echo python img2glb.py %%* >> "%~dp0run_cli.bat"

echo !GREEN![OK]!RESET! Todos los lanzadores actualizados y sincronizados con Disco %TARGET_DRIVE%:\.
echo.

:: ==============================================================================
:: RESUMEN FINAL Y GUIA DE USO
:: ==============================================================================
echo ==============================================================================
echo  !GREEN!¡INSTALACION Y CONFIGURACION COMPLETADAS EXITOSAMENTE!!RESET!
echo ==============================================================================
echo  Modelos y Cache:  %MODELS_ROOT% (100%%%% fuera de C:\)
echo  Entorno Virtual:  !VENV_DIR!
echo  Motor(es) Config: !ENGINE_CHOICE!
echo.
echo  ----------------------------------------------------------------------------
echo  MODOS DE EJECUCION WEB DISPONIBLES:
echo  ----------------------------------------------------------------------------
echo  1. Interfaz Web General (Impresion 3D / CAD / Modelos sin textura):
echo     ^> !CYAN!run_webui.bat!RESET!  (o webui.bat)
echo     * Por defecto: Hunyuan3D-2 Turbo ^| Modo Arcilla (Sin Textura)
echo     * Permite activar texturas o cambiar a SF3D desde el menu desplegable.
echo.
echo  2. Interfaz Kaboomi Party (Videojuegos Godot 4 / Assets PBR):
echo     ^> !CYAN!run_webui_kambo.bat!RESET!  (o webui-kambo.bat)
echo     * Por defecto: Stable Fast 3D (SF3D) ^| Con Textura PBR Completa (~1s)
echo     * Guarda directamente en assets/models de Kaboomi Party.
echo.
echo  ----------------------------------------------------------------------------
echo  MODO CONSOLA (CLI):
echo  ----------------------------------------------------------------------------
echo  ^> !CYAN!run_cli.bat --input mi_foto.png --model sf3d --output asset.glb!RESET!
echo  ^> !CYAN!run_cli.bat --input mi_foto.png --model hunyuan --no-texture --output pieza.glb!RESET!
echo ==============================================================================
echo.
pause
