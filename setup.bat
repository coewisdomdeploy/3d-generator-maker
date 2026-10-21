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

set "HF_HOME=!MODELS_ROOT!\huggingface"
set "HUGGINGFACE_HUB_CACHE=!MODELS_ROOT!\huggingface\hub"
set "TORCH_HOME=!MODELS_ROOT!\torch"
set "U2NET_HOME=!MODELS_ROOT!\u2net"
set "PIP_CACHE_DIR=!MODELS_ROOT!\pip_cache"
set "TEMP=!MODELS_ROOT!\temp"
set "TMP=!MODELS_ROOT!\temp"
set "TMPDIR=!MODELS_ROOT!\temp"
set "OUTPUT_DIR=!MODELS_ROOT!\output"
set "REPOS_DIR=!MODELS_ROOT!\repos"

:: Verificar carpetas existentes o crearlas
set "ALL_DIRS_EXIST=1"
if not exist "!MODELS_ROOT!\huggingface" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\huggingface" 2>nul)
if not exist "!MODELS_ROOT!\torch" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\torch" 2>nul)
if not exist "!MODELS_ROOT!\u2net" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\u2net" 2>nul)
if not exist "!MODELS_ROOT!\pip_cache" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\pip_cache" 2>nul)
if not exist "!MODELS_ROOT!\temp" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\temp" 2>nul)
if not exist "!MODELS_ROOT!\output" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\output" 2>nul)
if not exist "!MODELS_ROOT!\repos" (set "ALL_DIRS_EXIST=0" & mkdir "!MODELS_ROOT!\repos" 2>nul)

if "!ALL_DIRS_EXIST!"=="1" (
    echo !GREEN![OK]!RESET! Estructura de carpetas de almacenamiento y cache ya existente. Omitiendo creacion.
) else (
    echo !GREEN![OK]!RESET! Estructura de carpetas creada correctamente en "!MODELS_ROOT!".
)
echo.

:: ==============================================================================
:: 2. DETECCION Y PRIORIZACION DEL LANZADOR 'py' Y INTERPRETE DE PYTHON
:: ==============================================================================
:retry_python_detection
echo [1/9] Verificando interprete de Python (priorizando lanzador oficial 'py')...

set "PY_CMD="
set "PY_LAUNCHER="

rem Asegurar que las rutas oficiales del lanzador py esten en PATH
if exist "%SystemRoot%\py.exe" set "PATH=%SystemRoot%;!PATH!"
if exist "%LocalAppData%\Programs\Python\Launcher\py.exe" set "PATH=%LocalAppData%\Programs\Python\Launcher;!PATH!"

where py.exe >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Lanzador 'py.exe' no encontrado en el sistema. Buscando python.exe directo...
    goto :find_direct_python
)

echo [INFO] Lanzador de Windows 'py.exe' localizado. Buscando version compatible (Python 3.10 a 3.12 de 64 bits)...

set "PY_CANDIDATE=py -3.11-64"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3.11"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3.10-64"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3.10"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3.12-64"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3.12"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3-64"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py -3"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

set "PY_CANDIDATE=py"
!PY_CANDIDATE! -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! equ 0 goto :py_candidate_found

rem Advertir al usuario si py solo tiene versiones incompatibles (como Python 3.13, 3.14 o 3.15)
py -c "import sys; print('[AVISO] Se detecto Python ' + sys.version.split()[0] + '. PyTorch CUDA NO es compatible con versiones superiores a Python 3.12.')" 2>nul
echo !YELLOW![AVISO]!RESET! Buscando instalacion directa de Python 3.11 / 3.10 de 64 bits en el equipo...
goto :find_direct_python

:py_candidate_found
set "PY_LAUNCHER=!PY_CANDIDATE!"
    !PY_LAUNCHER! -c "import sys; print(sys.executable)" > "!TEMP!\py_detected.txt" 2>nul
    if exist "!TEMP!\py_detected.txt" (
        set /p "PY_CMD="<"!TEMP!\py_detected.txt"
        del "!TEMP!\py_detected.txt" 2>nul
    )
if defined PY_CMD (
    echo !GREEN![OK]!RESET! Lanzador '!PY_LAUNCHER!' compatible seleccionado: "!PY_CMD!"
    "!PY_CMD!" -c "import sys; print('     -> Version confirmada: Python ' + sys.version.split()[0] + ' 64-bit (Compatible con PyTorch CUDA)')"
    goto :python_found
)

:find_direct_python
rem Buscar ejecutables python.exe de 64 bits en rutas estandar del sistema
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    "%LocalAppData%\Programs\Python\Python311\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=%LocalAppData%\Programs\Python\Python311\python.exe"
)
if not defined PY_CMD if exist "%ProgramFiles%\Python311\python.exe" (
    "%ProgramFiles%\Python311\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=%ProgramFiles%\Python311\python.exe"
)
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    "%LocalAppData%\Programs\Python\Python310\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=%LocalAppData%\Programs\Python\Python310\python.exe"
)
if not defined PY_CMD if exist "%ProgramFiles%\Python310\python.exe" (
    "%ProgramFiles%\Python310\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=%ProgramFiles%\Python310\python.exe"
)
if not defined PY_CMD if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    "%LocalAppData%\Programs\Python\Python312\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
)
if not defined PY_CMD if exist "D:\Programas\Python311\python.exe" (
    "D:\Programas\Python311\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=D:\Programas\Python311\python.exe"
)
if not defined PY_CMD if exist "C:\Python311\python.exe" (
    "C:\Python311\python.exe" -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=C:\Python311\python.exe"
)

if not defined PY_CMD (
    for /f "delims=" %%I in ('where python.exe 2^>nul') do (
        if not defined PY_CMD (
            echo %%I | findstr /i "WindowsApps Inkscape" >nul
            if !errorlevel! neq 0 (
                "%%I" -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
                if !errorlevel! equ 0 set "PY_CMD=%%I"
            )
        )
    )
)

if defined PY_CMD (
    echo !GREEN![OK]!RESET! Interprete Python de 64 bits localizado con exito: "!PY_CMD!"
    goto :python_found
)

:: ASISTENTE: Si no se encontro Python
echo.
echo ==============================================================================
echo !YELLOW![ASISTENTE] NO SE DETECTO PYTHON DE 64 BITS EN EL SISTEMA!RESET!
echo ==============================================================================
echo Para este generador 3D se requiere Python 3.10 o 3.11 de 64 bits (AMD64).
echo.
echo Opciones de ayuda:
echo   [1] Instalar Python 3.11 (64-bit) automaticamente con winget (Recomendado)
echo   [2] Abrir enlace web de descarga oficial de Python (python.org)
echo   [3] Especificar ruta manual a tu python.exe de 64 bits
echo   [4] Salir
echo.
set "PY_ASSIST=1"
set /p "PY_ASSIST=Elige una opcion [1-4] (Enter para opcion 1): "
echo.

if "!PY_ASSIST:~0,1!"=="2" (
    echo [INFO] Abriendo pagina de descarga oficial...
    start https://www.python.org/downloads/release/python-3119/
    echo Recuerda descargar "Windows installer (64-bit)" y marcar "Add python.exe to PATH".
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

echo [INFO] Intentando instalar Python 3.11 x64 con winget...
winget install -e --id Python.Python.3.11 --architecture x64 --accept-package-agreements --accept-source-agreements
goto :retry_python_detection

:python_found
echo.

:: ==============================================================================
:: 3. CREACION, ACTIVACION Y VALIDACION RIGUROSA DEL ENTORNO VIRTUAL
:: ==============================================================================
echo [2/9] Configurando y validando entorno virtual (usando py / python 64-bit)...

if /i "!MODELS_ROOT!"=="%~dp0AI_Models" (
    set "VENV_DIR=%~dp0.venv"
) else (
    set "VENV_DIR=!MODELS_ROOT!\venv_3dgen"
)

set "VENV_EXISTS=0"
if exist "!VENV_DIR!\Scripts\python.exe" (
    "!VENV_DIR!\Scripts\python.exe" -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        echo !GREEN![OK]!RESET! Entorno virtual existente detectado y validado [Python 3.10-3.12 64-bit] en: "!VENV_DIR!"
        set "VENV_EXISTS=1"
    ) else (
        echo !YELLOW![AVISO]!RESET! El entorno virtual previo es incompatible [detectado 32-bit o Python incompatible]. Recreando...
        rmdir /s /q "!VENV_DIR!" 2>nul
    )
)

if "!VENV_EXISTS!"=="0" (
    echo [INFO] Creando nuevo entorno virtual en "!VENV_DIR!"...
    if defined PY_LAUNCHER (
        echo [INFO] Ejecutando: !PY_LAUNCHER! -m venv "!VENV_DIR!"
        !PY_LAUNCHER! -m venv "!VENV_DIR!"
    ) else (
        echo [INFO] Ejecutando: "!PY_CMD!" -m venv "!VENV_DIR!"
        "!PY_CMD!" -m venv "!VENV_DIR!"
    )
    if not exist "!VENV_DIR!\Scripts\python.exe" (
        echo [INFO] Intentando crear entorno alternativo local en %~dp0.venv...
        set "VENV_DIR=%~dp0.venv"
        if defined PY_LAUNCHER (
            !PY_LAUNCHER! -m venv "!VENV_DIR!"
        ) else (
            "!PY_CMD!" -m venv "!VENV_DIR!"
        )
    )
)

set "VENV_PY=!VENV_DIR!\Scripts\python.exe"
set "VENV_PIP=!VENV_DIR!\Scripts\pip.exe"

if not exist "!VENV_PY!" (
    echo !RED![ERROR CRITICO]!RESET! No se encontro python.exe en !VENV_DIR!\Scripts.
    pause
    exit /b 1
)

echo [INFO] Validando entorno virtual en paso 2...
"!VENV_PY!" -c "import sys; bits = 64 if sys.maxsize > 2**32 else 32; ok = 'OK' if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 'INCOMPATIBLE'; print('     -> Interprete verificado: Python ' + sys.version.split()[0] + ' ' + str(bits) + '-bit [' + ok + ']')"

"!VENV_PY!" -c "import sys; sys.exit(0 if (sys.maxsize > 2**32 and sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo !RED![ERROR FATAL EN PASO 2: VERSION DE PYTHON INCOMPATIBLE]!RESET!
    "!VENV_PY!" -c "import sys; print('Tu entorno virtual esta usando Python ' + sys.version.split()[0] + '.')"
    echo PyTorch con CUDA 12.4 NO existe para Python 3.13, 3.14 o 3.15 ni para 32 bits.
    echo Se requiere estrictamente Python 3.11 o 3.10 de 64 bits.
    echo.
    echo Instala Python 3.11 de 64 bits ejecutando en CMD:
    echo   winget install -e --id Python.Python.3.11 --architecture x64
    pause
    exit /b 1
)

rem Enlace simbolico a .venv si se creo en otra ruta
if /i "!VENV_DIR!" neq "%~dp0.venv" (
    if not exist "%~dp0.venv" mklink /J "%~dp0.venv" "!VENV_DIR!" >nul 2>&1
)

rem Activar el entorno para la sesion del script
if exist "!VENV_DIR!\Scripts\activate.bat" (
    call "!VENV_DIR!\Scripts\activate.bat"
)

echo [INFO] Actualizando pip en el entorno virtual...
"!VENV_PY!" -m pip install --upgrade pip >nul 2>&1
echo !GREEN![OK]!RESET! Gestor pip del entorno verificado y actualizado.
echo !GREEN![OK]!RESET! Paso [2/9] superado exitosamente.
echo.

:: ==============================================================================
:: TIPO DE ACELERACION (decide que PyTorch se instala)
:: ==============================================================================
:: La tarjeta concreta se elige cada vez que se abre la aplicacion (gpu_select.py):
:: aqui solo se decide si se instala PyTorch con CUDA (NVIDIA) o solo CPU.
echo ==============================================================================
echo       TIPO DE ACELERACION PARA LA IA
echo ==============================================================================
echo Adaptadores de video encontrados en Windows:
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | ForEach-Object { '  - ' + $_.Name }" 2>nul
echo.

set "HAS_NVIDIA=0"
set "NVIDIA_COUNT=0"
where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 (
    if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "PATH=%ProgramFiles%\NVIDIA Corporation\NVSMI;!PATH!"
)
where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 goto :accel_list_done
echo Tarjetas NVIDIA compatibles con CUDA:
for /f "tokens=1,2,3 delims=," %%A in ('nvidia-smi --query-gpu=index^,name^,memory.total --format=csv^,noheader^,nounits 2^>nul') do (
    set "HAS_NVIDIA=1"
    set /a NVIDIA_COUNT+=1
    echo   - [%%A] %%B ^(%%C MB VRAM^)
)
:accel_list_done
echo.

if "!HAS_NVIDIA!"=="1" (
    set "GPU_OPT=1"
    if !NVIDIA_COUNT! gtr 1 echo !CYAN![INFO]!RESET! Hay !NVIDIA_COUNT! tarjetas NVIDIA: la aplicacion preguntara cual usar cada vez que se abra.
) else (
    set "GPU_OPT=2"
    echo !YELLOW![AVISO]!RESET! No se detecto ninguna tarjeta NVIDIA ^(o falta el driver^). Intel/AMD no sirven para CUDA.
)
echo   [1] NVIDIA CUDA 12.4 - RECOMENDADO
echo   [2] Solo CPU - muy lento
set /p "GPU_OPT=Elige el tipo de PyTorch a instalar [1/2] (Enter = !GPU_OPT!): "
set "GPU_OPT=!GPU_OPT: =!"
if not "!GPU_OPT!"=="2" set "GPU_OPT=1"
if "!GPU_OPT!"=="1" (
    echo !GREEN![OK]!RESET! Se instalara PyTorch con CUDA 12.4.
    reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "!VENV_PY!" /t REG_SZ /d "GpuPreference=2;" /f >nul 2>&1
) else (
    echo !YELLOW![AVISO]!RESET! Se instalara PyTorch solo CPU.
)
echo.

:: ==============================================================================
:: 4. INSTALACION Y VERIFICACION DE PYTORCH CON ACELERACION CUDA
:: ==============================================================================
echo [3/9] Verificando PyTorch con aceleracion CUDA (usando entorno virtual)...

if "!GPU_OPT!"=="2" (
    echo [INFO] Modo CPU seleccionado. Verificando PyTorch CPU...
    "!VENV_PY!" -c "import torch" >nul 2>&1
    if !errorlevel! equ 0 goto :pytorch_ok
    echo [INFO] Instalando PyTorch en modo CPU...
    "!VENV_PY!" -m pip install torch torchvision --cache-dir "!PIP_CACHE_DIR!"
    goto :pytorch_done
)

"!VENV_PY!" -c "import torch; assert torch.cuda.is_available() and torch.version.cuda is not None" >nul 2>&1
if !errorlevel! equ 0 goto :pytorch_ok

rem Si PyTorch ya esta instalado pero sin CUDA, desinstalar version CPU para permitir reemplazo
"!VENV_PY!" -c "import torch" >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] Detectada instalacion previa de PyTorch sin soporte CUDA.
    echo [INFO] Desinstalando version CPU previa para instalar la version oficial con soporte CUDA...
    "!VENV_PY!" -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
)

echo [INFO] Instalando PyTorch con aceleracion CUDA 12.4...
"!VENV_PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --cache-dir "!PIP_CACHE_DIR!"
"!VENV_PY!" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! PyTorch con aceleracion CUDA 12.4 instalado con exito.
    goto :pytorch_done
)

rem Obtener etiqueta de version de Python (ej: cp311, cp310, cp312)
set "PY_TAG="
for /f "tokens=1,2 delims=." %%A in ('"!VENV_PY!" -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2^>nul') do (
    set "PY_TAG=cp%%A%%B"
)

if defined PY_TAG (
    echo [INFO] Reintentando instalacion con descarga directa de binario oficial (!PY_TAG!)...
    "!VENV_PY!" -m pip install "https://download.pytorch.org/whl/cu124/torch-2.6.0%%2Bcu124-!PY_TAG!-!PY_TAG!-win_amd64.whl" "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%%2Bcu124-!PY_TAG!-!PY_TAG!-win_amd64.whl" --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --cache-dir "!PIP_CACHE_DIR!"
    "!VENV_PY!" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
    if !errorlevel! equ 0 (
        echo !GREEN![OK]!RESET! PyTorch con aceleracion CUDA 12.4 instalado con exito por enlace directo.
        goto :pytorch_done
    )
)

echo [INFO] Reintentando con indice alternativo CUDA 12.1...
"!VENV_PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --cache-dir "!PIP_CACHE_DIR!"
"!VENV_PY!" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! PyTorch con aceleracion CUDA 12.1 instalado con exito.
    goto :pytorch_done
)

echo [INFO] Reintentando con indice alternativo CUDA 11.8...
"!VENV_PY!" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --cache-dir "!PIP_CACHE_DIR!"
"!VENV_PY!" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! PyTorch con aceleracion CUDA 11.8 instalado con exito.
    goto :pytorch_done
)

:: Diagnostico detallado si la instalacion falla
"!VENV_PY!" -c "import torch" >nul 2>&1
if !errorlevel! neq 0 goto :diag_torch_missing

echo !YELLOW![AVISO]!RESET! PyTorch se encuentra instalado pero la aceleracion CUDA no esta activa.
where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 goto :diag_no_driver
echo   - Tarjeta NVIDIA detectada, pero los controladores o el reinicio de Windows estan pendientes.
goto :pytorch_done

:diag_no_driver
echo   - Causa: Falta el controlador oficial de NVIDIA GeForce RTX.
echo     Instala el driver mas reciente desde www.nvidia.com/drivers o GeForce Experience.
goto :pytorch_done

:diag_torch_missing
echo !RED![ERROR CRITICO EN DESCARGA DE PYTORCH]!RESET!
echo No se pudo descargar PyTorch con CUDA en el entorno virtual.
echo Revisa tu conexion a internet o posibles bloqueos de firewall hacia download.pytorch.org.
echo.
echo Presiona una tecla para continuar de todos modos o Ctrl+C para cancelar...
pause
goto :pytorch_done

:pytorch_ok
echo !GREEN![OK]!RESET! PyTorch con aceleracion NVIDIA CUDA verificado y operativo:
"!VENV_PY!" -c "import torch; d = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print('     -> PyTorch ' + torch.__version__ + ' - CUDA ' + str(torch.version.cuda) + ' - ' + d)"

:pytorch_done
echo.

:: ==============================================================================
:: 5. DEPENDENCIAS DEL PROYECTO (sin tocar la version CUDA de PyTorch)
:: ==============================================================================
echo [4/9] Instalando / verificando dependencias del proyecto...
:: Se fija la version instalada de torch/torchvision: si otra dependencia pidiera torch,
:: pip bajaria la version CPU de PyPI y se perderia la aceleracion CUDA.
set "CONSTRAINTS_FILE=!TEMP!\torch_constraints.txt"
"!VENV_PY!" -c "import torch, torchvision; print('torch==' + torch.__version__); print('torchvision==' + torchvision.__version__)" > "!CONSTRAINTS_FILE!" 2>nul
if !errorlevel! neq 0 type nul > "!CONSTRAINTS_FILE!"

:: Python 3.11: entorno EXACTO verificado (requirements-lock.txt, --no-deps).
:: Otras versiones de Python o si falla: requirements.txt con resolucion normal.
"!VENV_PY!" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if !errorlevel! neq 0 goto :deps_loose
echo [INFO] Instalando el entorno exacto verificado (requirements-lock.txt)...
"!VENV_PY!" -m pip install -r "%~dp0requirements-lock.txt" --no-deps --cache-dir "!PIP_CACHE_DIR!"
if !errorlevel! equ 0 goto :deps_verify
echo !YELLOW![AVISO]!RESET! No se pudo instalar el entorno exacto. Probando con requirements.txt...

:deps_loose
"!VENV_PY!" -m pip install -r "%~dp0requirements.txt" -c "!CONSTRAINTS_FILE!" --extra-index-url https://download.pytorch.org/whl/cu124 --cache-dir "!PIP_CACHE_DIR!"

:deps_verify
"!VENV_PY!" -c "import importlib.util as u, sys; m = [x for x in ['torch', 'trimesh', 'pymeshlab', 'pyfqmr', 'rembg', 'pygltflib', 'gradio', 'xatlas', 'PIL', 'cv2', 'diffusers', 'transformers', 'accelerate', 'psutil', 'numba', 'requests', 'huggingface_hub', 'hy3dgen'] if u.find_spec(x) is None]; [print('   - falta:', x) for x in m]; sys.exit(1 if m else 0)"
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! Dependencias del proyecto verificadas.
) else (
    echo !RED![ERROR]!RESET! Faltan las dependencias listadas arriba. Revisa los mensajes de pip.
    pause
)
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
echo [5/9] Configurando motor: Tencent Hunyuan3D-2 Turbo...
call :check_and_install_hunyuan
goto :step_weights

:: ------------------------------------------------------------------------------
:setup_sf3d_section
echo [5/9] Configurando motor: Stability AI Stable Fast 3D (SF3D)...
call :check_and_install_sf3d
goto :step_weights

:: ------------------------------------------------------------------------------
:setup_both_section
echo [5/9] Configurando Motor 1 de 2: Tencent Hunyuan3D-2 Turbo...
call :check_and_install_hunyuan
echo.
echo [5/9] Configurando Motor 2 de 2: Stability AI Stable Fast 3D (SF3D)...
call :check_and_install_sf3d
goto :step_weights

:: ==============================================================================
:: SUBRUTINAS DE INSTALACION DE MOTORES
:: ==============================================================================
:check_and_install_hunyuan
"!VENV_PY!" -c "import hy3dgen" >nul 2>&1
if !errorlevel! equ 0 goto :hy3dgen_ok

echo [INFO] hy3dgen no encontrado. Verificando Git para instalacion...
where git.exe >nul 2>&1
if !errorlevel! equ 0 (
    "!VENV_PY!" -m pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git -c "!CONSTRAINTS_FILE!" --cache-dir "!PIP_CACHE_DIR!"
) else (
    "!VENV_PY!" -m pip install https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip -c "!CONSTRAINTS_FILE!" --cache-dir "!PIP_CACHE_DIR!"
)
"!VENV_PY!" -c "import hy3dgen" >nul 2>&1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! Modulo oficial hy3dgen instalado.
) else (
    echo !RED![ERROR]!RESET! No se pudo instalar hy3dgen. Revisa los mensajes de pip de arriba.
)
exit /b 0

:hy3dgen_ok
echo !GREEN![OK]!RESET! Modulo oficial hy3dgen ya instalado en el entorno.
:: Los pesos se verifican y descargan en el paso 6 (preparar_modelos.py)
exit /b 0

:check_and_install_sf3d
if exist "!REPOS_DIR!\stable-fast-3d\run.py" goto :sf3d_repo_ok
if exist "!REPOS_DIR!\stable-fast-3d\sf3d\system.py" goto :sf3d_repo_ok

echo [INFO] Clonando repositorio oficial de Stable Fast 3D...
where git.exe >nul 2>&1
if !errorlevel! equ 0 (
    git clone https://github.com/Stability-AI/stable-fast-3d.git "!REPOS_DIR!\stable-fast-3d"
) else (
    echo [INFO] Descargando repositorio ZIP de Stable Fast 3D...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/Stability-AI/stable-fast-3d/archive/refs/heads/main.zip' -OutFile '!TEMP!\sf3d.zip'; Expand-Archive -Path '!TEMP!\sf3d.zip' -DestinationPath '!REPOS_DIR!'; Rename-Item '!REPOS_DIR!\stable-fast-3d-main' 'stable-fast-3d'; Remove-Item '!TEMP!\sf3d.zip'" 2>nul
)
goto :sf3d_deps_check

:sf3d_repo_ok
echo !GREEN![OK]!RESET! Repositorio oficial de Stable Fast 3D presente en "!REPOS_DIR!\stable-fast-3d". Omitiendo clonacion.

:sf3d_deps_check
"!VENV_PY!" -c "import jaxtyping, open_clip, gpytoolbox, pynanoinstantmeshes, numba, xatlas" >nul 2>&1
if !errorlevel! equ 0 goto :sf3d_deps_ok

echo [INFO] Instalando dependencias especializadas de SF3D...
"!VENV_PY!" -m pip install jaxtyping open-clip-torch gpytoolbox pynanoinstantmeshes numba xatlas -c "!CONSTRAINTS_FILE!" --cache-dir "!PIP_CACHE_DIR!"
echo !GREEN![OK]!RESET! Dependencias especializadas de SF3D instaladas.
goto :sf3d_token_check

:sf3d_deps_ok
echo !GREEN![OK]!RESET! Dependencias especializadas de SF3D (jaxtyping, open-clip, gpytoolbox, numba, xatlas) ya instaladas. Omitiendo.

:sf3d_token_check
"!VENV_PY!" -c "import huggingface_hub; exit(0 if huggingface_hub.get_token() else 1)" >nul 2>&1
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
"!VENV_PY!" -c "import huggingface_hub; huggingface_hub.login(token=\"!INPUT_HF_TOKEN!\", add_to_git_credential=False); print(\"Token guardado.\")" 2>nul
if !errorlevel! equ 0 echo !GREEN![OK]!RESET! Token de Hugging Face configurado con exito.
exit /b 0

:sf3d_token_ok
echo !GREEN![OK]!RESET! Token de Hugging Face detectado para acceso a modelos SF3D.
exit /b 0

:sf3d_token_skipped
echo [INFO] Podras configurar el token mas adelante con: huggingface-cli login
exit /b 0

:: ==============================================================================
:: 6. PESOS DE LOS MODELOS (descarga previa: el primer uso no se queda esperando)
:: ==============================================================================
:step_weights
echo.
echo [6/9] Verificando pesos de los modelos...
"!VENV_PY!" "%~dp0preparar_modelos.py" status
echo.
if "!ENGINE_CHOICE!"=="2" goto :weights_sf3d
call :ensure_weights dit "Geometria Hunyuan3D-2 Turbo" "4.9 GB"
call :ensure_weights paint "Textura Hunyuan3D-Paint" "9 GB"
call :ensure_weights dit_fast "Hunyuan3D-2 Fast para la casilla Profundidad mejorada" "4.6 GB"
call :ensure_weights depth "Depth Anything V2 para el refuerzo guiado de Profundidad mejorada" "100 MB"
call :ensure_weights dit_full "Hunyuan3D-2 completo para la casilla Maxima calidad" "4.6 GB"

:: Cache fp16 de Hunyuan3D-Paint: se crea una vez y la carga de la textura pasa de ~2 min a ~10 s
"!VENV_PY!" "%~dp0preparar_modelos.py" check paint
if !errorlevel! neq 0 goto :weights_opt_done
"!VENV_PY!" "%~dp0preparar_modelos.py" check paint_opt
if !errorlevel! equ 0 goto :weights_opt_done
echo [INFO] Creando la cache optimizada de Hunyuan3D-Paint: una sola vez, unos 5 minutos...
"!VENV_PY!" "%~dp0preparar_modelos.py" optimize paint
:weights_opt_done
if "!ENGINE_CHOICE!"=="1" goto :step_paint
:weights_sf3d
call :ensure_weights sf3d "Stable Fast 3D" "3.8 GB"
goto :step_paint

:: Subrutina: %1 = item de preparar_modelos.py, %2 = nombre, %3 = tamano
:ensure_weights
"!VENV_PY!" "%~dp0preparar_modelos.py" check %~1
if !errorlevel! equ 0 exit /b 0
set "DL_ANSWER=S"
set /p "DL_ANSWER=Descargar ahora %~2 [%~3]? [S/N] (Enter = S): "
if /i not "!DL_ANSWER:~0,1!"=="S" (
    echo [INFO] Omitido: se descargara automaticamente la primera vez que se use.
    exit /b 0
)
"!VENV_PY!" "%~dp0preparar_modelos.py" download %~1
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! %~2 descargado.
) else (
    echo !YELLOW![AVISO]!RESET! No se pudo descargar %~2. Se reintentara en el primer uso.
)
exit /b 0

:: ==============================================================================
:: 7. TEXTURA HUNYUAN3D-PAINT (extension CUDA custom_rasterizer, opcional)
:: ==============================================================================
:step_paint
echo.
echo [7/9] Textura Hunyuan3D-Paint: aceleradores nativos opcionales...
if "!ENGINE_CHOICE!"=="2" goto :paint_skip
if "!GPU_OPT!"=="2" goto :paint_skip
"!VENV_PY!" "%~dp0preparar_modelos.py" check rasterizer
if !errorlevel! equ 0 (
    echo !GREEN![OK]!RESET! custom_rasterizer instalado: rasterizado de la textura acelerado.
    goto :step_gpu_check
)
echo La textura Hunyuan3D-Paint YA FUNCIONA sin instalar nada (rasterizador PyTorch en GPU).
echo Opcionalmente se compilan las versiones nativas, mas rapidas (mesh_processor C++ y
echo custom_rasterizer CUDA). Requieren:
echo   - Visual Studio 2022 Build Tools con C++  [~7 GB en C:]
echo   - CUDA Toolkit 12.4  [~4 GB en C:, NO cambia el driver de video]
echo No es necesaria: el resultado es el mismo.
set "PAINT_ANSWER=N"
set /p "PAINT_ANSWER=Instalar ahora? Tambien puedes hacerlo luego con instalar_textura_hunyuan.bat [S/N] (Enter = N): "
if /i not "!PAINT_ANSWER:~0,1!"=="S" goto :paint_later
call "%~dp0instalar_textura_hunyuan.bat" --from-setup
goto :step_gpu_check

:paint_later
echo [INFO] Puedes instalarla cuando quieras ejecutando instalar_textura_hunyuan.bat
goto :step_gpu_check

:paint_skip
echo [INFO] Omitido: solo aplica al motor Hunyuan3D-2 con GPU NVIDIA.
goto :step_gpu_check

:: ==============================================================================
:: 8. VERIFICACION DE GPU CUDA Y HARDWARE
:: ==============================================================================
:step_gpu_check
echo.
echo [8/9] Verificando aceleracion por hardware GPU (NVIDIA CUDA)...

where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 (
    if exist "%SystemRoot%\System32\nvidia-smi.exe" set "PATH=%SystemRoot%\System32;!PATH!"
    if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "PATH=%ProgramFiles%\NVIDIA Corporation\NVSMI;!PATH!"
)

where nvidia-smi.exe >nul 2>&1
if !errorlevel! neq 0 goto :skip_nvsmi_final
for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name^,memory.total --format=csv^,noheader^,nounits 2^>nul') do (
    echo !GREEN![OK]!RESET! Tarjeta Grafica Detectada: %%G MB VRAM
)
goto :nvsmi_final_done

:skip_nvsmi_final
echo !YELLOW![AVISO]!RESET! nvidia-smi no detecto tarjeta NVIDIA.

:nvsmi_final_done

"!VENV_PY!" -c "import torch; c=torch.cuda.is_available(); d=torch.cuda.get_device_name(0) if c else 'CPU'; v=round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 1) if c else 0; print('>>> CUDA Activo: ' + str(c) + ' - Dispositivo: ' + str(d) + ' - VRAM: ' + str(v) + ' GB')" >nul 2>&1
if !errorlevel! equ 0 (
    "!VENV_PY!" -c "import torch; c=torch.cuda.is_available(); d=torch.cuda.get_device_name(0) if c else 'CPU'; v=round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 1) if c else 0; print('>>> CUDA Activo: ' + str(c) + ' - Dispositivo: ' + str(d) + ' - VRAM: ' + str(v) + ' GB')"
) else (
    echo !YELLOW![AVISO]!RESET! PyTorch no detectado en el entorno virtual.
)
echo.

:: ==============================================================================
:: 9. CONFIGURACION (.env) Y LANZADORES
:: ==============================================================================
:: Los lanzadores (.bat) estan versionados en git y son la fuente oficial: setup.bat
:: no los regenera. Las rutas se leen del .env (studio_env.py) y la GPU se elige al
:: arrancar (gpu_select.py).
echo [9/9] Guardando configuracion en .env y verificando lanzadores...
"!VENV_PY!" "%~dp0studio_env.py" --set-root "!MODELS_ROOT!"
set "MISSING_LAUNCHER="
for %%L in (webui.bat webui-kambo.bat run_webui.bat run_webui_kambo.bat run_cli.bat instalar_textura_hunyuan.bat) do (
    if not exist "%~dp0%%L" (
        echo !RED![ERROR]!RESET! Falta el lanzador %%L. Restauralo con: git checkout -- %%L
        set "MISSING_LAUNCHER=1"
    )
)
if not defined MISSING_LAUNCHER echo !GREEN![OK]!RESET! Lanzadores presentes.
echo.

echo Estado final de los componentes:
"!VENV_PY!" "%~dp0preparar_modelos.py" status
echo.

:: ==============================================================================
:: RESUMEN FINAL Y GUIA DE USO
:: ==============================================================================
echo ==============================================================================
echo  !GREEN!INSTALACION Y CONFIGURACION COMPLETADAS EXITOSAMENTE!RESET!
echo ==============================================================================
echo  Modelos y Cache:  !MODELS_ROOT! (100%%%% fuera de C:\)
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
