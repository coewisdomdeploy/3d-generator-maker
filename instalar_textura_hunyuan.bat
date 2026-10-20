@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul 2>&1

:: Uso: instalar_textura_hunyuan.bat [--from-setup]
::   --from-setup: llamado desde setup.bat (sin pausa final)
set "FROM_SETUP="
if /i "%~1"=="--from-setup" set "FROM_SETUP=1"

echo ==============================================================================
echo   TEXTURA HUNYUAN3D-PAINT: extension CUDA custom_rasterizer
echo   Requisitos: Visual Studio 2022 Build Tools (C++) + CUDA Toolkit 12.4
echo   La compilacion usa solo la CPU (no la GPU) y tarda 5-15 minutos.
echo ==============================================================================
echo.

:: ------------------------------------------------------------------------------
:: 1. Entorno virtual y carpeta de modelos (del .env)
:: ------------------------------------------------------------------------------
set "VENV_PY="
if exist "%~dp0.venv\Scripts\python.exe" set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not defined VENV_PY if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_PY=D:\AI_Models\venv_3dgen\Scripts\python.exe"
if not defined VENV_PY (
    echo [ERROR] No se encontro el entorno virtual. Ejecuta setup.bat primero.
    goto :fin_error
)

"%VENV_PY%" "%~dp0preparar_modelos.py" check rasterizer
if !errorlevel! equ 0 (
    echo [OK] custom_rasterizer ya esta instalado. La textura Hunyuan3D-Paint esta disponible.
    goto :fin_ok
)

set "MODELS_ROOT="
for /f "usebackq delims=" %%R in (`call "%VENV_PY%" -c "import studio_env as s; print(s.MODELS_ROOT)"`) do set "MODELS_ROOT=%%R"
if not defined MODELS_ROOT set "MODELS_ROOT=D:\AI_Models"

:: ------------------------------------------------------------------------------
:: 2. Visual Studio 2022 Build Tools (compilador C++). Se instala ANTES que CUDA
::    para que CUDA registre su integracion con Visual Studio.
:: ------------------------------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
call :find_vs
if defined VS_PATH goto :vs_ok

echo [FALTA] Visual Studio 2022 Build Tools con "Desarrollo para el escritorio con C++".
call :check_c_space 8
set "DO_VS=S"
set /p "DO_VS=Instalarlo ahora con winget? Ocupa ~7 GB en C: [S/N] (Enter = S): "
if /i not "!DO_VS:~0,1!"=="S" goto :vs_manual
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
call :find_vs
if defined VS_PATH goto :vs_ok

:vs_manual
echo [ERROR] Visual Studio Build Tools no esta disponible.
echo         Instalalo a mano: https://visualstudio.microsoft.com/es/visual-cpp-build-tools/
echo         Marca "Desarrollo para el escritorio con C++" y vuelve a ejecutar este archivo.
goto :fin_error

:vs_ok
echo [OK] Visual Studio C++: !VS_PATH!

:: ------------------------------------------------------------------------------
:: 3. CUDA Toolkit 12.4 (misma version que PyTorch cu124). Solo los componentes de
::    desarrollo: NO se instala ni se toca el driver de video.
:: ------------------------------------------------------------------------------
call :find_nvcc
if defined NVCC_EXE goto :cuda_ok

echo [FALTA] CUDA Toolkit 12.4 (compilador nvcc).
call :check_c_space 5
set "DO_CUDA=S"
set /p "DO_CUDA=Instalarlo ahora con winget (sin tocar el driver)? Ocupa ~4 GB en C: [S/N] (Enter = S): "
if /i not "!DO_CUDA:~0,1!"=="S" goto :cuda_manual
winget install -e --id Nvidia.CUDA --version 12.4 --accept-package-agreements --accept-source-agreements --override "-s nvcc_12.4 cudart_12.4 cuda_profiler_api_12.4 nvrtc_12.4 nvrtc_dev_12.4 nvtx_12.4 thrust_12.4 cublas_12.4 cublas_dev_12.4 cufft_12.4 cufft_dev_12.4 curand_12.4 curand_dev_12.4 cusolver_12.4 cusolver_dev_12.4 cusparse_12.4 cusparse_dev_12.4 visual_studio_integration_12.4"
:: La consola actual no ve las variables nuevas: se busca en la ruta estandar
if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
call :find_nvcc
if defined NVCC_EXE goto :cuda_ok

:cuda_manual
echo [ERROR] CUDA Toolkit 12.4 no esta disponible.
echo         Instalalo a mano: https://developer.nvidia.com/cuda-12-4-0-download-archive
echo         Windows / x86_64 / 11 / exe (local). En "Personalizada" desmarca "Driver"
echo         y deja "CUDA". Despues abre una consola NUEVA y vuelve a ejecutar este archivo.
goto :fin_error

:cuda_ok
echo [OK] CUDA Toolkit: !NVCC_EXE!
set "PATH=!CUDA_PATH!\bin;!PATH!"

call "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo activar el entorno del compilador ^(vcvars64.bat^).
    goto :fin_error
)

:: ------------------------------------------------------------------------------
:: 4. Codigo fuente oficial de Hunyuan3D-2
:: ------------------------------------------------------------------------------
set "REPO_DIR=!MODELS_ROOT!\repos\Hunyuan3D-2"
if not exist "!REPO_DIR!\hy3dgen\texgen\custom_rasterizer\setup.py" (
    echo [INFO] Descargando el codigo fuente oficial en !REPO_DIR! ...
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "!REPO_DIR!"
    if !errorlevel! neq 0 (
        echo [ERROR] No se pudo clonar el repositorio. Revisa tu conexion o que git este instalado.
        goto :fin_error
    )
)

:: ------------------------------------------------------------------------------
:: 5. Compilacion (temporales fuera de C:, CPU limitada para no trabar el equipo)
:: ------------------------------------------------------------------------------
set "TEMP=!MODELS_ROOT!\temp"
set "TMP=!MODELS_ROOT!\temp"
if not exist "!TEMP!" mkdir "!TEMP!"
set "DISTUTILS_USE_SDK=1"
:: 8.6 = RTX 30xx, 8.9 = RTX 40xx (4050 / 4060). Solo estas arquitecturas: compila mucho mas rapido.
set "TORCH_CUDA_ARCH_LIST=8.6;8.9"
:: Evita "unsupported Microsoft Visual Studio version" con versiones recientes de MSVC
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler"
:: Pocos procesos en paralelo: el equipo sigue usable mientras compila
set "MAX_JOBS=2"

echo.
echo [INFO] Compilando custom_rasterizer (5-15 min)...
pushd "!REPO_DIR!\hy3dgen\texgen\custom_rasterizer"
"%VENV_PY%" -m pip install . --no-build-isolation
set "BUILD_RC=!errorlevel!"
popd
if not "!BUILD_RC!"=="0" (
    echo [ERROR] La compilacion fallo. Copia el error de arriba para revisarlo.
    goto :fin_error
)

:: ------------------------------------------------------------------------------
:: 6. Verificacion (solo importa el modulo: no ejecuta nada en la GPU)
:: ------------------------------------------------------------------------------
"%VENV_PY%" -c "import torch, custom_rasterizer; print('[OK] custom_rasterizer importado correctamente')"
if !errorlevel! neq 0 (
    echo [ERROR] Se compilo pero no se puede importar. Copia el error de arriba.
    goto :fin_error
)

:fin_ok
echo.
echo   LISTO: activa "Generar textura" en la WebUI con el motor Hunyuan3D-2.
if not defined FROM_SETUP pause
exit /b 0

:fin_error
echo.
echo   No se completo la instalacion de la textura Hunyuan3D-Paint.
echo   La aplicacion funciona igual: usara la textura de contingencia.
if not defined FROM_SETUP pause
exit /b 1

:: ==============================================================================
:: SUBRUTINAS
:: ==============================================================================
:find_vs
set "VS_PATH="
if not exist "%VSWHERE%" exit /b 0
for /f "usebackq delims=" %%V in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%V"
exit /b 0

:find_nvcc
set "NVCC_EXE="
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC_EXE=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC_EXE for /f "delims=" %%N in ('where nvcc 2^>nul') do if not defined NVCC_EXE set "NVCC_EXE=%%N"
if defined NVCC_EXE if not defined CUDA_PATH for %%D in ("!NVCC_EXE!\..\..") do set "CUDA_PATH=%%~fD"
exit /b 0

:check_c_space
set "C_FREE="
for /f %%F in ('powershell -NoProfile -Command "[int]((Get-PSDrive C).Free/1GB)"') do set "C_FREE=%%F"
if defined C_FREE if !C_FREE! lss %~1 (
    echo [AVISO] El disco C: solo tiene !C_FREE! GB libres y esta instalacion necesita ~%~1 GB.
    echo         Libera espacio en C: antes de continuar o la instalacion puede fallar.
)
exit /b 0
