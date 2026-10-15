@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
cls
echo ==============================================================================
echo       DIAGNOSTICO PROFUNDO DE ENTORNO - LAPTOP CON NVIDIA RTX 4050
echo ==============================================================================
echo Fecha y hora: %DATE% %TIME%
echo.

:: 1. DISCOS Y ESPACIO LIBRE
echo [1/5] Verificando espacio en disco...
powershell -NoProfile -Command "Get-PSDrive C, D, E -ErrorAction SilentlyContinue | Select-Object Name, @{Name='Libre_GB';Expression={[math]::Round($_.Free/1GB, 1)}}, @{Name='Total_GB';Expression={[math]::Round(($_.Used+$_.Free)/1GB, 1)}} | Format-Table -AutoSize"
echo.

:: 2. DETECCION DE TARJETAS GRAFICAS
echo [2/5] Verificando adaptadores de video y GPU...
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | ForEach-Object { '  - ' + $_.Name + ' (Driver: ' + $_.DriverVersion + ')' }" 2>nul
echo.
where nvidia-smi.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo nvidia-smi detectado:
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
) else (
    echo [AVISO] nvidia-smi.exe no esta en el PATH del sistema.
)
echo.

:: 3. DETECCION DE PYTHON Y ARQUITECTURA (32-bit vs 64-bit)
echo [3/5] Verificando interpretes Python en el sistema...
where py.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo   - Lanzador py.exe localizado:
    for /f "delims=" %%V in ('py -0p 2^>nul') do echo       %%V
    echo   - Arquitectura de 'py':
    py -c "import sys, platform; print('     Python ' + sys.version.split()[0] + ' | Arquitectura: ' + platform.architecture()[0] + ' | 64-bit: ' + str(sys.maxsize > 2**32))" 2>nul
) else (
    echo   - Lanzador py.exe NO encontrado.
)

where python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo   - python.exe en PATH:
    for /f "delims=" %%P in ('where python.exe 2^>nul') do (
        echo       Ruta: %%P
        "%%P" -c "import sys, platform; print('       Version: ' + sys.version.split()[0] + ' | ' + platform.architecture()[0] + ' | 64-bit: ' + str(sys.maxsize > 2**32))" 2>nul
    )
) else (
    echo   - python.exe no encontrado en PATH.
)
echo.

:: 4. VERIFICACION DEL ENTORNO VIRTUAL
echo [4/5] Verificando entornos virtuales existentes...
set "VENV_FOUND="
if exist "%~dp0.venv\Scripts\python.exe" set "VENV_FOUND=%~dp0.venv"
if not defined VENV_FOUND if exist "D:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_FOUND=D:\AI_Models\venv_3dgen"
if not defined VENV_FOUND if exist "C:\AI_Models\venv_3dgen\Scripts\python.exe" set "VENV_FOUND=C:\AI_Models\venv_3dgen"

if defined VENV_FOUND (
    echo   - Entorno virtual localizado en: "!VENV_FOUND!"
    "!VENV_FOUND!\Scripts\python.exe" -c "import sys, platform; print('     Python: ' + sys.version.split()[0] + ' | Arquitectura: ' + platform.architecture()[0] + ' | Es 64-bit: ' + str(sys.maxsize > 2**32))"
    "!VENV_FOUND!\Scripts\pip.exe" --version 2>nul
    echo.
    echo   - Estado de PyTorch en el entorno virtual:
    "!VENV_FOUND!\Scripts\python.exe" -c "import torch; print('     PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('     Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Ninguno')" 2>nul
    if !errorlevel! neq 0 (
        echo     PyTorch NO esta instalado en este entorno virtual.
    )
) else (
    echo   - No se detecto ninguna carpeta venv activa en las rutas tipicas.
)
echo.

:: 5. PRUEBA DE RED DIRECTA A PYTORCH
echo [5/5] Probando conexion hacia servidores de PyTorch CUDA...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'https://download.pytorch.org/whl/cu124' -UseBasicParsing -TimeoutSec 10; Write-Host '  [OK] Conexion a download.pytorch.org exitosa (Status ' $r.StatusCode ')' -ForegroundColor Green } catch { Write-Host '  [ERROR] Fallo conexion a download.pytorch.org: ' $_.Exception.Message -ForegroundColor Red }"
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl' -Method Head -UseBasicParsing -TimeoutSec 10; Write-Host '  [OK] Descarga de archivo de rueda PyTorch CUDA accesible' -ForegroundColor Green } catch { Write-Host '  [AVISO] Verificacion de rueda: ' $_.Exception.Message -ForegroundColor Yellow }"

if defined VENV_FOUND (
    echo.
    echo Probando consulta simulada de pip con diagnostico verbose:
    "!VENV_FOUND!\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --dry-run
)
echo.
echo ==============================================================================
echo                  FIN DEL DIAGNOSTICO
echo ==============================================================================
pause
