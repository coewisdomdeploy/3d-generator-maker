@echo off
setlocal
cd /d "%~dp0"
:: Misma WebUI en modo Kaboomi Party (SF3D + textura PBR, guardado directo en el juego)
set "STUDIO_PROFILE=kambo"
call "%~dp0webui.bat" %*
