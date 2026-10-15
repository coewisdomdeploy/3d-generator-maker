@echo off 
setlocal 
cd /d "%~dp0" 
chcp 65001 >nul 
call "%~dp0webui.bat" 
