
@echo off
setlocal
cd /d "%~dp0"

if exist "C:\IQ_env32\Scripts\python.exe" (
    "C:\IQ_env32\Scripts\python.exe" get_models.py %*
) else if exist "C:\IQ_env\Scripts\python.exe" (
    "C:\IQ_env\Scripts\python.exe" get_models.py %*
) else (
    py -3.8-32 get_models.py %*
)

pause
