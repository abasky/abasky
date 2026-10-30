@echo off
cd /d "%~dp0"
rem Downloads the optional AI models once. Add  --best  for the larger more accurate person model.
if exist "C:\IQ_env\Scripts\python.exe" (
    "C:\IQ_env\Scripts\python.exe" get_models.py %*
) else (
    py -3 get_models.py %*
)
pause
