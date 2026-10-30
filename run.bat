@echo off
cd /d "%~dp0"
rem Runs the program from source (no exe). Uses the 64-bit env from build.bat.
if not exist "C:\IQ_env\Scripts\python.exe" (
    echo Run build.bat first.
    pause
    exit /b 1
)
"C:\IQ_env\Scripts\python.exe" main.py
if errorlevel 1 pause
