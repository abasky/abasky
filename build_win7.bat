@echo off
setlocal
cd /d "%~dp0"
rem Builds a 32-bit program that runs on Windows 7 (and also on 10/11).
rem Run this on ANY Windows PC that has Python 3.8.10 32-bit installed.
rem Output: dist\Abasky-IQ-Win7\Abasky-IQ-Win7.exe
set VENV=C:\IQ_env32

echo [1/4] Looking for Python 3.8 32-bit ...
py -3.8-32 -c "import sys,struct; print('found Python', sys.version.split()[0], struct.calcsize('P')*8, 'bit')"
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.8 32-bit was not found.
    echo   1. Download: https://www.python.org/ftp/python/3.8.10/python-3.8.10.exe
    echo      ^(the 32-bit installer - the name does NOT contain amd64^)
    echo   2. Install it ^(keep "py launcher" ticked^), then run this file again.
    pause
    exit /b 1
)

echo [2/4] Creating environment in %VENV% ...
if not exist "%VENV%\Scripts\python.exe" (
    py -3.8-32 -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: could not create the environment.
        pause
        exit /b 1
    )
)

echo [3/4] Installing libraries - this can take a few minutes ...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip uninstall -y numpy
"%VENV%\Scripts\python.exe" -m pip install -r requirements_win7.txt
if errorlevel 1 (
    echo.
    echo ERROR: installing libraries failed. Send me the red messages above.
    pause
    exit /b 1
)

echo [4/4] Building the program ...
"%VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name Abasky-IQ-Win7 --icon iq.ico --collect-data cv2 --hidden-import iq_ai --exclude-module tkinter main.py
if errorlevel 1 (
    echo.
    echo ERROR: build failed. Send me the red messages above.
    pause
    exit /b 1
)

if exist models xcopy /E /I /Y models dist\Abasky-IQ-Win7\models >nul
copy /Y get_models.py dist\Abasky-IQ-Win7\ >nul
copy /Y get_models.bat dist\Abasky-IQ-Win7\ >nul
echo.
echo Done. Copy the whole folder  dist\Abasky-IQ-Win7  to the Windows 7 PC
echo and run  Abasky-IQ-Win7.exe  from inside it.
echo For an installer: open installer.iss in Inno Setup, add  /DBUILD=Win7  or edit the define.
pause
