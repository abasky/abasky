@echo off
setlocal
cd /d "%~dp0"
rem Builds the program for Windows 10/11 (64-bit). Output: dist\Abasky-IQ\Abasky-IQ.exe
rem Short venv path on purpose: avoids the Windows 260-character path limit.
set VENV=C:\IQ_env

echo [1/3] Creating virtual environment in %VENV% ...
if not exist "%VENV%\Scripts\python.exe" (
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

echo [2/3] Installing requirements - this can take a few minutes ...
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo ERROR: installing requirements failed. Read the messages above.
    pause
    exit /b 1
)

echo [3/3] Building the program ...
"%VENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name Abasky-IQ --icon iq.ico --collect-data cv2 --hidden-import iq_ai --exclude-module tkinter main.py
if errorlevel 1 (
    echo ERROR: build failed. Read the messages above.
    pause
    exit /b 1
)

if exist models xcopy /E /I /Y models dist\Abasky-IQ\models >nul
copy /Y get_models.py dist\Abasky-IQ\ >nul
copy /Y get_models.bat dist\Abasky-IQ\ >nul
echo.
echo Done. Program folder: dist\Abasky-IQ  ^(run Abasky-IQ.exe inside it^)
echo To make an installer: open installer.iss with Inno Setup and press Compile.
pause
