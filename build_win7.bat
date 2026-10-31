
@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV=C:\IQ_env32"
set "PY=%VENV%\Scripts\python.exe"
set "DIST=dist\Abasky-IQ-Win7"
set "MODEL=models\modnet_photographic.onnx"

echo ==========================================
echo       ABASKY IQ - WINDOWS 7 BUILD
echo       Python 3.8.10 / 32-bit
echo       MODNet + ONNX Runtime
echo ==========================================
echo.

echo [1/6] Checking Python 3.8.10 32-bit...

py -3.8-32 -c "import sys,struct; assert sys.version_info[:3] == (3,8,10) and struct.calcsize('P')*8 == 32, 'Python must be 3.8.10 32-bit'; print(sys.version)"
if errorlevel 1 (
    echo ERROR: Python 3.8.10 32-bit is required.
    pause
    exit /b 1
)

echo.
echo [2/6] Checking virtual environment...

if not exist "%PY%" (
    py -3.8-32 -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
)

"%PY%" -c "import sys,struct; assert sys.version_info[:3] == (3,8,10) and struct.calcsize('P')*8 == 32, 'Wrong Python version or architecture'; print('Virtual environment OK')"
if errorlevel 1 (
    echo ERROR: Wrong Python environment.
    pause
    exit /b 1
)

echo.
echo [3/6] Installing pinned libraries...

"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :PIP_ERROR

"%PY%" -m pip uninstall -y numpy

"%PY%" -m pip install --no-cache-dir -r requirements_win7.txt
if errorlevel 1 goto :PIP_ERROR

"%PY%" -m pip check
if errorlevel 1 goto :PIP_ERROR

echo.
echo [4/6] Testing ONNX Runtime...

"%PY%" -c "import sys,struct,numpy,cv2,onnxruntime as ort; print('Python:',sys.version); print('Architecture:',struct.calcsize('P')*8,'bit'); print('NumPy:',numpy.__version__); print('OpenCV:',cv2.__version__); print('ONNX Runtime:',ort.__version__); print('Providers:',ort.get_available_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers(); print('ONNX Runtime: OK')"
if errorlevel 1 (
    echo ERROR: ONNX Runtime failed its test.
    pause
    exit /b 1
)

echo.
echo [5/6] Testing MODNet model...

if not exist "%MODEL%" (
    echo ERROR: MODNet model is missing:
    echo %MODEL%
    echo Run get_models.py first.
    pause
    exit /b 1
)

"%PY%" -c "import onnxruntime as o,numpy as n; p=r'models\modnet_photographic.onnx'; s=o.InferenceSession(p,providers=['CPUExecutionProvider']); i=s.get_inputs()[0]; print('Model input:',i.shape); x=n.zeros((1,3,512,512),dtype=n.float32); y=s.run(None,{i.name:x}); assert len(y)>0 and n.isfinite(y[0]).all(); print('MODNet inference: OK'); print('Output:',y[0].shape)"

if errorlevel 1 (
    echo.
    echo ERROR: MODNet inference test failed.
    pause
    exit /b 1
)

echo.
echo [6/6] Building Windows 7 application...

"%PY%" -c "import shutil; shutil.rmtree(r'dist\Abasky-IQ-Win7',ignore_errors=True); shutil.rmtree(r'build\Abasky-IQ-Win7',ignore_errors=True)"

"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed --name Abasky-IQ-Win7 --icon iq.ico --collect-data cv2 --collect-all onnxruntime --hidden-import iq_ai --exclude-module tkinter main.py

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo Copying AI models...

if not exist "%DIST%\models" mkdir "%DIST%\models"

xcopy /E /I /Y models "%DIST%\models" >nul

if errorlevel 1 (
    echo ERROR: Could not copy models.
    pause
    exit /b 1
)

copy /Y get_models.py "%DIST%\" >nul
copy /Y get_models.bat "%DIST%\" >nul

if not exist "%DIST%\Abasky-IQ-Win7.exe" (
    echo ERROR: EXE file was not created.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo BUILD COMPLETED
echo ==========================================
echo Output: %DIST%
echo Executable: %DIST%\Abasky-IQ-Win7.exe
echo.
echo IMPORTANT:
echo Test on a real Windows 7 PC.
echo Verify MODNet availability inside the application.
echo Installer script: installer-win7.iss
echo ==========================================

pause
exit /b 0

:PIP_ERROR
echo.
echo ERROR: Dependency installation or validation failed.
echo Check the error messages above.
pause
exit /b 1
