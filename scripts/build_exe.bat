@echo off
REM Build executable for vfp_dbf_to_rdsv2.py
REM This script creates a Windows executable with all dependencies

REM Change to the directory where this script is located
cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Installing/upgrading PyInstaller and required dependencies...
pip install --upgrade pip
pip install --upgrade pyinstaller
pip install -r requirements-uploader.txt

echo.
echo Building executable...
echo Script location: %CD%
echo.

REM Check if the Python script exists
if not exist "vfp_dbf_to_rdsv2.py" (
    echo ERROR: vfp_dbf_to_rdsv2.py not found in current directory!
    echo Please run this script from the scripts folder.
    pause
    exit /b 1
)

REM Generate spec file with correct MySQL plugin paths
python gen_spec.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Spec file verification failed.
    pause
    exit /b 1
)

REM Use spec file for more precise control
if not exist "vfp_dbf_to_rdsv2.spec" (
    echo ERROR: vfp_dbf_to_rdsv2.spec was not found even after verification.
    pause
    exit /b 1
)

echo Using checked-in spec file...
pyinstaller --clean --noconfirm vfp_dbf_to_rdsv2.spec
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build complete! Executable is in: %CD%\dist\VFP_DBF_Uploader\VFP_DBF_Uploader.exe
    echo.
    echo IMPORTANT: Users must install ODBC Driver 17 for SQL Server separately.
    echo See README_ODBC.txt for instructions.
    echo.
    echo USAGE NOTES:
    echo   - Double-click to launch GUI (default behavior)
    echo   - For Task Scheduler: Use --auto-sync --silent
    echo   - For headless mode: Use --headless --config path/to/config.yaml
    echo   - See BUILD_INSTRUCTIONS.md for details
) else (
    echo.
    echo ERROR: Build failed! Check the error messages above.
)

pause

