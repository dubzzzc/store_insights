@echo off
REM Build executable for vfp_dbf_to_rdsv2.py
REM This script creates a Windows executable with all dependencies

REM Change to the directory where this script is located
cd /d "%~dp0"

echo Current directory: %CD%
echo.

echo Installing/upgrading PyInstaller...
pip install --upgrade pyinstaller

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
echo Generating spec file with correct paths...
python gen_spec.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Could not generate spec file. Using command line options.
    goto :build_cmd
)

REM Check if runtime hook exists
if not exist "mysql_plugin_hook.py" (
    echo WARNING: mysql_plugin_hook.py not found. Plugins may not work correctly.
)

REM Use spec file for more precise control
if exist "vfp_dbf_to_rdsv2.spec" (
    echo Using auto-generated spec file...
    pyinstaller vfp_dbf_to_rdsv2.spec --clean
    goto :build_done
)

:build_cmd
echo Building with command line options...
pyinstaller --name="VFP_DBF_Uploader" ^
        --onefile ^
        --windowed ^
        --icon=NONE ^
        --add-data "README_ODBC.txt;." ^
        --hidden-import yaml ^
        --hidden-import dbfread ^
        --hidden-import pyodbc ^
        --hidden-import mysql.connector ^
        --hidden-import mysql.connector.pooling ^
        --hidden-import mysql.connector.cursor ^
        --hidden-import mysql.connector.plugins ^
        --hidden-import mysql.connector.plugins.mysql_native_password ^
        --hidden-import mysql.connector.plugins.caching_sha2_password ^
        --hidden-import tkinter ^
        --hidden-import tkinter.ttk ^
        --hidden-import tkinter.filedialog ^
        --hidden-import tkinter.messagebox ^
        --collect-all dbfread ^
        --collect-all yaml ^
        --collect-all mysql.connector ^
        --collect-binaries mysql.connector ^
        vfp_dbf_to_rdsv2.py

:build_done

if %ERRORLEVEL% == 0 (
    echo.
    echo Build complete! Executable is in: %CD%\dist\VFP_DBF_Uploader.exe
    echo.
    echo IMPORTANT: Users must install ODBC Driver 17 for SQL Server separately.
    echo See README_ODBC.txt for instructions.
) else (
    echo.
    echo ERROR: Build failed! Check the error messages above.
)

pause

