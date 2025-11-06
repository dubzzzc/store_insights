@echo off
REM Check and install ODBC Driver 17 for SQL Server
echo Checking for ODBC Driver 17 for SQL Server...

REM Check if driver is already installed
powershell -Command "Get-OdbcDriver | Where-Object {$_.Name -like '*SQL Server*17*'}" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ODBC Driver 17 for SQL Server is already installed.
    pause
    exit /b 0
)

echo ODBC Driver not found. Starting installation...
echo.

REM Determine if system is 64-bit or 32-bit
if "%PROCESSOR_ARCHITECTURE%" == "AMD64" (
    set ARCH=x64
    set URL=https://go.microsoft.com/fwlink/?linkid=2249004
) else (
    set ARCH=x86
    set URL=https://go.microsoft.com/fwlink/?linkid=2249003
)

echo Downloading ODBC Driver 17 for SQL Server (%ARCH%)...
echo URL: %URL%
echo.

REM Download the installer
powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%TEMP%\msodbcsql.msi'"

if not exist "%TEMP%\msodbcsql.msi" (
    echo ERROR: Failed to download ODBC driver installer.
    echo Please download manually from: %URL%
    pause
    exit /b 1
)

echo Installing ODBC Driver...
echo This may require administrator privileges.
echo.

REM Install silently (requires admin)
msiexec /i "%TEMP%\msodbcsql.msi" /quiet /norestart IACCEPTMSODBCSQLLICENSETERMS=YES

if %ERRORLEVEL% == 0 (
    echo.
    echo ODBC Driver 17 for SQL Server installed successfully!
    echo Please restart this application if it's running.
) else (
    echo.
    echo Installation may have failed or requires administrator privileges.
    echo Please run this script as Administrator or install manually.
)

REM Cleanup
del "%TEMP%\msodbcsql.msi" >nul 2>&1

pause

