ODBC Driver Installation for Windows
=====================================

This application requires the Microsoft ODBC Driver 17 for SQL Server to connect to SQL Server databases.

INSTALLATION INSTRUCTIONS FOR END USERS:
----------------------------------------

1. Download the ODBC Driver installer from Microsoft:
   https://go.microsoft.com/fwlink/?linkid=2249004

2. Run the installer (msodbcsql.msi) and follow the installation wizard.

3. For 64-bit Windows, install the 64-bit driver.
   For 32-bit Windows, install the 32-bit driver.

4. After installation, restart this application if it's running.

ALTERNATIVE: Automated Installation Script
------------------------------------------
You can also run the included install_odbc.bat script which will:
- Check if ODBC driver is already installed
- Download and install the driver automatically if missing
- Provide status feedback

QUICK CHECK:
------------
To verify ODBC driver is installed, run this command in PowerShell:
    Get-OdbcDriver | Where-Object {$_.Name -like "*SQL Server*"}

For MySQL connections, no additional driver is needed (uses mysql-connector-python).

