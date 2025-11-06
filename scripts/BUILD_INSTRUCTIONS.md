# Building Windows Executable for VFP DBF to RDS Uploader

## Prerequisites

1. **Python 3.8+** installed on Windows
2. Install the uploader dependencies:
   ```bash
   pip install -r requirements-uploader.txt
   ```
3. **PyInstaller** (installed automatically by `build_exe.bat` or manually with `pip install pyinstaller`)

## Quick Build (Recommended)

1. Open Command Prompt or PowerShell in the `scripts` folder
2. Run:
   ```bash
   build_exe.bat
   ```
3. The executable will be in `dist\VFP_DBF_Uploader\VFP_DBF_Uploader.exe`

## Manual Build with PyInstaller

If you prefer to build manually:

```bash
pip install -r requirements-uploader.txt
pip install pyinstaller

pyinstaller --clean --noconfirm vfp_dbf_to_rdsv2.spec
```

## Creating an Installer (Optional)

### Option 1: Using NSIS (Nullsoft Scriptable Install System)

1. Download and install NSIS from https://nsis.sourceforge.io/
2. Build the executable first using `build_exe.bat`
3. Compile `build_exe_with_installer.nsi` with NSIS
4. This creates an installer that:
   - Installs the application
   - Optionally installs ODBC Driver 17 for SQL Server
   - Creates shortcuts
   - Adds uninstaller

### Option 2: Using Inno Setup (Alternative)

1. Download Inno Setup from https://jrsoftware.org/isinfo.php
2. Use the Inno Setup Script Wizard or create a script similar to the NSIS one

## Including ODBC Driver

**Important**: ODBC drivers cannot be bundled directly into the executable. They must be installed on the target system.

### Solution 1: Include Installer Script (Recommended)
- Include `install_odbc.bat` with your distribution
- Users can run it to automatically download and install the ODBC driver
- Or provide download link in README

### Solution 2: Create Full Installer
- Use NSIS/Inno Setup to create an installer that:
  - Installs your application
  - Optionally downloads and installs ODBC driver
  - Provides all-in-one installation

### Solution 3: Portable Distribution
- Distribute the executable with:
  - `install_odbc.bat` (for ODBC driver installation)
  - `README_ODBC.txt` (instructions)
  - User manual/instructions

## Distribution Package Contents

Your distribution package should include:

```
VFP_DBF_Uploader_Release/
├── VFP_DBF_Uploader.exe          (Main executable)
├── install_odbc.bat              (ODBC driver installer script)
├── README_ODBC.txt               (ODBC installation instructions)
├── README.md                     (User manual)
└── LICENSE.txt                   (If applicable)
```

## Testing the Executable

1. Test on a clean Windows machine (VM recommended)
2. Verify:
   - Application launches
   - Database connections work (MySQL and SQL Server)
   - File operations work
   - GUI displays correctly

## Troubleshooting

### "mysql_native_password cannot be loaded" error
This means the MySQL connector plugins were not copied into the build output. Fix:
1. Ensure you installed the dependencies with `pip install -r requirements-uploader.txt` before running PyInstaller.
2. Rebuild using `build_exe.bat` or the provided `vfp_dbf_to_rdsv2.spec`, both of which bundle the MySQL plugin binaries automatically.
3. After rebuild, test the executable again.

### "ODBC Driver not found" error
- User needs to install ODBC Driver 17 for SQL Server
- Provide `install_odbc.bat` or link to Microsoft download

### "Module not found" errors
- Ensure all hidden imports are included in PyInstaller command
- Use `--collect-all` for packages with submodules

### Large executable size
- Normal - PyInstaller bundles Python interpreter and all dependencies
- Typical size: 50-100 MB

### Antivirus warnings
- Some antivirus software flags PyInstaller executables
- Consider code signing certificate for production distribution

