; NSIS Installer Script for VFP DBF Uploader
; Requires: NSIS 3.x and PyInstaller output
; Usage: Compile this script with NSIS after building the exe

!include "MUI2.nsh"

; Installer Information
Name "VFP DBF to RDS Uploader"
OutFile "VFP_DBF_Uploader_Setup.exe"
InstallDir "$PROGRAMFILES\VFP_DBF_Uploader"
RequestExecutionLevel admin

; Interface Settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "README_ODBC.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installer Sections
Section "Application" SecApp
    SectionIn RO
    SetOutPath "$INSTDIR"
    File "dist\VFP_DBF_Uploader.exe"
    File "README_ODBC.txt"
    File "install_odbc.bat"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\VFP DBF Uploader"
    CreateShortcut "$SMPROGRAMS\VFP DBF Uploader\VFP DBF Uploader.lnk" "$INSTDIR\VFP_DBF_Uploader.exe"
    CreateShortcut "$SMPROGRAMS\VFP DBF Uploader\Install ODBC Driver.lnk" "$INSTDIR\install_odbc.bat"
    CreateShortcut "$SMPROGRAMS\VFP DBF Uploader\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
    CreateShortcut "$DESKTOP\VFP DBF Uploader.lnk" "$INSTDIR\VFP_DBF_Uploader.exe"
    
    ; Write uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    ; Add to registry for Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VFP_DBF_Uploader" \
        "DisplayName" "VFP DBF to RDS Uploader"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VFP_DBF_Uploader" \
        "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VFP_DBF_Uploader" \
        "Publisher" "Store Insights"
SectionEnd

Section "ODBC Driver 17 for SQL Server" SecODBC
    ; Check if already installed
    ExecWait 'powershell -Command "Get-OdbcDriver | Where-Object {$_.Name -like ''*SQL Server*17*''}"' $0
    IntCmp $0 0 skip_install
    
    ; Download and install ODBC driver
    DetailPrint "Downloading ODBC Driver 17 for SQL Server..."
    ExecWait 'powershell -Command "Invoke-WebRequest -Uri ''https://go.microsoft.com/fwlink/?linkid=2249004'' -OutFile ''$TEMP\msodbcsql.msi''' $0
    
    DetailPrint "Installing ODBC Driver..."
    ExecWait 'msiexec /i "$TEMP\msodbcsql.msi" /quiet /norestart IACCEPTMSODBCSQLLICENSETERMS=YES' $0
    
    skip_install:
SectionEnd

; Uninstaller
Section "Uninstall"
    Delete "$INSTDIR\VFP_DBF_Uploader.exe"
    Delete "$INSTDIR\README_ODBC.txt"
    Delete "$INSTDIR\install_odbc.bat"
    Delete "$INSTDIR\Uninstall.exe"
    
    RMDir /r "$SMPROGRAMS\VFP DBF Uploader"
    Delete "$DESKTOP\VFP DBF Uploader.lnk"
    RMDir "$INSTDIR"
    
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VFP_DBF_Uploader"
SectionEnd

