# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for the VFP DBF → RDS uploader."""
import importlib
import pathlib
import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# Get project root - handle both direct execution and PyInstaller execution
# In PyInstaller's execution context, __file__ is not defined
# build_exe.bat runs from scripts/ directory, so cwd should be scripts/
try:
    # Try to use __file__ if available (when spec is executed directly)
    spec_dir = pathlib.Path(__file__).resolve().parent
except NameError:
    # Fallback for PyInstaller execution context
    # build_exe.bat sets cwd to scripts/, so use that
    cwd = pathlib.Path(os.getcwd()).resolve()
    # Check if we're in scripts/ directory (has vfp_dbf_to_rdsv2.py)
    if (cwd / "vfp_dbf_to_rdsv2.py").exists():
        spec_dir = cwd
    # Check if we're in project root (has scripts/ subdirectory)
    elif (cwd / "scripts" / "vfp_dbf_to_rdsv2.py").exists():
        spec_dir = cwd / "scripts"
    else:
        # Last resort: assume current directory is scripts/
        spec_dir = cwd

# Assume spec file is in scripts/ directory, so project_root is parent
project_root = spec_dir.parent
script_path = spec_dir / "vfp_dbf_to_rdsv2.py"

OPTIONAL_MODULES = [
    "pyodbc",
    "mysql.connector",
    "mysql.connector.plugins.mysql_native_password",
    "mysql.connector.plugins.caching_sha2_password",
    "mysql.connector.plugins.sha256_password",
    "mysql.connector.plugins.mysql_clear_password",
    "dearpygui.dearpygui",
    "pystray",
    "PIL.Image",
    "PIL.ImageDraw",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "win32api",
    "win32event",
    "win32gui",
    "win32con",
]

hiddenimports = []
for module_name in OPTIONAL_MODULES:
    try:
        importlib.import_module(module_name)
    except ImportError:
        continue
    hiddenimports.append(module_name)

binaries = []
datas = []

runtime_hooks = []

def extend_package(package: str) -> None:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except ImportError:
        return
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    for item in pkg_hidden:
        if item not in hiddenimports:
            hiddenimports.append(item)


for optional_package in ["mysql.connector", "pystray", "PIL", "dbfread", "yaml"]:
    extend_package(optional_package)

for optional_package in ["mysql.connector", "pywin32"]:
    try:
        binaries.extend(collect_dynamic_libs(optional_package))
    except ImportError:
        continue

# Explicitly ensure MySQL connector plugins are included
try:
    import mysql.connector
    mysql_connector_path = pathlib.Path(mysql.connector.__file__).parent
    plugins_path = mysql_connector_path / "plugins"
    if plugins_path.exists():
        # Add plugins directory recursively to datas to ensure all plugin files are included
        # PyInstaller will recursively include all files when using directory tuple format
        datas.append((str(plugins_path), "mysql/connector/plugins"))
        # Also add plugin modules as hidden imports to ensure they're importable
        plugin_modules = [
            'mysql.connector.plugins.mysql_native_password',
            'mysql.connector.plugins.caching_sha2_password',
            'mysql.connector.plugins.sha256_password',
            'mysql.connector.plugins.mysql_clear_password',
        ]
        for plugin_module in plugin_modules:
            if plugin_module not in hiddenimports:
                hiddenimports.append(plugin_module)
except ImportError:
    pass

runtime_hook_candidate = project_root / "scripts" / "mysql_plugin_hook.py"
if runtime_hook_candidate.exists():
    runtime_hooks.append(str(runtime_hook_candidate))

readme_path = project_root / "scripts" / "README_ODBC.txt"
if readme_path.exists():
    datas.append((str(readme_path), '.'))

block_cipher = None

a = Analysis(
    [str(script_path)],
    pathex=[str(project_root / "scripts"), str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VFP_DBF_Uploader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="VFP_DBF_Uploader",
)
