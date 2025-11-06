# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for vfp_dbf_to_rdsv2.py
# Auto-generated with correct MySQL plugin paths

block_cipher = None

a = Analysis(
    ['vfp_dbf_to_rdsv2.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README_ODBC.txt', '.'),
        # Include entire mysql/connector package structure to preserve imports
        (r'C:\Python312\Lib\site-packages\mysql', 'mysql'),
    ],
    hiddenimports=[
        'yaml',
        'dbfread',
        'pyodbc',
        'mysql.connector',
        'mysql.connector.pooling',
        'mysql.connector.cursor',
        'mysql.connector.plugins',
        'mysql.connector.plugins.mysql_native_password',
        'mysql.connector.plugins.caching_sha2_password',
        'mysql.connector.plugins.sha256_password',
        'mysql.connector.plugins.mysql_clear_password',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'pystray',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'win32event',
        'win32api',
        'win32gui',
        'win32con',
        'winerror',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['mysql_plugin_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    collect_all=['pystray', 'PIL', 'mysql.connector', 'dbfread', 'yaml'],
    collect_binaries=['mysql.connector', 'pywin32'],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VFP_DBF_Uploader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
