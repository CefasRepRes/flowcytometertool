# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['flow_cytometer_tool.py'],
    pathex=[],
    binaries=[],	
    datas=[('readme.md', '.'),('expertise_matrix.csv', '.')],
    hiddenimports=['functions', 'custom_functions_for_python', 'listmode','sys','requests'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='flow_cytometer_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons8-laser-96.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='flow_cytometer_tool',
)