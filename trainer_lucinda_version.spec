# -*- mode: python ; coding: utf-8 -*-
import os
import tkinter

tk_lib = os.path.join(os.path.dirname(tkinter.__file__), 'tcl')
tk_dll = os.path.join(os.path.dirname(tkinter.__file__), 'tk')

a = Analysis(
    ['trainer_lucinda_version.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('readme.md', '.'),
        (tk_lib, 'tcl'),
        (tk_dll, 'tk'),
    ],
    hiddenimports=['functions', 'custom_functions_for_python', 'listmode'],
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
    exclude_binaries=False,  # safer for CI
    name='trainer_lucinda_version',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='trainer_lucinda_version',
)
