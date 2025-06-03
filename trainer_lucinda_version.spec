# Corrected PyInstaller .spec file for 'trainer_lucinda_version.py'

import os
import tkinter

# Dynamically locate Tcl/Tk paths
tkinter_root = os.path.dirname(tkinter.__file__)
tcl_dir = os.path.join(tkinter_root, 'tcl')
tk_dir = os.path.join(tkinter_root, 'tk')

# Define datas list
datas = [
    ('readme.md', '.'),
]

# Only add Tcl/Tk if they exist (safe for CI)
if os.path.exists(tcl_dir):
    datas.append((tcl_dir, 'tcl'))
if os.path.exists(tk_dir):
    datas.append((tk_dir, 'tk'))

a = Analysis(
    ['trainer_lucinda_version.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
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

