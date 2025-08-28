# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'cv2', 'scipy', 'pip', 'Pyinstaller', '_pyinstaller_hooks_constrib', 'peutils', 'pkg_resources', 'win32ctypes', 'xml'],
    noarchive=False,
    optimize=2,
)

to_keep = []
to_exclude = {'Qt5dbus.dll', 'Qt5Network.dll', 'Qt5Qml.dll', 'Qt5Quick.dll', 'Qt5Svg.dll', 'Qt5WebSockets.dll', 'opengl32sw.dll', 'd3dcompiler_47.dll', 'Qt5Qml.dll', 'Qt5QmlModels.dll', 'libGLESv2.dll', 'libEGL.dll', 'Qt5DBus.dll', 'libcrypto-3.dll'}

# Iterate through the list of included binaries.
for (dest, source, kind) in a.binaries:
    # Skip anything we don't need.
    if os.path.split(dest)[1] in to_exclude:
        continue
    to_keep.append((dest, source, kind))

# Replace list of data files with filtered one.
a.binaries = to_keep

# Remove Qt translation files
a.datas = [d for d in a.datas if 'translations' not in d[0].lower()]


pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    name='ImageAnalyzerAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir='.\\data\TEMP_DIR',
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
