# -*- mode: python ; coding: utf-8 -*-
#
# Cross-platform PyInstaller spec for "Sound Wave" (one-dir build).
#
# Mirrors the historical auto-py-to-exe config (py-to-exe-settings.json) but is
# OS-agnostic: it is the single source of truth used by both the GitHub Actions
# release workflow (.github/workflows/release-build.yml) and manual local builds:
#
#     uv run --with pyinstaller pyinstaller --noconfirm sound-wave.spec
#
# Output: dist/sound wave/  (a one-dir bundle; the launcher is "sound wave[.exe]").
#
# Platform notes:
#  * PySide6 (Qt for Python) is collected by PyInstaller's built-in hook, which
#    bundles the needed Qt plugins (platforms, styles) automatically; do NOT
#    collect_all it (that would pull in every Qt module). qdarktheme ships the
#    flat-theme QSS/resources, so its whole package payload is collected.
#  * imageio_ffmpeg bundles a static ffmpeg binary used to decode video
#    backgrounds; collect_all pulls it in (imageio loads plugins lazily, hence
#    the explicit hidden imports).
#  * The `glfw` wheel bundles the GLFW shared library per-platform
#    (libglfw.so on Linux, glfw3.dll on Windows) -> collected for every OS.
#    On Windows we ALSO ship the repo's own glfw3.dll next to the launcher as a
#    fallback, matching the proven manual build.

import os
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = ['imageio', 'imageio_ffmpeg', 'imageio.plugins.ffmpeg']

# PyOpenGL picks its platform backend (WGL / GLX / EGL / OSMesa) at runtime via a
# plugin registry that uses dynamic imports PyInstaller's static analysis can't
# see, so the chosen backend module may be missing from the bundle — at import of
# OpenGL this surfaces as "No module named 'OpenGL.platform.egl'" followed by
# "TypeError: 'NoneType' object is not callable". Force-bundle the per-OS backends.
if sys.platform == 'win32':
    hiddenimports += ['OpenGL.platform.win32']
else:
    hiddenimports += ['OpenGL.platform.egl', 'OpenGL.platform.glx', 'OpenGL.platform.osmesa']

# Collect full package payloads (data files, bundled native libs, submodules).
for _pkg in ('qdarktheme', 'imageio_ffmpeg', 'glfw'):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# imageio reads its own version via importlib.metadata at import time, so its
# package metadata (.dist-info) must be bundled or it raises PackageNotFoundError
# at startup (OS-independent — affects every frozen build).
datas += copy_metadata('imageio')
datas += copy_metadata('imageio_ffmpeg')

# Project assets (resources/bg, resources/icons).
datas.append((os.path.join(SPECPATH, 'resources'), 'resources'))

# Windows ships its own glfw3.dll alongside the executable.
if sys.platform == 'win32':
    binaries.append((os.path.join(SPECPATH, 'glfw3.dll'), '.'))

a = Analysis(
    [os.path.join(SPECPATH, 'main.py')],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sound wave',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='sound wave',
)
