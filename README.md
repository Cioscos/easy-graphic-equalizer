# Sound Wave 🎵

A real-time desktop audio graphic equalizer with a modern control panel and a high-performance, borderless-fullscreen OpenGL visualization.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

> 🇮🇹 **Italiano**: [Leggi questo README in italiano](README_IT.md)

## 📋 Overview

Sound Wave captures your system audio in real time, runs an FFT analysis and renders animated frequency visualizations. It is built as two cooperating parts:

- A **PySide6 control panel** where you pick the audio device and configure every aspect of the look.
- A single **borderless-windowed GLFW/OpenGL renderer** that covers the chosen monitor. It is *not* exclusive fullscreen — alt-tab is instant and there is no video-mode switch. The renderer can draw an image or a looping (muted) video behind the bars.

Configurations can be saved as **named profiles** and shared as files, so you never have to set everything up from scratch again.

### ✨ Key Features

- **🎤 Multi-device audio capture** — captures system audio (44.1 kHz, stereo) from any available device.
- **📊 Four visualization modes** — **Bars**, **Radial**, **Oscilloscope**, **Line/Area**.
- **🎨 Four color modes** — **Classic** (green/yellow/red), **Solid**, **Gradient**, **Spectrum** (HSV across the frequency axis).
- **🎛️ Shape controls** — bar width, rounded caps, center-mirror anchor, and a symmetric "bass-centered" band order.
- **✨ Reactive effects** — per-bar **peak-cap**, **bloom/glow**, and a **beat pulse** that zooms the background on detected beats.
- **🖼️ Backgrounds** — custom **image** *or* looping **video** background (purely decorative; no audio is decoded from the video).
- **💾 Profile manager** — save/load named profiles, **import/export** them as `.json`, auto-reload the last used profile on startup, with a **safe fallback** when a referenced background file is missing.
- **🔬 Advanced DSP** — absolute dB scale with adjustable floor/ceiling, attack/release ballistics, and a spectral tilt control.
- **🌗 Theming** — light / dark / system, switchable live.
- **🖥️ Multi-monitor** — pick which display the renderer covers.

## 🖼️ Screenshots

### Control Panel
The PySide6 control panel: device selector and profile bar on top, a navigation sidebar (Visualization / Shape / Color / Audio / Effects) driving the settings pages.

### Fullscreen Renderer
Immersive monitor-covering OpenGL visualization with VSync, MSAA 4× antialiasing and instanced rendering.

## 🛠️ Installation

### System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Windows, macOS, Linux
- **Hardware**: an audio device, and a GPU/driver capable of OpenGL 3.3 core
- **Linux runtime extras**: PulseAudio/PipeWire (for `soundcard`) and a GL-capable display; `libxcb-cursor0` for the Qt xcb platform plugin

### Dependencies

This project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + the pinned `uv.lock`, `requires-python = ">=3.12"`). There is **no** `requirements.txt`.

```bash
uv sync          # create/refresh the virtual env from the lockfile
```

Main dependencies:
- `PySide6` — the control-panel GUI (Qt for Python, LGPL)
- `pyqtdarktheme-fork` (imported as `qdarktheme`) — light/dark/system theming
- `soundcard` — cross-platform system-audio capture
- `numpy` — FFT and vectorized DSP
- `PyOpenGL` + `PyOpenGL-accelerate` — OpenGL rendering (must resolve to the **same** version)
- `glfw` — OpenGL window/context management
- `Pillow` — image handling
- `imageio` + `imageio-ffmpeg` — video-background decoding (bundles a static `ffmpeg`)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Cioscos/easy-graphic-equalizer.git
   cd easy-graphic-equalizer
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run the application**:
   ```bash
   uv run python main.py
   ```

## 🚀 Usage

### Getting Started

1. **Select an audio device** from the list at the top. The action button at the bottom turns green ("Start fullscreen") once a device is selected.
2. **Configure the look** using the settings pages (see below) — changes apply live while the renderer is running.
3. **Start the renderer** with the bottom button. It opens a borderless window covering the selected monitor and the button turns red ("Stop").
4. **Exit the renderer** by pressing **ESC** (or stop it from the control panel).

### Settings Pages

The navigation sidebar switches between five pages:

- **🖥️ Visualization** — visualization mode, background (image/video) + transparency, bars opacity, target monitor.
- **🎛️ Shape** — bar width, rounded caps, anchor, band order; plus mode-specific shape controls (radial hole radius, line thickness, fill, oscilloscope mirror/smoothing).
- **🎨 Color** — color mode and its colors/thresholds (shown conditionally per mode).
- **🎵 Audio** — noise threshold, number of frequency bands, and an **Advanced (DSP)** group (dB floor/ceiling, attack, release, spectral tilt).
- **✨ Effects** — peak-cap, glow (bloom), and beat-pulse (all default **off**).

### ⌨️ Keyboard Shortcuts

- **ESC** — exit the fullscreen renderer.

## 💾 Profiles (import / export)

A **profile** captures the full visual/DSP/audio/background configuration (it deliberately excludes machine-specific things like the audio device, the monitor and the app theme).

The **profile bar** at the top of the control panel lets you:

- **Select** a saved profile from the dropdown to apply it instantly (live if the renderer is running).
- **Save** — overwrite the selected profile with the current settings.
- **Save as…** — store the current settings under a new name.
- **Delete** — remove the selected profile.
- **Import…** — load a `.json` profile from anywhere (names are de-duplicated automatically).
- **Export…** — write a profile to a `.json` file to back it up or share it.

The last selected profile is **reloaded automatically** on the next launch.

### Storage & safe fallback

Profiles are stored as one JSON file each under a per-user config directory:

| OS | Location |
|----|----------|
| Windows | `%APPDATA%\sound-wave\profiles\` |
| macOS | `~/Library/Application Support/sound-wave/profiles/` |
| Linux | `$XDG_CONFIG_HOME/sound-wave/profiles/` (or `~/.config/sound-wave/profiles/`) |

Background references are **portable**: a background that ships with the app is stored as a resource reference (so it resolves on any machine and in a packaged build), while an external file is stored as an absolute path. If a referenced background **can't be found** when a profile is applied, Sound Wave shows a non-blocking warning and falls back to the bundled default image — it never crashes on a missing file.

## 🎛️ Visualization Modes

| Mode | Description |
|------|-------------|
| **Bars** | Classic instanced frequency bars (supports peak-cap, rounded caps, center anchor). |
| **Radial** | The bars wrapped into a polar ring with an adjustable inner radius. |
| **Oscilloscope** | The (channel-averaged) waveform, zero-crossing trigger-aligned, with optional mirroring and temporal smoothing. |
| **Line/Area** | A continuous line spectrum, optionally filled into an area with a vertical alpha fade. |

The symmetric "bass-centered" band order applies to Bars / Radial / Line. Peak-cap is Bars-only.

## ✨ Reactive Effects (default off)

- **Peak-cap** — a falling cap that marks each bar's recent peak (configurable color, hold and fall rate).
- **Glow (bloom)** — a half-resolution gaussian bloom composited additively over the visualization (works in every mode).
- **Beat pulse** — detects beats from low-frequency FFT energy and drives a background-zoom envelope.

## 🔬 Technical Details

### Audio Processing

- **Sample rate**: 44.1 kHz, **stereo** (2 channels), captured in 1024-frame chunks.
- **FFT size**: 4096 samples, recomputed **every frame** on the freshest window from a ring buffer.
- **Window function**: Blackman (reduces spectral leakage).
- **Single-pass vectorized FFT**: one `numpy.rfft` over both channels at once; bins are aggregated into bands via a precomputed `bin→band` map. There is **no** multiprocessing — a single 4096-point rFFT is ~0.1 ms, so a per-frame serial FFT is cheaper than IPC.
- **Absolute dB scale**: per-band power → dB → mapped from a fixed `[DB_FLOOR, DB_CEILING]` range to `[0, 1]` (bar height reflects absolute loudness, not a per-frame max).
- **Ballistics**: frame-rate-independent **attack/release** smoothing (fast rise, slow fall) toward the FFT targets.

### Rendering

- **OpenGL 3.3 core profile** with shaders, **MSAA 4×** antialiasing.
- **Borderless windowed** output sized to the selected monitor (`GLFW_DECORATED=FALSE` + window positioning) — instant alt-tab, no mode switch.
- **VSync-locked** to the monitor refresh rate.
- Bars use **instanced rendering** (a static unit-quad VBO + a per-instance heights VBO); changing the band count only resizes the heights buffer.
- The background is a textured fullscreen quad whose alpha is a shader uniform; **video** frames are streamed via 2 ping-pong **PBOs** (async DMA upload) and decoded on a dedicated thread, paced to the video's real frame rate.

## 📁 Project Structure

```
easy-graphic-equalizer/
├── main.py                          # Entry point (builds the QApplication)
├── pyproject.toml                   # Project metadata and dependencies (uv)
├── uv.lock                          # Dependency lockfile (uv)
├── sound-wave.spec                  # Cross-platform PyInstaller build spec
├── profiles.py                      # Profile model + ProfileStore (persistence, import/export)
├── resource_manager.py              # Resource path resolution (frozen-aware)
│
├── gui/                             # PySide6 control panel
│   ├── main_window_gui.py           # Main window, threads/queues, collect/apply profile
│   ├── profile_bar_frame.py         # Profile selector + actions bar
│   ├── animated_button.py           # Animated action button
│   ├── color_picker_frame.py        # Color picker + presets
│   ├── background_filepicker_frame.py # Image/video file picker
│   ├── slider_frame.py              # Custom slider component
│   ├── optionmenu_frame.py          # Dropdown component
│   ├── help_window.py               # Help dialog
│   └── theme.py                     # Style tokens, fonts, qdarktheme layer
│
├── thread/                          # Producer/consumer pipeline
│   ├── audioCaptureThread.py        # Audio capture producer
│   ├── opengl_thread.py             # OpenGL renderer (single consumer)
│   ├── video_decode_thread.py       # Optional video-decode producer
│   ├── AudioBufferAccumulator.py    # Ring buffer for FFT windows
│   └── frame_time_stats.py          # Frame-time statistics (debug)
│
├── tests/                           # Pure-Python checks (no pytest)
│   ├── test_profiles.py             # Profile serialization/store tests
│   └── test_frame_time_stats.py     # Frame-time stats tests
│
└── resources/                       # Application assets
    ├── bg/                          # Background images
    └── icons/                       # Interface icons
```

## 📦 Building a Release

The app is packaged with **PyInstaller** using the single cross-platform spec:

```bash
uv run --with pyinstaller pyinstaller --noconfirm sound-wave.spec
# output: dist/sound wave/   (one-dir bundle)
```

The GitHub Actions workflow `.github/workflows/release-build.yml` builds the bundle on **Windows** and **Linux** and attaches `sound-wave-windows.zip` / `sound-wave-linux.zip` to a published release. PyInstaller doesn't cross-compile, hence one runner per OS. The `resources/` folder is bundled into the build, and `ResourceManager` resolves asset paths from `sys._MEIPASS` when frozen.

## 🐛 Troubleshooting

#### The app doesn't detect audio devices
- Verify your audio drivers are up to date.
- Check microphone/recording permissions (macOS/Linux).
- On Linux, ensure PulseAudio/PipeWire is running.

#### The renderer won't start / shows nothing
- Verify your GPU drivers are up to date and support OpenGL 3.3 core.
- On Linux, ensure GL libraries and a GL-capable display are available.

#### Performance issues with many bands
- Reduce the number of frequency bands.
- Disable heavy effects (bloom) if your GPU is modest.

#### A background image/video won't load
- Verify the format (images: PNG/JPG/JPEG/GIF; video: MP4/AVI/MOV/MKV).
- Check file read permissions and that the path still exists.
- For profiles, a missing background falls back to the default with a warning.

### Logging and Debug

Enable detailed logging via the module-level flags at the top of `thread/opengl_thread.py`:

```python
GENERAL_LOG = True   # general logs
FFT_LOGS = True      # FFT processing logs
PERF_LOGS = True     # aggregated frame-time summaries
```

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. Create a **branch** for your feature (`git checkout -b feature/new-feature`)
3. **Commit** your changes
4. **Push** to the branch
5. Open a **Pull Request**

### Guidelines

- Follow the existing code style and patterns.
- Add docstrings for new functions/classes.
- Keep the two renderers' DSP aligned if you touch the FFT/frequency-band logic.
- Update documentation when behavior changes.

## 📄 License

Released under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- **PySide6 / Qt** — control-panel GUI
- **pyqtdarktheme-fork** — theming
- **SoundCard** — cross-platform audio capture
- **NumPy** — FFT and DSP
- **PyOpenGL / GLFW** — high-performance rendering
- **imageio / FFmpeg** — video-background decoding

---

**Developed with ❤️**
