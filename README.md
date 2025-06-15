# Sound Wave 🎵

A real-time audio graphic equalizer with modern interface and high-performance fullscreen visualization.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macos-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

> 🇮🇹 **Italiano**: [Leggi questo README in italiano](README_IT.md)

## 📋 Overview

Sound Wave is a desktop application that captures real-time audio from system input devices and visualizes it as a dynamic graphic equalizer. The application offers both an integrated preview mode and a high-performance fullscreen mode for multiple monitors.

### ✨ Key Features

- **🎤 Multi-Device Audio Capture**: Supports all available audio devices on the system
- **📊 Real-Time Visualization**: Equalizer with colored bars (green, yellow, red)
- **🖥️ Fullscreen Mode**: High-performance visualization with multi-monitor support
- **⚙️ Advanced Configuration**: 
  - Adjustable noise threshold
  - Dynamic frequency bands (1-100)
  - Customizable background transparency
  - Configurable parallel processing
- **🎨 Visual Customization**:
  - Support for custom background images
  - Light/dark/automatic themes
  - Background alpha control
- **⚡ Performance Optimizations**:
  - Multi-core parallel FFT computation
  - OpenGL rendering for fullscreen mode
  - Adaptive smoothing based on monitor refresh rate

## 🖼️ Screenshots

### Main Interface
The main interface with controls on the left and equalizer preview on the right.

### Fullscreen Mode
Immersive full-screen visualization with high-performance OpenGL rendering.

## 🛠️ Installation

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Hardware**: Audio card with compatible drivers

### Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `customtkinter` - Modern user interface
- `soundcard` - Cross-platform audio capture
- `numpy` & `scipy` - Numerical processing and FFT
- `PyOpenGL` - High-performance graphics rendering
- `Pillow` - Image handling
- `glfw` - OpenGL window management

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sound-wave.git
   cd sound-wave
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## 🚀 Usage

### Initial Setup

1. **Select an Audio Device**: 
   - Double-click on a device from the list on the left
   - The "Start/Pause" button will turn green when ready

2. **Start Visualization**:
   - Click the "Start/Pause" button to begin
   - The button will turn red during playback

### Controls and Settings

#### 🎛️ Settings Panel

- **Appearance Mode**: Select light, dark, or automatic theme
- **Background Image Path**: Load a custom background image
- **Alpha Amount**: Control background transparency (0-1)
- **Noise Threshold**: Set minimum signal threshold for visualization
- **Frequency Bands**: Number of equalizer bars (1-100)
- **Number of Processors**: Configure parallel processing

#### 🖥️ Fullscreen Mode

- Click the "Fullscreen" button to open immersive visualization
- Use **ESC** to exit fullscreen mode
- On multi-monitor systems, select preferred display from settings

### ⌨️ Keyboard Shortcuts

- **ESC**: Exit fullscreen mode
- **Space**: Start/Stop audio capture (when application has focus)

## 🔧 Advanced Configuration

### Performance Optimization

#### Parallel Processing
The application supports parallel FFT processing to improve performance:

- **Auto**: Automatic calculation based on number of bands
- **Manual**: Manually set number of workers (1-5)

#### Hardware Recommendations

| Number of Bands | Recommended Workers | Notes |
|------------------|-------------------|-------|
| 1-15 | 1 | Single-thread processing |
| 16-30 | 2 | Optimal CPU balance |
| 31-50 | 3 | Intensive CPU usage |
| 51+ | 4-5 | Fullscreen mode recommended |

### Background Images

Supported formats: PNG, JPG, JPEG, GIF

For best results:
- Minimum resolution: 1920x1080
- Preferred format: PNG with alpha channel
- Avoid overly detailed images that might interfere with visualization

## 📁 Project Structure

```
sound-wave/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── resource_manager.py              # Resource management (images, assets)
│
├── gui/                            # User interface
│   ├── main_window_gui.py          # Main window
│   ├── help_window.py              # Help window
│   ├── slider_frame.py             # Custom slider component
│   ├── optionmenu_frame.py         # Options menu component
│   ├── background_filepicker_frame.py # Image file picker
│   └── processes_number_frame.py   # Processor count control
│
├── thread/                         # Processing threads
│   ├── audioCaptureThread.py       # Background audio capture
│   ├── equalizer_tkinter_thread.py # Equalizer rendering (Tkinter)
│   ├── opengl_thread.py           # Fullscreen rendering (OpenGL)
│   └── AudioBufferAccumulator.py  # Audio buffer with overlap
│
└── resources/                      # Application assets
    ├── bg/                        # Background images
    └── icons/                     # Interface icons
```

## 🔬 Technical Details

### Audio Processing

- **Sample Rate**: 44.1 kHz
- **Channels**: Stereo (2 channels)
- **FFT Size**: 4096 samples (fullscreen mode) / 8192 (preview mode)
- **Overlap**: 50% between successive windows
- **Window Function**: Blackman (to reduce spectral leakage)

### Visualization Algorithms

#### Frequency Distribution
Frequency bands are distributed on a logarithmic scale from 20 Hz to 20 kHz:

```python
def generate_frequency_bands(self, num_bands: int) -> list[tuple[float, float]]:
    """
    Generate frequency bands on logarithmic scale.
    
    Args:
        num_bands (int): Desired number of bands
        
    Returns:
        list[tuple[float, float]]: List of (freq_min, freq_max) tuples
    """
```

#### Adaptive Smoothing
The system implements adaptive temporal smoothing that automatically adjusts based on monitor refresh rate:

- **60 Hz**: Base factor 0.25
- **120 Hz**: Base factor 0.30  
- **144+ Hz**: Base factor 0.35

### Graphics Rendering

#### Tkinter Mode (Preview)
- Canvas-based rendering
- Frame rate: 120 FPS
- Colors: Green (0-50%), Yellow (50-80%), Red (80-100%)

#### OpenGL Mode (Fullscreen)
- Hardware-accelerated rendering
- VSync enabled for monitor synchronization
- Circular buffers for optimal performance
- Texture support for custom backgrounds

## 🐛 Troubleshooting

### Common Issues

#### Application doesn't detect audio devices
**Solution**: 
- Verify audio drivers are updated
- Check microphone permissions (macOS/Linux)
- Restart application as administrator (Windows)

#### Poor performance with many bands
**Solutions**:
- Reduce number of frequency bands
- Use fullscreen mode for better performance
- Increase number of workers in settings
- Close other CPU-intensive applications

#### Fullscreen mode won't start
**Checks**:
- Verify graphics drivers are updated
- Check system OpenGL support
- On Linux, ensure OpenGL libraries are installed

#### Background image won't load
**Solutions**:
- Verify image format is supported (PNG, JPG, JPEG, GIF)
- Check file read permissions
- Use absolute paths instead of relative

### Logging and Debug

To enable detailed logging, modify variables in `thread/opengl_thread.py`:

```python
GENERAL_LOG = True   # General logs
FFT_LOGS = True      # FFT processing logs  
PERF_LOGS = True     # Performance logs
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. Create a **branch** for your feature (`git checkout -b feature/new-feature`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/new-feature`)
5. Open a **Pull Request**

### Contribution Guidelines

- Follow existing code style
- Add docstrings for new functions/classes
- Test changes on multiple platforms when possible
- Update documentation if necessary

## 📄 License

This project is released under the MIT License. See the `LICENSE` file for complete details.

## 🙏 Acknowledgments

- **CustomTkinter** - For the modern user interface
- **SoundCard** - For cross-platform audio capture
- **PyOpenGL** - For high-performance rendering
- **GLFW** - For OpenGL window management

---

**Developed with ❤️**