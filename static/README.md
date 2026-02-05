# Test GUI

This directory contains a simple web-based test GUI for the Qwen3-TTS API.

## Usage

Simply open the [`test_gui.html`](test_gui.html) file in your web browser:

```bash
# On macOS
open test_gui.html

# On Linux
xdg-open test_gui.html

# On Windows
start test_gui.html
```

Or serve it via the API server by adding static file serving to `api/main.py`.

## Features

The test GUI provides:

1. **API Configuration** - Set the API base URL and check health
2. **Preset Speaker Mode** - Test the 9 built-in speakers
3. **Voice Clone Mode** - Upload audio and clone voices instantly
4. **Saved Voice Mode** - Use previously saved voice IDs
5. **Save Voice** - Permanently save cloned voices for reuse

## All Parameters Available

- Model selection
- Text input with pause tag support
- Speaker/Voice selection
- Audio file upload
- Sample text (for better cloning quality)
- Style/Emotion (happy, sad, whisper, etc.)
- Speed control (0.5x - 2.0x)
- Output format (MP3, WAV, OGG, OPUS)
- Temperature, Top P (advanced generation controls)

## Screenshot

The GUI features a modern, responsive design with:
- Tab-based navigation between modes
- Real-time status updates
- Audio playback and download
- Form validation
