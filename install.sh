#!/bin/bash
# Installation script for Qwen3-TTS API

set -e

echo "=========================================="
echo "Qwen3-TTS API Installation"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

# Check if Python version is supported by PyTorch
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo ""
    echo "ERROR: Python $PYTHON_VERSION is not supported by PyTorch yet."
    echo "Please use Python 3.10, 3.11, or 3.12 instead."
    echo ""
    echo "To install a different Python version:"
    echo "  - Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "  - Then recreate venv: python3.10 -m venv venv"
    echo ""
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    USE_GPU=true
else
    echo "No NVIDIA GPU detected, using CPU"
    USE_GPU=false
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA support
echo ""
echo "Installing PyTorch..."
if [ "$USE_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install transformers and other core ML packages
echo ""
echo "Installing ML dependencies..."
pip install transformers accelerate

# Install audio processing
echo ""
echo "Installing audio processing libraries..."
pip install soundfile librosa ffmpeg-python pydub

# Install API framework
echo ""
echo "Installing API framework..."
pip install fastapi uvicorn[standard] python-multipart
pip install pydantic pydantic-settings python-dotenv

# Install model download utilities
echo ""
echo "Installing model utilities..."
pip install huggingface-hub modelscope

# Install utilities
echo ""
echo "Installing utilities..."
pip install numpy aiofiles

# Try to install onnxruntime (required by qwen-tts)
echo ""
echo "Installing onnxruntime..."
if [ "$USE_GPU" = true ]; then
    pip install onnxruntime-gpu || pip install onnxruntime || echo "WARNING: onnxruntime installation failed"
else
    pip install onnxruntime || echo "WARNING: onnxruntime installation failed"
fi

# Install qwen-tts without dependencies (to avoid conflicts)
echo ""
echo "Installing qwen-tts..."
pip install qwen-tts --no-deps || echo "WARNING: qwen-tts installation failed"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then open static/test_gui.html in your browser"
echo ""
