#!/bin/bash
# Fixed Installation script for Qwen3-TTS API
set -e

echo "=========================================="
echo "Qwen3-TTS API Installation"
echo "=========================================="

# 1. Environment Check
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python version: $PYTHON_VERSION"

# 2. GPU Detection
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Preparing for CUDA..."
    USE_GPU=true
else
    echo "No NVIDIA GPU detected, defaulting to CPU-only."
    USE_GPU=false
fi

# 3. Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Clean Slate (Crucial to fix your current conflicts)
echo "Cleaning up conflicting packages..."
pip uninstall -y transformers numpy qwen-tts onnxruntime 2>/dev/null || true

# 5. Base Upgrades
echo "Upgrading pip/setuptools..."
pip install --upgrade pip setuptools wheel

# 6. PyTorch Installation (Optimized for Hardware)
echo "Installing PyTorch..."
if [ "$USE_GPU" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 7. Main Installation
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found! Please create it first."
    exit 1
fi

# 8. Post-Install Fix for ONNX GPU
if [ "$USE_GPU" = true ]; then
    echo "Swapping to onnxruntime-gpu..."
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu
fi

echo ""
echo "=========================================="
echo "SUCCESS: Environment is ready."
echo "=========================================="
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"