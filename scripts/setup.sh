#!/bin/bash
# Setup script for Qwen3-TTS API

set -e

echo "=========================================="
echo "Qwen3-TTS API Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p models voices

# Copy environment file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env to configure your settings"
else
    echo ".env already exists, skipping"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "To download models (optional - they auto-download on first use):"
echo "  python scripts/download_models.py --model all"
echo ""
