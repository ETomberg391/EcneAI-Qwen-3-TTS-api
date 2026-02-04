# Qwen3-TTS API

OpenAI-compatible API service for Qwen3-TTS voice cloning and text-to-speech synthesis.

## Features

- 🎯 **OpenAI Compatible**: Drop-in replacement for `/v1/audio/speech` endpoint
- 🎭 **Voice Cloning**: Clone any voice from 3+ seconds of audio
- 🎙️ **Preset Speakers**: 9 built-in voices (Vivian, Ryan, Aiden, etc.)
- 🚀 **Multiple Formats**: MP3, WAV, OGG, OPUS output
- ⚡ **Speed Control**: 0.5x to 2.0x playback speed
- 🎨 **Emotion Control**: Happy, sad, whisper, angry, and more
- 💾 **Voice Persistence**: Save and reuse cloned voices
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 📦 **Auto-Download**: Automatic model download from HuggingFace/ModelScope

## Quick Start

### Installation

```bash
# Clone or navigate to the workflow directory
cd workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (optional - they auto-download on first use)
python scripts/download_models.py --model qwen3-tts-1.7b-base
```

### Run the API Server

```bash
# Copy environment file
cp .env.example .env

# Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or use the module directly
python api/main.py
```

The API will be available at `http://localhost:8000`

View interactive documentation at `http://localhost:8000/docs`

## API Usage

### Using Preset Speakers

```bash
# List available preset speakers
curl http://localhost:8000/v1/speakers

# Generate speech with a preset speaker
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts-1.7b-customvoice" \
  -F "input=Hello, world! This is Vivian speaking." \
  -F "voice=Vivian" \
  -F "instructions=happy" \
  -F "response_format=mp3" \
  --output speech.mp3
```

Available preset speakers:
- `Vivian` - Chinese - Bright, Sharp, Young Female
- `Serena` - Chinese - Warm, Soft, Young Female
- `Uncle_Fu` - Chinese - Deep, Mellow, Mature Male
- `Dylan` - Chinese Beijing - Clear, Natural Young Male
- `Eric` - Chinese Sichuan - Lively, Husky Male
- `Ryan` - English - Rhythmic, Dynamic Male
- `Aiden` - English - Sunny, Clear American Male
- `Ono_Anna` - Japanese - Light, Playful Female
- `Sohee` - Korean - Emotional, Warm Female

### Instant Voice Cloning

```bash
# Clone a voice and generate speech immediately
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "model=qwen3-tts-1.7b-base" \
  -F "input=This is my cloned voice speaking! It's amazing!" \
  -F "voice_sample=@path/to/your/voice_sample.wav" \
  -F "voice_sample_text=This is the original text spoken in the sample." \
  -F "instructions=excited" \
  -F "speed=1.0" \
  -F "response_format=mp3" \
  --output cloned_voice.mp3
```

### Save and Reuse a Voice

```bash
# Step 1: Clone and save the voice
curl -X POST http://localhost:8000/v1/voices \
  -F "name=My Cloned Voice" \
  -F "description=A warm, friendly voice" \
  -F "voice_sample=@sample.wav" \
  -F "voice_sample_text=Text in the sample"

# Response: {"voice_id": "voice_abc123", "name": "My Cloned Voice", ...}

# Step 2: Use the saved voice
curl -X POST http://localhost:8000/v1/audio/speech \
  -F "voice=voice_abc123" \
  -F "input=Hello from my saved voice!" \
  -F "response_format=mp3" \
  --output speech.mp3
```

### JSON API (for saved voices)

```bash
curl -X POST http://localhost:8000/v1/audio/speech/json \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-1.7b-base",
    "input": "Hello from my saved voice!",
    "voice": "voice_abc123",
    "response_format": "mp3",
    "speed": 1.0,
    "instructions": "happy",
    "temperature": 0.9
  }' \
  --output speech.mp3
```

## Voice Management

### List Saved Voices

```bash
curl http://localhost:8000/v1/voices
```

### Get Voice Details

```bash
curl http://localhost:8000/v1/voices/{voice_id}
```

### Delete a Voice

```bash
curl -X DELETE http://localhost:8000/v1/voices/{voice_id}
```

## Emotion/Style Instructions

You can use emotion keywords or custom instructions:

**Built-in emotions:**
- `happy` / `开心` - Cheerful, energetic tone
- `sad` / `难过` - Low, sorrowful tone
- `angry` / `生气` - Stern, aggressive tone
- `whisper` / `低语` - Soft whisper
- `excited` / `激动` - Fast, excited tone
- `gentle` / `温柔` - Soft, warm tone
- `fearful` / `恐惧` - Trembling, scared tone
- `surprised` / `惊讶` - Shocked tone
- `neutral` / `平静` - Natural, steady tone

**Custom instructions:**
```bash
-F "instructions=Speak like a professional news anchor with a calm demeanor"
```

## Pause Control

Use `[pause:X]` tags to add pauses in your text:

```bash
-F "input=Hello! [pause:0.5] How are you? [pause:1.0] I'm doing great!"
```

## Models

Available models:

| Model | Size | Use Case |
|-------|------|----------|
| `qwen3-tts-1.7b-base` | 3.4GB | Voice cloning, dialogue (recommended) |
| `qwen3-tts-1.7b-customvoice` | 3.4GB | Preset speakers |
| `qwen3-tts-1.7b-voicedesign` | 3.4GB | Voice design from text |
| `qwen3-tts-0.6b-base` | 1.2GB | Lightweight base model |
| `qwen3-tts-0.6b-customvoice` | 1.2GB | Lightweight preset speakers |

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
cd workflow/docker

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker directly

```bash
# Build
docker build -f docker/Dockerfile -t qwen3-tts-api .

# Run with GPU support
docker run -d \
  --name qwen3-tts-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/voices:/app/voices \
  qwen3-tts-api
```

## Configuration

All settings can be configured via environment variables or `.env` file:

```bash
# Device settings
DEVICE=auto              # auto, cuda, cpu, mps
PRECISION=bf16           # bf16, fp16, fp32
ATTN_MODE=sdpa           # sdpa, flash_attention_2, eager

# Model settings
MODEL_CACHE_DIR=./models
ENABLE_AUTO_DOWNLOAD=true
DEFAULT_MODEL=qwen3-tts-1.7b-base

# API settings
API_HOST=0.0.0.0
API_PORT=8000
MAX_AUDIO_FILE_SIZE=52428800

# HuggingFace token (optional)
HF_TOKEN=your_token_here
```

## Model Download

Models are automatically downloaded on first use. To pre-download:

```bash
# Download all models
python scripts/download_models.py --model all

# Download specific model
python scripts/download_models.py --model qwen3-tts-1.7b-base

# Use ModelScope (for users in China)
python scripts/download_models.py --source modelscope

# List available models
python scripts/download_models.py --list
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/audio/speech` | POST | Generate speech |
| `/v1/audio/speech/json` | POST | Generate speech (JSON body) |
| `/v1/speakers` | GET | List preset speakers |
| `/v1/voices` | GET | List saved voices |
| `/v1/voices` | POST | Clone a new voice |
| `/v1/voices/{id}` | GET | Get voice details |
| `/v1/voices/{id}` | DELETE | Delete a voice |
| `/v1/voices/{id}` | PATCH | Update voice metadata |

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- 10GB+ disk space for models

## Performance Tips

1. **Use GPU**: Significantly faster inference on CUDA
2. **Flash Attention**: Install `flash-attn` for 2-3x speedup
3. **Model Caching**: Models stay loaded in memory between requests
4. **Pre-clone Voices**: Save voice prompts for faster generation
5. **Batch Processing**: Generate multiple segments at once

## License

Apache License 2.0

## Acknowledgments

- Based on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- OpenAI-compatible API design
