# Models Directory

This directory stores downloaded TTS models.

Models are automatically downloaded from HuggingFace or ModelScope on first use.

## Manual Download

To pre-download models, use the provided script:

```bash
python scripts/download_models.py --model all
```

## Model Storage Structure

```
models/
├── Qwen--Qwen3-TTS-12Hz-1.7B-Base/     # Base model for voice cloning
├── Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/  # Preset speakers
├── Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/  # Voice design
└── iic--SenseVoiceSmall/               # ASR model (optional)
```

## Model Sizes

| Model | Size |
|-------|------|
| Qwen3-TTS-12Hz-1.7B-* | ~3.4 GB |
| Qwen3-TTS-12Hz-0.6B-* | ~1.2 GB |
| SenseVoiceSmall | ~1.0 GB |

## Note

Do not commit model files to git. They are excluded via `.gitignore`.
