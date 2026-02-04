#!/usr/bin/env python3
"""
Automated model download script for Qwen3-TTS API.
Downloads required models from HuggingFace or ModelScope.
"""

import os
import sys
import argparse
from pathlib import Path

# Model repository mappings
MODEL_REPOS = {
    "qwen3-tts-1.7b-base": {
        "huggingface": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "modelscope": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    },
    "qwen3-tts-1.7b-customvoice": {
        "huggingface": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "modelscope": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "qwen3-tts-1.7b-voicedesign": {
        "huggingface": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "modelscope": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
    "qwen3-tts-0.6b-base": {
        "huggingface": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "modelscope": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
    "qwen3-tts-0.6b-customvoice": {
        "huggingface": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "modelscope": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
    "sensevoice-small": {
        "huggingface": "funasr/paraformer-zh",
        "modelscope": "iic/SenseVoiceSmall",
    },
}


def download_from_huggingface(repo_id: str, local_dir: str, token: str = None):
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"📥 Downloading from HuggingFace: {repo_id}")
        print(f"   Destination: {local_dir}")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"✅ Successfully downloaded to {local_dir}")
        return True
    except ImportError:
        print("❌ huggingface-hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"❌ Error downloading from HuggingFace: {e}")
        return False


def download_from_modelscope(repo_id: str, local_dir: str):
    """Download model from ModelScope."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        print(f"📥 Downloading from ModelScope: {repo_id}")
        print(f"   Destination: {local_dir}")
        
        snapshot_download(
            model_id=repo_id,
            local_dir=local_dir,
        )
        print(f"✅ Successfully downloaded to {local_dir}")
        return True
    except ImportError:
        print("❌ modelscope not installed. Run: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ Error downloading from ModelScope: {e}")
        return False


def download_model(model_name: str, cache_dir: str, source: str = "huggingface", token: str = None):
    """Download a specific model."""
    if model_name not in MODEL_REPOS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_REPOS.keys())}")
        return False
    
    model_config = MODEL_REPOS[model_name]
    
    # Create local directory
    local_dir = os.path.join(cache_dir, model_config[source].replace("/", "--"))
    os.makedirs(local_dir, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"⚠️  Model {model_name} already exists at {local_dir}")
        response = input("   Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping download.")
            return True
    
    # Download
    if source == "huggingface":
        return download_from_huggingface(model_config["huggingface"], local_dir, token)
    else:
        return download_from_modelscope(model_config["modelscope"], local_dir)


def download_all(cache_dir: str, source: str = "huggingface", token: str = None):
    """Download all available models."""
    print("=" * 60)
    print("Downloading all Qwen3-TTS models")
    print("=" * 60)
    
    success_count = 0
    for model_name in MODEL_REPOS.keys():
        print(f"\n[{list(MODEL_REPOS.keys()).index(model_name) + 1}/{len(MODEL_REPOS)}]")
        if download_model(model_name, cache_dir, source, token):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{len(MODEL_REPOS)} models successfully")
    print("=" * 60)
    return success_count == len(MODEL_REPOS)


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-TTS models from HuggingFace or ModelScope"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REPOS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("MODEL_CACHE_DIR", "./models"),
        help="Directory to store downloaded models",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default=os.environ.get("MODEL_SOURCE", "huggingface"),
        help="Download source (default: huggingface)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace token (for gated models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for name, repos in MODEL_REPOS.items():
            print(f"  - {name}")
            print(f"    HuggingFace: {repos['huggingface']}")
            print(f"    ModelScope:  {repos['modelscope']}")
        return
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Download models
    if args.model == "all":
        success = download_all(args.cache_dir, args.source, args.token)
    else:
        success = download_model(args.model, args.cache_dir, args.source, args.token)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
