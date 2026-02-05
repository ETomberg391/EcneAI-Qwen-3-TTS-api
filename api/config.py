"""
Configuration settings for Qwen3-TTS API.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    MAX_CONCURRENT_REQUESTS: int = 1
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    
    # Model Settings
    MODEL_CACHE_DIR: str = "./models"
    MODEL_SOURCE: Literal["huggingface", "modelscope"] = "huggingface"
    DEFAULT_MODEL: str = "qwen3-tts-1.7b-base"
    
    # Device Settings
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    PRECISION: Literal["bf16", "fp16", "fp32"] = "bf16"
    ATTN_MODE: Literal["sdpa", "flash_attention_2", "eager", "sage_attention"] = "sdpa"
    ENABLE_AUTO_DOWNLOAD: bool = True
    PRELOAD_MODELS: Optional[str] = None  # Comma-separated list of models to preload on startup
    
    # Voice Storage
    VOICE_STORAGE_DIR: str = "./voices"
    MAX_VOICE_AGE_DAYS: int = 30
    
    # Audio Settings
    DEFAULT_SAMPLE_RATE: int = 24000
    OUTPUT_SAMPLE_RATE: int = 44100
    DEFAULT_SPEED: float = 1.0
    DEFAULT_FADE_IN_MS: int = 10
    DEFAULT_FADE_OUT_MS: int = 50
    
    # File Upload Limits
    MAX_AUDIO_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_TEXT_LENGTH: int = 5000
    
    # HuggingFace Token (optional)
    HF_TOKEN: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    API_KEY: Optional[str] = None  # Set to enable API key authentication
    ENABLE_CORS: bool = True
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def model_cache_path(self) -> Path:
        """Get absolute path to model cache directory."""
        return Path(self.MODEL_CACHE_DIR).resolve()
    
    @property
    def voice_storage_path(self) -> Path:
        """Get absolute path to voice storage directory."""
        path = Path(self.VOICE_STORAGE_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def device_str(self) -> str:
        """Get the device string for model loading."""
        import torch
        
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.DEVICE
    
    @property
    def torch_dtype(self):
        """Get torch dtype based on precision setting."""
        import torch
        
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        
        dtype = dtype_map.get(self.PRECISION, torch.bfloat16)
        
        # MPS doesn't support bf16 well
        if self.device_str == "mps" and dtype == torch.bfloat16:
            return torch.float16
        
        return dtype


# Global settings instance
settings = Settings()

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
}

# Preset speakers mapping
PRESET_SPEAKERS = {
    "Vivian": "Vivian (Chinese - Bright, Sharp, Young Female)",
    "Serena": "Serena (Chinese - Warm, Soft, Young Female)",
    "Uncle_Fu": "Uncle_Fu (Chinese - Deep, Mellow, Mature Male)",
    "Dylan": "Dylan (Chinese Beijing - Clear, Natural Young Male)",
    "Eric": "Eric (Chinese Sichuan - Lively, Husky Male)",
    "Ryan": "Ryan (English - Rhythmic, Dynamic Male)",
    "Aiden": "Aiden (English - Sunny, Clear American Male)",
    "Ono_Anna": "Ono_Anna (Japanese - Light, Playful Female)",
    "Sohee": "Sohee (Korean - Emotional, Warm Female)",
}

# Emotion mappings
EMOTION_MAP = {
    # English
    "happy": "Speak in a very happy and cheerful tone, full of energy.",
    "excited": "Speak in a very excited tone, slightly faster, full of excitement.",
    "angry": "Speak with an angry and stern tone, aggressive and sharp.",
    "sad": "Speak in a sad, low-spirited voice with a hint of sorrow.",
    "gentle": "Speak softly and warmly, full of love and care.",
    "fearful": "Speak with a trembling, terrified voice, sounding panic-stricken.",
    "cold": "Speak in a cold tone, emotionless, distant and mechanical.",
    "whisper": "Speak in a very soft whisper, as if sharing a secret.",
    "surprised": "Speak with a shocked and incredulous tone, pitch raised.",
    "disgusted": "Speak with a tone of revulsion and strong dislike.",
    "neutral": "Speak in a natural, calm, and steady tone.",
    # Chinese
    "开心": "用愉快、开心的语气说话，充满活力和阳光。",
    "激动": "语气非常激动，语速稍快，充满兴奋感。",
    "生气": "用愤怒、严厉的语气说话，语调生硬且带有攻击性。",
    "难过": "语气低沉、忧伤，带有明显的哀伤和哭腔感。",
    "温柔": "声音轻柔、温婉，充满爱意和关怀。",
    "恐惧": "声音颤抖，语气惊恐不安，呼吸感加重。",
    "冷酷": "语气冰冷、没有任何情感波动，显得疏离而机械。",
    "低语": "用极小的声音说话，像是在耳边轻声细语，充满神秘感。",
    "惊讶": "语气充满震惊和不可思议，音调上扬，带有明显的意外感。",
    "厌恶": "语气充满嫌弃和反感，带有不屑和排斥的情感。",
    "平静": "语气平稳自然，没有明显的情绪波动，清晰且沉稳。",
}


# Audio format settings
AUDIO_FORMATS = {
    "mp3": {"mimetype": "audio/mpeg", "extension": ".mp3"},
    "wav": {"mimetype": "audio/wav", "extension": ".wav"},
    "ogg": {"mimetype": "audio/ogg", "extension": ".ogg"},
    "opus": {"mimetype": "audio/opus", "extension": ".opus"},
}
