"""API routes package."""
from .health import router as health_router
from .tts import router as tts_router
from .voices import router as voices_router

__all__ = ["health_router", "tts_router", "voices_router"]
