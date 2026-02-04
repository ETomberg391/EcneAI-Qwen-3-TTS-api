"""
Main FastAPI application for Qwen3-TTS API.

This is an OpenAI-compatible API for voice cloning and text-to-speech
using the Qwen3-TTS model.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routes import health_router, tts_router, voices_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Qwen3-TTS API Starting Up")
    logger.info("=" * 60)
    logger.info(f"Version: 1.0.0")
    logger.info(f"Device: {settings.device_str}")
    logger.info(f"Precision: {settings.PRECISION}")
    logger.info(f"Model Cache: {settings.model_cache_path}")
    logger.info(f"Voice Storage: {settings.voice_storage_path}")
    logger.info(f"Auto Download: {settings.ENABLE_AUTO_DOWNLOAD}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Qwen3-TTS API...")
    
    # Unload all models
    from api.services.model_manager import model_manager
    model_manager.unload_all()
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Qwen3-TTS API",
    description="""
    OpenAI-compatible API for Qwen3-TTS voice cloning and text-to-speech.
    
    ## Features
    
    - **Voice Cloning**: Clone any voice from 3+ seconds of audio
    - **Preset Speakers**: 9 built-in voices (Vivian, Ryan, Aiden, etc.)
    - **OpenAI Compatible**: Drop-in replacement for /v1/audio/speech
    - **Multiple Formats**: MP3, WAV, OGG, OPUS output
    - **Speed Control**: 0.5x to 2.0x playback speed
    - **Emotion Control**: Happy, sad, whisper, angry, etc.
    
    ## Quick Start
    
    ### Using a preset speaker:
    ```bash
    curl -X POST http://localhost:8000/v1/audio/speech \\
      -F "model=qwen3-tts-1.7b-customvoice" \\
      -F "input=Hello, world!" \\
      -F "voice=Vivian" \\
      -F "response_format=mp3" \\
      --output speech.mp3
    ```
    
    ### Instant voice cloning:
    ```bash
    curl -X POST http://localhost:8000/v1/audio/speech \\
      -F "model=qwen3-tts-1.7b-base" \\
      -F "input=This is my cloned voice speaking!" \\
      -F "voice_sample=@my_voice.wav" \\
      -F "voice_sample_text=Original text in sample" \\
      --output cloned.mp3
    ```
    
    ### Save and reuse a voice:
    ```bash
    # First, clone and save the voice
    curl -X POST http://localhost:8000/v1/voices \\
      -F "name=My Voice" \\
      -F "voice_sample=@sample.wav" \\
      -F "voice_sample_text=Original text"
    
    # Returns: {"voice_id": "voice_abc123", ...}
    
    # Then use the saved voice
    curl -X POST http://localhost:8000/v1/audio/speech \\
      -F "voice=voice_abc123" \\
      -F "input=Hello from my saved voice!" \\
      --output speech.mp3
    ```
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(health_router)
app.include_router(tts_router)
app.include_router(voices_router)


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    """Redirect to Swagger UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=False,
    )
