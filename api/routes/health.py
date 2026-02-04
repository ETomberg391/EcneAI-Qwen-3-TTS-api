"""
Health check and status endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, status

from api.config import settings
from api.services.model_manager import model_manager
from api.models.responses import HealthResponse, ModelInfo

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API service.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        models_loaded=model_manager.list_loaded_models(),
        device=settings.device_str,
    )


@router.get("/v1/models", response_model=list[ModelInfo])
async def list_models():
    """
    List available TTS models.
    
    OpenAI-compatible endpoint that returns available models.
    """
    from api.config import MODEL_REPOS
    
    models = []
    for model_id in MODEL_REPOS.keys():
        description = {
            "qwen3-tts-1.7b-base": "Base model for voice cloning and dialogue (3.4GB)",
            "qwen3-tts-1.7b-customvoice": "Model with preset speakers (3.4GB)",
            "qwen3-tts-1.7b-voicedesign": "Model for voice design from text (3.4GB)",
            "qwen3-tts-0.6b-base": "Lightweight base model (1.2GB)",
            "qwen3-tts-0.6b-customvoice": "Lightweight preset speaker model (1.2GB)",
        }.get(model_id, "Qwen3-TTS Model")
        
        models.append(ModelInfo(
            id=model_id,
            object="model",
            created=int(datetime.utcnow().timestamp()),
            owned_by="qwen",
            description=description,
        ))
    
    return models


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Qwen3-TTS API",
        "version": "1.0.0",
        "description": "OpenAI-compatible API for Qwen3-TTS voice cloning and synthesis",
        "docs_url": "/docs",
        "health_check": "/health",
        "models": "/v1/models",
    }
