"""
Health check and status endpoints.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, status, HTTPException

from api.config import settings
from api.services.model_manager import model_manager
from api.models.responses import (
    HealthResponse, 
    ModelInfo, 
    ModelStatus,
    ModelLoadResponse
)

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


@router.get("/v1/models/status", response_model=list[ModelStatus])
async def get_models_status():
    """
    Get download and load status of all models.
    
    Returns information about which models are downloaded and loaded.
    """
    statuses = model_manager.get_all_models_status()
    return [
        ModelStatus(
            model_id=s["model_id"],
            downloaded=s["downloaded"],
            loaded=s["loaded"],
            path=s.get("path"),
            size_gb=s.get("size_gb"),
        )
        for s in statuses
    ]


@router.get("/v1/models/{model_id}/status", response_model=ModelStatus)
async def get_model_status(model_id: str):
    """
    Get status of a specific model.
    
    Args:
        model_id: Model identifier (e.g., qwen3-tts-1.7b-base)
    """
    from api.config import MODEL_REPOS
    
    if model_id not in MODEL_REPOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(MODEL_REPOS.keys())}"
        )
    
    status = model_manager.get_model_status(model_id)
    return ModelStatus(
        model_id=status["model_id"],
        downloaded=status["downloaded"],
        loaded=status["loaded"],
        path=status.get("path"),
        size_gb=status.get("size_gb"),
    )


@router.post("/v1/models/{model_id}/download", response_model=ModelLoadResponse)
async def download_model(model_id: str):
    """
    Download a model without loading it into memory.
    
    This endpoint downloads the model files to local cache.
    Use /v1/models/{model_id}/load to download AND load the model.
    
    Args:
        model_id: Model identifier (e.g., qwen3-tts-1.7b-base)
    """
    from api.config import MODEL_REPOS
    
    if model_id not in MODEL_REPOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(MODEL_REPOS.keys())}"
        )
    
    # Check if already downloaded
    if model_manager.is_model_downloaded(model_id):
        return ModelLoadResponse(
            success=True,
            model_id=model_id,
            message="Model already downloaded",
            downloaded=True,
            loaded=model_manager.is_model_loaded(model_id),
        )
    
    # Check if auto-download is enabled
    if not settings.ENABLE_AUTO_DOWNLOAD:
        raise HTTPException(
            status_code=400,
            detail="Auto-download is disabled. Enable it in settings or download manually."
        )
    
    try:
        success = model_manager.download_model(model_id)
        if success:
            return ModelLoadResponse(
                success=True,
                model_id=model_id,
                message="Model downloaded successfully",
                downloaded=True,
                loaded=model_manager.is_model_loaded(model_id),
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to download model. Check logs for details."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading model: {str(e)}"
        )


@router.post("/v1/models/{model_id}/load", response_model=ModelLoadResponse)
async def load_model(model_id: str):
    """
    Download (if needed) and load a model into memory.
    
    This endpoint will:
    1. Download the model if not already downloaded
    2. Load the model into GPU/CPU memory
    
    Args:
        model_id: Model identifier (e.g., qwen3-tts-1.7b-base)
    """
    from api.config import MODEL_REPOS
    
    if model_id not in MODEL_REPOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(MODEL_REPOS.keys())}"
        )
    
    # Check if already loaded
    if model_manager.is_model_loaded(model_id):
        return ModelLoadResponse(
            success=True,
            model_id=model_id,
            message="Model already loaded",
            downloaded=True,
            loaded=True,
        )
    
    # Check if auto-download is enabled and model not downloaded
    if not settings.ENABLE_AUTO_DOWNLOAD and not model_manager.is_model_downloaded(model_id):
        raise HTTPException(
            status_code=400,
            detail="Model not downloaded and auto-download is disabled. "
                   "Enable ENABLE_AUTO_DOWNLOAD in settings or download manually."
        )
    
    try:
        success = model_manager.preload_model(model_id)
        if success:
            return ModelLoadResponse(
                success=True,
                model_id=model_id,
                message="Model downloaded and loaded successfully",
                downloaded=True,
                loaded=True,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model. Check logs for details."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )


@router.post("/v1/models/{model_id}/unload", response_model=ModelLoadResponse)
async def unload_model(model_id: str):
    """
    Unload a model from memory.
    
    This keeps the model files on disk but frees GPU/CPU memory.
    
    Args:
        model_id: Model identifier (e.g., qwen3-tts-1.7b-base)
    """
    from api.config import MODEL_REPOS
    
    if model_id not in MODEL_REPOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(MODEL_REPOS.keys())}"
        )
    
    was_loaded = model_manager.is_model_loaded(model_id)
    model_manager.unload_model(model_id)
    
    return ModelLoadResponse(
        success=True,
        model_id=model_id,
        message="Model unloaded from memory" if was_loaded else "Model was not loaded",
        downloaded=model_manager.is_model_downloaded(model_id),
        loaded=False,
    )


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
        "model_status": "/v1/models/status",
    }
