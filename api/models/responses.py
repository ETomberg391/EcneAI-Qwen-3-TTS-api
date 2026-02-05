"""
Pydantic models for API responses.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    models_loaded: List[str] = Field(default_factory=list)
    device: str = Field(default="cpu")


class ModelInfo(BaseModel):
    """OpenAI-compatible model information."""
    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model")
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    owned_by: str = Field(default="qwen")
    description: Optional[str] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "qwen3-tts-1.7b-base",
                "object": "model",
                "created": 1704067200,
                "owned_by": "qwen",
                "description": "Base model for voice cloning and dialogue"
            }
        }


class Voice(BaseModel):
    """Voice information for voice management endpoints."""
    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Voice name")
    description: Optional[str] = Field(default=None, description="Voice description")
    created_at: datetime = Field(..., description="When the voice was created")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    model: str = Field(default="qwen3-tts-1.7b-base", description="Model used for cloning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "voice_id": "voice_abc123",
                "name": "My Cloned Voice",
                "description": "A warm, friendly voice",
                "created_at": "2024-01-01T00:00:00Z",
                "sample_rate": 24000,
                "model": "qwen3-tts-1.7b-base"
            }
        }


class VoiceList(BaseModel):
    """List of voices response."""
    voices: List[Voice] = Field(default_factory=list)
    total: int = Field(default=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "voices": [
                    {
                        "voice_id": "voice_abc123",
                        "name": "My Cloned Voice",
                        "description": "A warm, friendly voice",
                        "created_at": "2024-01-01T00:00:00Z",
                        "sample_rate": 24000,
                        "model": "qwen3-tts-1.7b-base"
                    }
                ],
                "total": 1
            }
        }


class VoiceCloneResponse(BaseModel):
    """Response from voice cloning endpoint."""
    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Voice name")
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model: str = Field(default="qwen3-tts-1.7b-base")
    
    class Config:
        json_schema_extra = {
            "example": {
                "voice_id": "voice_abc123",
                "name": "My Cloned Voice",
                "description": "A warm, friendly voice",
                "created_at": "2024-01-01T00:00:00Z",
                "model": "qwen3-tts-1.7b-base"
            }
        }


class PresetSpeakerInfo(BaseModel):
    """Information about a preset speaker."""
    id: str = Field(..., description="Speaker identifier")
    name: str = Field(..., description="Speaker name")
    description: str = Field(..., description="Speaker characteristics")
    language: str = Field(..., description="Primary language")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "Vivian",
                "name": "Vivian",
                "description": "Chinese - Bright, Sharp, Young Female",
                "language": "Chinese"
            }
        }


class PresetSpeakersList(BaseModel):
    """List of available preset speakers."""
    speakers: List[PresetSpeakerInfo] = Field(default_factory=list)
    total: int = Field(default=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "speakers": [
                    {
                        "id": "Vivian",
                        "name": "Vivian",
                        "description": "Chinese - Bright, Sharp, Young Female",
                        "language": "Chinese"
                    },
                    {
                        "id": "Ryan",
                        "name": "Ryan",
                        "description": "English - Rhythmic, Dynamic Male",
                        "language": "English"
                    }
                ],
                "total": 9
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Model not found",
                "detail": "Model qwen3-tts-1.7b-base is not downloaded. Run download first."
            }
        }


class ModelStatus(BaseModel):
    """Model download/load status."""
    model_id: str = Field(..., description="Model identifier")
    downloaded: bool = Field(..., description="Whether model files are downloaded")
    loaded: bool = Field(..., description="Whether model is loaded in memory")
    path: Optional[str] = Field(default=None, description="Local path to model")
    size_gb: Optional[float] = Field(default=None, description="Model size in GB")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "qwen3-tts-1.7b-base",
                "downloaded": True,
                "loaded": False,
                "path": "/path/to/models/Qwen--Qwen3-TTS-1.7B-base",
                "size_gb": 3.4
            }
        }


class ModelLoadResponse(BaseModel):
    """Response from model load/download endpoint."""
    success: bool = Field(..., description="Whether operation succeeded")
    model_id: str = Field(..., description="Model identifier")
    message: str = Field(..., description="Status message")
    downloaded: bool = Field(..., description="Whether model is now downloaded")
    loaded: bool = Field(..., description="Whether model is now loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_id": "qwen3-tts-1.7b-base",
                "message": "Model downloaded and loaded successfully",
                "downloaded": True,
                "loaded": True
            }
        }
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": {"field": "speed", "issue": "must be between 0.5 and 2.0"}
            }
        }
