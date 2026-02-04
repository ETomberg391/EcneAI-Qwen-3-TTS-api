"""
Voice management API routes.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status

from api.models.requests import VoiceCloneRequest, VoiceUpdateRequest
from api.models.responses import Voice, VoiceList, VoiceCloneResponse
from api.services.tts_service import tts_service
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Voices"])


@router.get("/v1/voices", response_model=VoiceList)
async def list_voices():
    """
    List all saved cloned voices.
    
    Returns metadata for all voices that have been cloned and saved.
    """
    try:
        voices_data = tts_service.list_voices()
        
        voice_models = [
            Voice(
                voice_id=v["voice_id"],
                name=v["name"],
                description=v.get("description"),
                created_at=v.get("created_at"),
                sample_rate=v.get("sample_rate", 24000),
                model=v.get("model", "qwen3-tts-1.7b-base"),
            )
            for v in voices_data
        ]
        
        return VoiceList(voices=voice_models, total=len(voice_models))
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/voices", response_model=VoiceCloneResponse, status_code=status.HTTP_201_CREATED)
async def clone_voice(
    name: str = Form(..., min_length=1, max_length=100),
    description: Optional[str] = Form(default=None),
    voice_sample: UploadFile = File(...),
    voice_sample_text: Optional[str] = Form(default=None),
    model: str = Form(default="qwen3-tts-1.7b-base"),
    x_vector_only: bool = Form(default=False),
):
    """
    Clone a voice from an audio sample.
    
    Upload an audio sample (3+ seconds recommended) and create a voice clone
    that can be reused for future text-to-speech generation.
    
    ## Parameters:
    - **name**: Name for the cloned voice
    - **description**: Optional description
    - **voice_sample**: Audio file containing the voice to clone
    - **voice_sample_text**: Transcription of the sample (optional, improves quality)
    - **model**: Model to use for cloning
    - **x_vector_only**: Use X-Vector only mode (works without transcript but lower quality)
    
    ## Returns:
    Voice metadata including the voice_id for future use.
    
    ## Example:
    ```bash
    curl -X POST http://localhost:8000/v1/voices \\
      -F "name=My Voice" \\
      -F "voice_sample=@sample.wav" \\
      -F "voice_sample_text=Hello, this is my voice"
    ```
    """
    try:
        # Read voice sample
        voice_sample_bytes = await voice_sample.read()
        
        # Check file size
        if len(voice_sample_bytes) > settings.MAX_AUDIO_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Voice sample too large. Max size: {settings.MAX_AUDIO_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Validate audio format
        allowed_content_types = [
            "audio/wav", "audio/x-wav", "audio/wave",
            "audio/mp3", "audio/mpeg",
            "audio/ogg", "audio/flac", "audio/x-flac",
            "audio/m4a", "audio/mp4",
        ]
        
        if voice_sample.content_type not in allowed_content_types:
            logger.warning(f"Content-Type '{voice_sample.content_type}' may not be supported")
        
        logger.info(f"Cloning voice from sample: {voice_sample.filename}")
        
        # Clone the voice
        result = await tts_service.clone_voice(
            voice_sample=voice_sample_bytes,
            name=name,
            description=description,
            voice_sample_text=voice_sample_text,
            model=model,
            x_vector_only=x_vector_only,
        )
        
        return VoiceCloneResponse(
            voice_id=result["voice_id"],
            name=result["name"],
            description=result.get("description"),
            created_at=result["created_at"],
            model=result["model"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cloning voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/voices/{voice_id}", response_model=Voice)
async def get_voice(voice_id: str):
    """
    Get details of a specific voice.
    
    ## Parameters:
    - **voice_id**: The unique voice identifier
    """
    try:
        voice_data = tts_service.get_voice(voice_id)
        
        if voice_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice not found: {voice_id}"
            )
        
        return Voice(
            voice_id=voice_data["voice_id"],
            name=voice_data["name"],
            description=voice_data.get("description"),
            created_at=voice_data.get("created_at"),
            sample_rate=voice_data.get("sample_rate", 24000),
            model=voice_data.get("model", "qwen3-tts-1.7b-base"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/voices/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(voice_id: str):
    """
    Delete a cloned voice.
    
    ## Parameters:
    - **voice_id**: The unique voice identifier
    """
    try:
        success = tts_service.delete_voice(voice_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Voice not found: {voice_id}"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/v1/voices/{voice_id}", response_model=Voice)
async def update_voice(voice_id: str, request: VoiceUpdateRequest):
    """
    Update voice metadata (name and/or description).
    
    ## Parameters:
    - **voice_id**: The unique voice identifier
    - **name**: New name (optional)
    - **description**: New description (optional)
    """
    try:
        # Get existing voice
        voice_data = tts_service.get_voice(voice_id)
        
        if voice_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Voice not found: {voice_id}"
            )
        
        # Update fields
        if request.name is not None:
            voice_data["name"] = request.name
        if request.description is not None:
            voice_data["description"] = request.description
        
        # Save updated metadata
        import json
        from pathlib import Path
        meta_path = Path(settings.VOICE_STORAGE_DIR) / f"{voice_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(voice_data, f, indent=2)
        
        return Voice(
            voice_id=voice_data["voice_id"],
            name=voice_data["name"],
            description=voice_data.get("description"),
            created_at=voice_data.get("created_at"),
            sample_rate=voice_data.get("sample_rate", 24000),
            model=voice_data.get("model", "qwen3-tts-1.7b-base"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
