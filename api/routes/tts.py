"""
Text-to-Speech API routes.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse

from api.models.requests import SpeechRequest
from api.models.responses import PresetSpeakersList, PresetSpeakerInfo
from api.services.tts_service import tts_service
from api.config import settings, AUDIO_FORMATS, PRESET_SPEAKERS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Speech"])


@router.post("/v1/audio/speech")
async def create_speech(
    model: str = Form(default="qwen3-tts-1.7b-base"),
    input: str = Form(..., min_length=1, max_length=5000),
    voice: Optional[str] = Form(default=None),
    voice_sample: Optional[UploadFile] = File(default=None),
    voice_sample_text: Optional[str] = Form(default=None),
    response_format: str = Form(default="mp3"),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
    instructions: Optional[str] = Form(default=None),
    temperature: float = Form(default=0.9, ge=0.1, le=2.0),
    top_p: float = Form(default=1.0, ge=0.1, le=1.0),
    top_k: int = Form(default=50, ge=0, le=200),
    repetition_penalty: float = Form(default=1.05, ge=0.1, le=2.0),
    max_new_tokens: int = Form(default=2048, ge=64, le=8192),
    language: str = Form(default="auto"),
):
    """
    Generate speech from text.
    
    This endpoint is compatible with OpenAI's /v1/audio/speech API.
    Supports both preset speakers and voice cloning.
    
    ## Parameters:
    - **model**: TTS model to use (qwen3-tts-1.7b-base, qwen3-tts-1.7b-customvoice)
    - **input**: Text to synthesize (max 5000 characters)
    - **voice**: Voice ID for cloned voices OR preset speaker name
    - **voice_sample**: Audio file for instant voice cloning
    - **voice_sample_text**: Transcription of voice sample (improves cloning quality)
    - **response_format**: Output format (mp3, wav, ogg, opus)
    - **speed**: Speech speed (0.5 to 2.0)
    - **instructions**: Voice style (happy, sad, whisper, etc.)
    - **temperature**: Sampling temperature (0.1 to 2.0)
    - **top_p**: Nucleus sampling (0.1 to 1.0)
    - **top_k**: Top-k sampling (0 to 200)
    - **repetition_penalty**: Penalty for repetition (0.1 to 2.0)
    - **max_new_tokens**: Max generation length (64 to 8192)
    - **language**: Language code (auto, Chinese, English, Japanese, etc.)
    
    ## Examples:
    
    ### Using a preset speaker:
    ```
    model=qwen3-tts-1.7b-customvoice
    input=Hello world!
    voice=Vivian
    instructions=happy
    ```
    
    ### Instant voice cloning:
    ```
    model=qwen3-tts-1.7b-base
    input=This is my cloned voice speaking
    voice_sample=<upload audio file>
    voice_sample_text=Original text in sample
    ```
    
    ### Using a saved cloned voice:
    ```
    model=qwen3-tts-1.7b-base
    input=Hello from my saved voice
    voice=voice_abc123
    ```
    """
    try:
        # Validate response format
        if response_format.lower() not in AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format. Must be one of: {list(AUDIO_FORMATS.keys())}"
            )
        
        # Check if using preset speaker
        if voice in PRESET_SPEAKERS:
            # Use customvoice model with preset speaker
            if "customvoice" not in model:
                model = "qwen3-tts-1.7b-customvoice"
            
            logger.info(f"Generating speech with preset speaker: {voice}")
            audio_bytes, content_type = await tts_service.synthesize_with_preset(
                text=input,
                speaker=voice,
                model=model,
                instructions=instructions,
                speed=speed,
                response_format=response_format,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        else:
            # Voice cloning path
            voice_sample_bytes = None
            if voice_sample is not None:
                voice_sample_bytes = await voice_sample.read()
                if len(voice_sample_bytes) > settings.MAX_AUDIO_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Voice sample too large. Max size: {settings.MAX_AUDIO_FILE_SIZE / 1024 / 1024:.1f}MB"
                    )
            
            # Check that we have either voice_id or voice_sample
            if voice is None and voice_sample_bytes is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'voice' (voice_id or preset speaker) or 'voice_sample' (audio file) must be provided"
                )
            
            logger.info(f"Generating speech with voice cloning")
            audio_bytes, content_type = await tts_service.synthesize(
                text=input,
                voice_id=voice,
                voice_sample=voice_sample_bytes,
                voice_sample_text=voice_sample_text,
                model=model,
                instructions=instructions,
                speed=speed,
                response_format=response_format,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/speech/json")
async def create_speech_json(request: SpeechRequest):
    """
    Generate speech from text using JSON body (alternative to form data).
    
    Note: For voice cloning with audio upload, use the form data endpoint instead.
    """
    try:
        # This endpoint only works with saved voices or preset speakers
        if request.voice in PRESET_SPEAKERS:
            audio_bytes, content_type = await tts_service.synthesize_with_preset(
                text=request.input,
                speaker=request.voice,
                model=request.model,
                instructions=request.instructions,
                speed=request.speed,
                response_format=request.response_format,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=request.max_new_tokens,
            )
        else:
            # Use saved voice
            audio_bytes, content_type = await tts_service.synthesize(
                text=request.input,
                voice_id=request.voice,
                model=request.model,
                instructions=request.instructions,
                speed=request.speed,
                response_format=request.response_format,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_new_tokens=request.max_new_tokens,
            )
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/speakers", response_model=PresetSpeakersList)
async def list_preset_speakers():
    """
    List available preset speakers.
    
    These speakers are built into the CustomVoice model and can be used
    directly without any voice cloning.
    """
    speakers = tts_service.list_preset_speakers()
    
    speaker_infos = [
        PresetSpeakerInfo(
            id=s["id"],
            name=s["name"],
            description=s["description"],
            language=s["language"],
        )
        for s in speakers
    ]
    
    return PresetSpeakersList(
        speakers=speaker_infos,
        total=len(speaker_infos)
    )
