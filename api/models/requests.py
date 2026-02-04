"""
Pydantic models for API request validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator


class SpeechRequest(BaseModel):
    """
    OpenAI-compatible speech generation request.
    
    Matches the OpenAI /v1/audio/speech endpoint format.
    """
    model: str = Field(
        default="qwen3-tts-1.7b-base",
        description="TTS model to use",
        examples=["qwen3-tts-1.7b-base", "qwen3-tts-1.7b-customvoice"]
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize",
        examples=["Hello, this is a test of the voice cloning system."]
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice ID (for cloned voices) or preset speaker name",
        examples=["voice_abc123", "Vivian", "Ryan"]
    )
    response_format: str = Field(
        default="mp3",
        description="Audio output format",
        examples=["mp3", "wav", "ogg", "opus"]
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5 = half speed, 2.0 = double speed)"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Voice style instructions (e.g., 'happy', 'sad', 'whisper')",
        examples=["happy", "Speak in a cheerful tone"]
    )
    
    # Advanced generation parameters
    temperature: float = Field(
        default=0.9,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more expressive, lower = more stable)"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling probability"
    )
    top_k: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Top-k sampling"
    )
    repetition_penalty: float = Field(
        default=1.05,
        ge=0.1,
        le=2.0,
        description="Repetition penalty (higher = less repetition)"
    )
    max_new_tokens: int = Field(
        default=2048,
        ge=64,
        le=8192,
        description="Maximum tokens to generate"
    )
    
    @validator('response_format')
    def validate_response_format(cls, v):
        allowed = ['mp3', 'wav', 'ogg', 'opus']
        if v.lower() not in allowed:
            raise ValueError(f"response_format must be one of {allowed}")
        return v.lower()
    
    @validator('voice')
    def validate_voice(cls, v):
        if v is not None:
            # Allow preset speakers and voice IDs
            return v
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen3-tts-1.7b-base",
                "input": "Hello, this is a voice clone test!",
                "voice": "voice_abc123",
                "response_format": "mp3",
                "speed": 1.0,
                "instructions": "happy",
                "temperature": 0.9,
            }
        }


class VoiceCloneRequest(BaseModel):
    """
    Request to create a new cloned voice.
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name for the cloned voice",
        examples=["My Cloned Voice"]
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional description of the voice",
        examples=["A warm, friendly voice for narrations"]
    )
    voice_sample_text: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Transcription of the voice sample (optional, improves quality)",
        examples=["This is the text being spoken in the sample."]
    )
    x_vector_only: bool = Field(
        default=False,
        description="Use X-Vector only mode (no text reference, lower quality but works with unknown text)"
    )
    model: str = Field(
        default="qwen3-tts-1.7b-base",
        description="Model to use for voice cloning"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Cloned Voice",
                "description": "A warm, friendly voice",
                "voice_sample_text": "Hello, this is my voice sample.",
                "x_vector_only": False,
                "model": "qwen3-tts-1.7b-base"
            }
        }


class VoiceUpdateRequest(BaseModel):
    """Request to update voice metadata."""
    name: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)


class BatchSpeechRequest(BaseModel):
    """Request for batch speech generation."""
    model: str = Field(default="qwen3-tts-1.7b-base")
    inputs: list[str] = Field(..., min_items=1, max_items=10)
    voice: Optional[str] = Field(default=None)
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instructions: Optional[str] = Field(default=None)
    concatenate: bool = Field(
        default=True,
        description="Concatenate all outputs into single audio file"
    )
    
    @validator('response_format')
    def validate_response_format(cls, v):
        allowed = ['mp3', 'wav', 'ogg', 'opus']
        if v.lower() not in allowed:
            raise ValueError(f"response_format must be one of {allowed}")
        return v.lower()
