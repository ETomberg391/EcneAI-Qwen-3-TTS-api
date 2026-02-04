"""
Core TTS service for voice cloning and synthesis.
"""

import os
import re
import uuid
import logging
import hashlib
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import torch

from .model_manager import model_manager
from .audio_utils import AudioProcessor
from api.config import settings, PRESET_SPEAKERS, EMOTION_MAP

logger = logging.getLogger(__name__)


class VoicePrompt:
    """Container for voice clone prompts."""
    
    def __init__(self, prompt_data: Any, model_name: str, created_at: datetime = None):
        self.prompt_data = prompt_data
        self.model_name = model_name
        self.created_at = created_at or datetime.utcnow()
    
    def save(self, path: Path):
        """Save prompt to disk."""
        torch.save({
            "prompt": self.prompt_data,
            "model_name": self.model_name,
            "created_at": self.created_at,
        }, path)
    
    @classmethod
    def load(cls, path: Path):
        """Load prompt from disk."""
        data = torch.load(path, weights_only=False)
        return cls(
            prompt_data=data["prompt"],
            model_name=data["model_name"],
            created_at=data.get("created_at", datetime.utcnow())
        )


class TTSService:
    """
    Main TTS service for voice cloning and synthesis.
    
    Supports:
    - Voice cloning from audio samples
    - Preset speaker voices
    - Voice design (with voice design model)
    - Audio post-processing (speed, fade, resample)
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.voice_storage = Path(settings.VOICE_STORAGE_DIR)
        self.voice_storage.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded voice prompts
        self._voice_cache: Dict[str, VoicePrompt] = {}
        
        logger.info("TTSService initialized")
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_sample: Optional[bytes] = None,
        voice_sample_text: Optional[str] = None,
        model: str = None,
        instructions: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        language: str = "auto",
        **gen_kwargs
    ) -> Tuple[bytes, str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: ID of a saved cloned voice
            voice_sample: Raw bytes of voice sample audio (for instant cloning)
            voice_sample_text: Transcription of voice sample
            model: Model to use
            instructions: Voice style instructions
            speed: Speech speed multiplier
            response_format: Output audio format
            language: Language code
            **gen_kwargs: Generation parameters (temperature, top_p, etc.)
            
        Returns:
            Tuple of (audio_bytes, content_type)
        """
        # Set default model
        if model is None:
            model = settings.DEFAULT_MODEL
        
        # Get emotion instruction
        final_instruct = self._process_instructions(instructions)
        
        # Parse text for pauses
        segments = self.audio_processor.parse_text_with_pauses(text)
        
        # Generate audio based on voice type
        if voice_sample is not None:
            # Instant voice cloning from uploaded sample
            logger.info("Performing instant voice cloning")
            audio_array = await self._synthesize_with_sample(
                segments=segments,
                voice_sample=voice_sample,
                voice_sample_text=voice_sample_text,
                model=model,
                instruct=final_instruct,
                language=language,
                **gen_kwargs
            )
        elif voice_id is not None:
            # Use saved voice
            logger.info(f"Using saved voice: {voice_id}")
            audio_array = await self._synthesize_with_saved_voice(
                segments=segments,
                voice_id=voice_id,
                model=model,
                instruct=final_instruct,
                language=language,
                **gen_kwargs
            )
        else:
            # No voice specified - use default or raise error
            raise ValueError(
                "Either voice_id or voice_sample must be provided"
            )
        
        # Apply speed adjustment
        if speed != 1.0:
            audio_array = self.audio_processor.adjust_speed(
                audio_array, settings.DEFAULT_SAMPLE_RATE, speed
            )
        
        # Resample to output sample rate if different
        if settings.OUTPUT_SAMPLE_RATE != settings.DEFAULT_SAMPLE_RATE:
            audio_array = self.audio_processor.resample(
                audio_array,
                settings.DEFAULT_SAMPLE_RATE,
                settings.OUTPUT_SAMPLE_RATE
            )
            sr = settings.OUTPUT_SAMPLE_RATE
        else:
            sr = settings.DEFAULT_SAMPLE_RATE
        
        # Convert to output format
        audio_bytes = self.audio_processor.save_audio_to_bytes(
            audio_array,
            sr,
            format=response_format,
            fade_in_ms=settings.DEFAULT_FADE_IN_MS,
            fade_out_ms=settings.DEFAULT_FADE_OUT_MS
        )
        
        # Determine content type
        from api.config import AUDIO_FORMATS
        content_type = AUDIO_FORMATS.get(response_format, {}).get(
            "mimetype", "audio/mpeg"
        )
        
        return audio_bytes, content_type
    
    async def synthesize_with_preset(
        self,
        text: str,
        speaker: str,
        model: str = "qwen3-tts-1.7b-customvoice",
        instructions: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        language: str = "auto",
        **gen_kwargs
    ) -> Tuple[bytes, str]:
        """
        Synthesize using a preset speaker voice.
        
        Args:
            text: Text to synthesize
            speaker: Speaker name (e.g., "Vivian", "Ryan")
            model: Model to use (should be customvoice model)
            instructions: Voice style instructions
            speed: Speech speed multiplier
            response_format: Output audio format
            language: Language code
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (audio_bytes, content_type)
        """
        # Validate speaker
        if speaker not in PRESET_SPEAKERS:
            raise ValueError(
                f"Unknown speaker: {speaker}. "
                f"Available: {list(PRESET_SPEAKERS.keys())}"
            )
        
        # Get model
        tts_model = model_manager.get_model(model)
        
        # Process instructions
        final_instruct = self._process_instructions(instructions)
        
        # Parse segments
        segments = self.audio_processor.parse_text_with_pauses(text)
        
        # Generate audio
        audio_results = []
        sr = settings.DEFAULT_SAMPLE_RATE
        
        for seg_type, content in segments:
            if seg_type == "pause":
                if content > 0:
                    silence_samples = int(content * sr)
                    audio_results.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                try:
                    wavs, current_sr = tts_model.generate_custom_voice(
                        text=[content],
                        language=[language],
                        speaker=[speaker],
                        instruct=[final_instruct] if final_instruct else None,
                        **self._pack_gen_kwargs(gen_kwargs),
                    )
                    sr = current_sr
                    wav = wavs[0]
                    if wav.ndim > 1:
                        wav = wav.squeeze()
                    audio_results.append(wav)
                except Exception as e:
                    logger.error(f"Error generating segment '{content}': {e}")
                    raise
        
        # Concatenate
        if not audio_results:
            raise RuntimeError("No audio generated")
        
        audio_array = np.concatenate(audio_results)
        
        # Apply speed adjustment
        if speed != 1.0:
            audio_array = self.audio_processor.adjust_speed(audio_array, sr, speed)
        
        # Resample if needed
        if settings.OUTPUT_SAMPLE_RATE != sr:
            audio_array = self.audio_processor.resample(
                audio_array, sr, settings.OUTPUT_SAMPLE_RATE
            )
            sr = settings.OUTPUT_SAMPLE_RATE
        
        # Convert to output format
        audio_bytes = self.audio_processor.save_audio_to_bytes(
            audio_array,
            sr,
            format=response_format,
            fade_in_ms=settings.DEFAULT_FADE_IN_MS,
            fade_out_ms=settings.DEFAULT_FADE_OUT_MS
        )
        
        from api.config import AUDIO_FORMATS
        content_type = AUDIO_FORMATS.get(response_format, {}).get(
            "mimetype", "audio/mpeg"
        )
        
        return audio_bytes, content_type
    
    async def clone_voice(
        self,
        voice_sample: bytes,
        name: str,
        description: Optional[str] = None,
        voice_sample_text: Optional[str] = None,
        model: str = None,
        x_vector_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Clone a voice from an audio sample and save it.
        
        Args:
            voice_sample: Raw audio file bytes
            name: Name for the cloned voice
            description: Optional description
            voice_sample_text: Transcription of the sample
            model: Model to use
            x_vector_only: Use X-Vector only mode
            
        Returns:
            Voice metadata dict
        """
        if model is None:
            model = settings.DEFAULT_MODEL
        
        # Get model
        tts_model = model_manager.get_model(model)
        
        # Load audio
        wav_np, sr = self.audio_processor.load_audio_file(voice_sample)
        
        # Create voice prompt
        logger.info("Creating voice clone prompt...")
        prompt = tts_model.create_voice_clone_prompt(
            ref_audio=(wav_np, sr),
            ref_text=voice_sample_text if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )
        
        # Generate voice ID
        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        
        # Save prompt
        voice_prompt = VoicePrompt(prompt, model)
        voice_path = self.voice_storage / f"{voice_id}.pt"
        voice_prompt.save(voice_path)
        
        # Save metadata
        metadata = {
            "voice_id": voice_id,
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "model": model,
            "sample_rate": sr,
            "x_vector_only": x_vector_only,
        }
        
        meta_path = self.voice_storage / f"{voice_id}.json"
        import json
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Cache the prompt
        self._voice_cache[voice_id] = voice_prompt
        
        logger.info(f"Voice cloned successfully: {voice_id}")
        return metadata
    
    async def _synthesize_with_sample(
        self,
        segments: List[Tuple[str, str]],
        voice_sample: bytes,
        voice_sample_text: Optional[str],
        model: str,
        instruct: Optional[str],
        language: str,
        **gen_kwargs
    ) -> np.ndarray:
        """Synthesize using an uploaded voice sample."""
        # Get model
        tts_model = model_manager.get_model(model)
        
        # Load audio
        wav_np, sr = self.audio_processor.load_audio_file(voice_sample)
        
        # Create voice prompt
        x_vector_only = voice_sample_text is None or voice_sample_text.strip() == ""
        prompt = tts_model.create_voice_clone_prompt(
            ref_audio=(wav_np, sr),
            ref_text=voice_sample_text if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )
        
        # Generate audio
        return await self._generate_with_prompt(
            tts_model, segments, prompt, instruct, language, **gen_kwargs
        )
    
    async def _synthesize_with_saved_voice(
        self,
        segments: List[Tuple[str, str]],
        voice_id: str,
        model: str,
        instruct: Optional[str],
        language: str,
        **gen_kwargs
    ) -> np.ndarray:
        """Synthesize using a saved voice."""
        # Get voice prompt
        voice_prompt = self._get_voice_prompt(voice_id)
        
        if voice_prompt is None:
            raise ValueError(f"Voice not found: {voice_id}")
        
        # Get model (use voice's model if available)
        voice_model = voice_prompt.model_name or model
        tts_model = model_manager.get_model(voice_model)
        
        # Generate audio
        return await self._generate_with_prompt(
            tts_model, segments, voice_prompt.prompt_data, instruct, language, **gen_kwargs
        )
    
    async def _generate_with_prompt(
        self,
        model,
        segments: List[Tuple[str, str]],
        prompt: Any,
        instruct: Optional[str],
        language: str,
        **gen_kwargs
    ) -> np.ndarray:
        """Generate audio using a voice prompt."""
        audio_results = []
        sr = settings.DEFAULT_SAMPLE_RATE
        
        for seg_type, content in segments:
            if seg_type == "pause":
                if content > 0:
                    silence_samples = int(content * sr)
                    audio_results.append(np.zeros(silence_samples, dtype=np.float32))
            else:
                try:
                    wavs, current_sr = model.generate_voice_clone(
                        text=[content],
                        language=[language],
                        voice_clone_prompt=prompt,
                        instruct=[instruct] if instruct else None,
                        **self._pack_gen_kwargs(gen_kwargs),
                    )
                    sr = current_sr
                    wav = wavs[0]
                    if wav.ndim > 1:
                        wav = wav.squeeze()
                    audio_results.append(wav)
                except Exception as e:
                    logger.error(f"Error generating segment '{content}': {e}")
                    raise
        
        if not audio_results:
            raise RuntimeError("No audio generated")
        
        return np.concatenate(audio_results)
    
    def _get_voice_prompt(self, voice_id: str) -> Optional[VoicePrompt]:
        """Get a voice prompt from cache or disk."""
        # Check cache
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]
        
        # Load from disk
        voice_path = self.voice_storage / f"{voice_id}.pt"
        if voice_path.exists():
            voice_prompt = VoicePrompt.load(voice_path)
            self._voice_cache[voice_id] = voice_prompt
            return voice_prompt
        
        return None
    
    def _process_instructions(self, instructions: Optional[str]) -> Optional[str]:
        """Process emotion/style instructions."""
        if not instructions:
            return None
        
        instructions = instructions.strip()
        
        # Check for emotion keywords
        if instructions.lower() in EMOTION_MAP:
            return EMOTION_MAP[instructions.lower()]
        
        # Check Chinese emotions
        if instructions in EMOTION_MAP:
            return EMOTION_MAP[instructions]
        
        # Return as-is
        return instructions
    
    def _pack_gen_kwargs(self, kwargs: dict) -> dict:
        """Pack generation parameters."""
        keys = [
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "subtalker_temperature",
            "subtalker_top_p",
            "subtalker_top_k",
        ]
        
        result = {}
        for key in keys:
            if key in kwargs:
                result[key] = kwargs[key]
        
        result["subtalker_dosample"] = True
        result["do_sample"] = True
        
        return result
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List all saved voices."""
        voices = []
        
        for meta_file in self.voice_storage.glob("*.json"):
            try:
                import json
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    voices.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading voice metadata {meta_file}: {e}")
        
        return sorted(voices, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get voice metadata."""
        meta_path = self.voice_storage / f"{voice_id}.json"
        
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice."""
        # Remove from cache
        if voice_id in self._voice_cache:
            del self._voice_cache[voice_id]
        
        # Remove files
        pt_path = self.voice_storage / f"{voice_id}.pt"
        json_path = self.voice_storage / f"{voice_id}.json"
        
        deleted = False
        for path in [pt_path, json_path]:
            if path.exists():
                path.unlink()
                deleted = True
        
        return deleted
    
    def list_preset_speakers(self) -> List[Dict[str, str]]:
        """List available preset speakers."""
        speakers = []
        
        for speaker_id, description in PRESET_SPEAKERS.items():
            # Extract language from description
            lang_match = re.search(r'\((\w+)', description)
            language = lang_match.group(1) if lang_match else "Unknown"
            
            speakers.append({
                "id": speaker_id,
                "name": speaker_id,
                "description": description,
                "language": language,
            })
        
        return speakers


# Global TTS service instance
tts_service = TTSService()
