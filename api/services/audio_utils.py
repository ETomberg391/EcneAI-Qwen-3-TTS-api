"""
Audio processing utilities.
"""

import io
import os
import re
import tempfile
import logging
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Utility class for audio processing operations."""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def load_audio_file(self, file_bytes: bytes, file_name: str = "audio.wav") -> Tuple[np.ndarray, int]:
        """
        Load audio from uploaded file bytes.
        
        Args:
            file_bytes: Raw bytes of the audio file
            file_name: Original file name for format detection
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            
            # Write to temporary file
            suffix = Path(file_name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                # Load with soundfile
                audio, sr = sf.read(tmp_path, dtype='float32')
                
                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                
                # Ensure minimum length
                if audio.size < 1024:
                    audio = np.pad(audio, (0, 1024 - audio.size))
                
                return audio, sr
                
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except ImportError:
            raise ImportError("soundfile is required for audio loading")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise ValueError(f"Failed to load audio file: {e}")
    
    def save_audio_to_bytes(
        self,
        audio: np.ndarray,
        sr: int,
        format: str = "mp3",
        fade_in_ms: int = 10,
        fade_out_ms: int = 50
    ) -> bytes:
        """
        Save audio array to bytes in specified format.
        
        Args:
            audio: Audio array (numpy array)
            sr: Sample rate
            format: Output format (mp3, wav, ogg, opus)
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
            
        Returns:
            Audio file as bytes
        """
        # Apply fades
        audio = self._apply_fade(audio, sr, fade_in_ms, fade_out_ms)
        
        # Convert to tensor for consistent output
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Export based on format
        format = format.lower()
        
        if format == "wav":
            return self._export_wav(audio, sr)
        elif format == "mp3":
            return self._export_mp3(audio, sr)
        elif format in ["ogg", "opus"]:
            return self._export_ogg(audio, sr, format)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _apply_fade(
        self,
        audio: np.ndarray,
        sr: int,
        fade_in_ms: int,
        fade_out_ms: int
    ) -> np.ndarray:
        """Apply fade in/out to audio."""
        audio = audio.copy()
        
        # Fade in
        if fade_in_ms > 0:
            fade_samples = int((fade_in_ms / 1000.0) * sr)
            fade_samples = min(fade_samples, len(audio))
            fade_curve = np.linspace(0.0, 1.0, fade_samples)
            audio[:fade_samples] *= fade_curve
        
        # Fade out
        if fade_out_ms > 0:
            fade_samples = int((fade_out_ms / 1000.0) * sr)
            fade_samples = min(fade_samples, len(audio))
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            audio[-fade_samples:] *= fade_curve
        
        return audio
    
    def _export_wav(self, audio: torch.Tensor, sr: int) -> bytes:
        """Export audio to WAV format."""
        import soundfile as sf
        
        # Convert to numpy
        if audio.dim() > 1:
            audio_np = audio.squeeze().numpy()
        else:
            audio_np = audio.numpy()
        
        # Ensure float32
        audio_np = audio_np.astype(np.float32)
        
        # Write to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sr, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    def _export_mp3(self, audio: torch.Tensor, sr: int) -> bytes:
        """Export audio to MP3 format using pydub."""
        try:
            from pydub import AudioSegment
            
            # First export as WAV
            wav_bytes = self._export_wav(audio, sr)
            
            # Convert to MP3 using pydub
            audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            
            buffer = io.BytesIO()
            audio_segment.export(buffer, format='mp3', bitrate='192k')
            buffer.seek(0)
            return buffer.read()
            
        except ImportError:
            logger.warning("pydub not installed, falling back to WAV format")
            return self._export_wav(audio, sr)
        except Exception as e:
            logger.error(f"Error exporting MP3: {e}, falling back to WAV")
            return self._export_wav(audio, sr)
    
    def _export_ogg(self, audio: torch.Tensor, sr: int, format: str = "ogg") -> bytes:
        """Export audio to OGG/OPUS format using pydub."""
        try:
            from pydub import AudioSegment
            
            # First export as WAV
            wav_bytes = self._export_wav(audio, sr)
            
            # Convert to OGG
            audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            
            buffer = io.BytesIO()
            if format == "opus":
                audio_segment.export(buffer, format='opus')
            else:
                audio_segment.export(buffer, format='ogg')
            buffer.seek(0)
            return buffer.read()
            
        except ImportError:
            logger.warning("pydub not installed, falling back to WAV format")
            return self._export_wav(audio, sr)
        except Exception as e:
            logger.error(f"Error exporting OGG: {e}, falling back to WAV")
            return self._export_wav(audio, sr)
    
    def adjust_speed(
        self,
        audio: np.ndarray,
        sr: int,
        speed: float,
        method: str = "librosa"
    ) -> np.ndarray:
        """
        Adjust audio speed.
        
        Args:
            audio: Audio array
            sr: Sample rate
            speed: Speed multiplier (0.5 = half speed, 2.0 = double)
            method: Method to use (librosa, ffmpeg, resample)
            
        Returns:
            Speed-adjusted audio array
        """
        if speed == 1.0:
            return audio
        
        if method == "librosa":
            return self._adjust_speed_librosa(audio, sr, speed)
        elif method == "ffmpeg":
            return self._adjust_speed_ffmpeg(audio, sr, speed)
        elif method == "resample":
            return self._adjust_speed_resample(audio, sr, speed)
        else:
            raise ValueError(f"Unknown speed adjustment method: {method}")
    
    def _adjust_speed_librosa(self, audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
        """Adjust speed using librosa time stretch."""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed, n_fft=4096)
        except ImportError:
            logger.warning("librosa not installed, using resample method")
            return self._adjust_speed_resample(audio, sr, speed)
    
    def _adjust_speed_ffmpeg(self, audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
        """Adjust speed using FFmpeg atempo filter."""
        try:
            import ffmpeg
            import soundfile as sf
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                sf.write(f_in.name, audio, sr)
                input_path = f_in.name
            
            output_path = input_path.replace(".wav", "_speed.wav")
            
            try:
                # Build FFmpeg command with multiple atempo filters if needed
                # (atempo range is 0.5 to 2.0)
                stream = ffmpeg.input(input_path)
                curr_speed = speed
                
                while curr_speed > 2.0:
                    stream = stream.filter("atempo", 2.0)
                    curr_speed /= 2.0
                while curr_speed < 0.5:
                    stream = stream.filter("atempo", 0.5)
                    curr_speed /= 0.5
                
                if abs(curr_speed - 1.0) > 0.01:
                    stream = stream.filter("atempo", curr_speed)
                
                stream.output(output_path).run(overwrite_output=True, quiet=True)
                
                # Read result
                result, new_sr = sf.read(output_path, dtype='float32')
                return result
                
            finally:
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
                        
        except ImportError:
            logger.warning("ffmpeg-python not installed, using librosa method")
            return self._adjust_speed_librosa(audio, sr, speed)
        except Exception as e:
            logger.error(f"FFmpeg error: {e}, falling back to librosa")
            return self._adjust_speed_librosa(audio, sr, speed)
    
    def _adjust_speed_resample(self, audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
        """Adjust speed by resampling (pitch changes)."""
        samples = audio.shape[0]
        new_samples = int(samples / speed)
        
        # Convert to tensor for interpolation
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        new_audio = F.interpolate(
            audio_tensor,
            size=new_samples,
            mode="linear",
            align_corners=False
        )
        
        return new_audio.squeeze().numpy()
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback to simple interpolation
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            resampled = F.interpolate(
                audio_tensor,
                size=new_length,
                mode="linear",
                align_corners=False
            )
            return resampled.squeeze().numpy()
    
    def parse_text_with_pauses(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse text for pause tags.
        
        Args:
            text: Input text with optional [pause:X] tags
            
        Returns:
            List of tuples (type, content) where type is 'text' or 'pause'
        """
        segments = []
        pause_pattern = re.compile(r"\[(?:pause|p):(\d+(?:\.\d+)?)\]", re.IGNORECASE)
        
        lines = [t.strip() for t in text.split("\n") if t.strip()]
        
        for line in lines:
            last_idx = 0
            for match in pause_pattern.finditer(line):
                text_part = line[last_idx:match.start()].strip()
                if text_part:
                    segments.append(("text", text_part))
                
                try:
                    duration = float(match.group(1))
                    segments.append(("pause", duration))
                except ValueError:
                    pass
                
                last_idx = match.end()
            
            remaining_text = line[last_idx:].strip()
            if remaining_text:
                segments.append(("text", remaining_text))
        
        if not segments:
            segments.append(("text", text))
        
        return segments
    
    def concatenate_audio(self, audio_segments: List[np.ndarray], sr: int) -> np.ndarray:
        """
        Concatenate multiple audio segments with pauses.
        
        Args:
            audio_segments: List of audio arrays and pause durations
            sr: Sample rate
            
        Returns:
            Concatenated audio array
        """
        valid_segments = []
        
        for segment in audio_segments:
            if isinstance(segment, np.ndarray) and segment.size > 0:
                valid_segments.append(segment)
        
        if not valid_segments:
            return np.zeros(1024, dtype=np.float32)
        
        return np.concatenate(valid_segments)
