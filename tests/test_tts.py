"""
TTS service tests.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.services.audio_utils import AudioProcessor


class TestAudioProcessor:
    """Test AudioProcessor utility class."""
    
    def setup_method(self):
        self.processor = AudioProcessor()
    
    def test_parse_text_with_pauses(self):
        """Test text pause parsing."""
        text = "Hello! [pause:0.5] How are you? [pause:1.0] I'm fine."
        segments = self.processor.parse_text_with_pauses(text)
        
        assert len(segments) == 5
        assert segments[0] == ("text", "Hello!")
        assert segments[1] == ("pause", 0.5)
        assert segments[2] == ("text", "How are you?")
        assert segments[3] == ("pause", 1.0)
        assert segments[4] == ("text", "I'm fine.")
    
    def test_parse_text_no_pauses(self):
        """Test text without pauses."""
        text = "Hello world!"
        segments = self.processor.parse_text_with_pauses(text)
        
        assert len(segments) == 1
        assert segments[0] == ("text", "Hello world!")
    
    def test_apply_fade(self):
        """Test fade in/out application."""
        audio = np.ones(10000, dtype=np.float32)
        sr = 24000
        
        result = self.processor._apply_fade(audio, sr, 10, 10)
        
        # Check that fade was applied
        assert result[0] < 1.0  # Fade in starts low
        assert result[-1] < 1.0  # Fade out ends low
        assert result[5000] == 1.0  # Middle should be full volume
    
    def test_adjust_speed_no_change(self):
        """Test speed adjustment with speed=1.0."""
        audio = np.random.randn(10000).astype(np.float32)
        sr = 24000
        
        result = self.processor.adjust_speed(audio, sr, 1.0)
        
        np.testing.assert_array_equal(audio, result)


class TestConfig:
    """Test configuration settings."""
    
    def test_emotion_map(self):
        """Test emotion mappings exist."""
        from api.config import EMOTION_MAP
        
        assert "happy" in EMOTION_MAP
        assert "sad" in EMOTION_MAP
        assert "angry" in EMOTION_MAP
    
    def test_preset_speakers(self):
        """Test preset speakers exist."""
        from api.config import PRESET_SPEAKERS
        
        assert "Vivian" in PRESET_SPEAKERS
        assert "Ryan" in PRESET_SPEAKERS
        assert len(PRESET_SPEAKERS) == 9
    
    def test_model_repos(self):
        """Test model repositories configured."""
        from api.config import MODEL_REPOS
        
        assert "qwen3-tts-1.7b-base" in MODEL_REPOS
        assert "huggingface" in MODEL_REPOS["qwen3-tts-1.7b-base"]
        assert "modelscope" in MODEL_REPOS["qwen3-tts-1.7b-base"]
