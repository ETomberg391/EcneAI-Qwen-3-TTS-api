"""
Model management service with lazy loading and caching.
"""

import os
import sys
import gc
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from api.config import settings, MODEL_REPOS

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages TTS model loading, caching, and lifecycle.
    
    Implements lazy loading - models are only loaded when first requested.
    Supports multiple model variants with automatic downloading.
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure single model manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = settings.device_str
        self.dtype = settings.torch_dtype
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager initialized: device={self.device}, dtype={self.dtype}")
    
    def get_model(self, model_name: str = None, unload_others: bool = True) -> Any:
        """
        Get or load a TTS model.
        
        Args:
            model_name: Name of the model to load (e.g., "qwen3-tts-1.7b-base")
            unload_others: If True, unload other models to free VRAM
            
        Returns:
            Loaded model instance
        """
        if model_name is None:
            model_name = settings.DEFAULT_MODEL
        
        # Normalize model name
        model_name = model_name.lower().strip()
        
        # Return cached model if available
        if model_name in self._models:
            logger.debug(f"Using cached model: {model_name}")
            return self._models[model_name]
        
        # Unload other models to free VRAM before loading new one
        if unload_others and self._models:
            logger.info(f"Unloading other models to free VRAM for {model_name}")
            self.unload_all_models(except_model=model_name)
        
        # Load the model
        logger.info(f"Loading model: {model_name}")
        model = self._load_model(model_name)
        self._models[model_name] = model
        
        return model
    
    def unload_model(self, model_name: str):
        """Unload a specific model and free its VRAM."""
        if model_name in self._models:
            logger.info(f"Unloading model: {model_name}")
            model = self._models[model_name]
            
            # Move to CPU first to free GPU memory
            if hasattr(model, 'to'):
                model.to('cpu')
            
            # Delete model reference
            del self._models[model_name]
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Model {model_name} unloaded, VRAM freed")
    
    def unload_all_models(self, except_model: str = None):
        """Unload all models except optionally one."""
        models_to_unload = [
            name for name in list(self._models.keys())
            if name != except_model
        ]
        
        for model_name in models_to_unload:
            self.unload_model(model_name)
    
    def _load_model(self, model_name: str):
        """
        Load a TTS model from disk or download it.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Loaded Qwen3TTSModel instance
        """
        if model_name not in MODEL_REPOS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_REPOS.keys())}"
            )
        
        # Get model path
        model_path = self._get_model_path(model_name)
        
        # Import Qwen3TTSModel
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen-tts is not installed. Run: pip install qwen-tts>=0.0.4"
            )
        
        # Determine attention implementation
        attn_impl = settings.ATTN_MODE
        if attn_impl == "flash_attention_2":
            if self.device == "cpu" or self.dtype == torch.float32:
                logger.warning(
                    "Flash Attention 2 requires GPU + half precision. Falling back to SDPA."
                )
                attn_impl = "sdpa"
        elif attn_impl == "sage_attention":
            # Sage attention is handled via patching
            attn_impl = "sdpa"
        
        # Load model
        logger.info(f"Loading {model_name} from {model_path}")
        logger.info(f"Device: {self.device}, Dtype: {self.dtype}, Attn: {attn_impl}")
        
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=self.dtype,
                attn_implementation=attn_impl,
            )
            
            # Apply patches
            self._apply_patches(model)
            
            # Store model type for compatibility checking
            model.model_type_str = model_name
            
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Error loading model {model_name}: {e}")
    
    def _get_model_path(self, model_name: str) -> Path:
        """
        Get the local path for a model, downloading if necessary.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Path to local model directory
        """
        repo_id = MODEL_REPOS[model_name][settings.MODEL_SOURCE]
        
        # Create sanitized directory name
        local_name = repo_id.replace("/", "--")
        local_path = self.cache_dir / local_name
        
        # Check if model exists
        if local_path.exists() and any(local_path.iterdir()):
            logger.debug(f"Model found at {local_path}")
            return local_path
        
        # Download model if enabled
        if settings.ENABLE_AUTO_DOWNLOAD:
            logger.info(f"Model not found locally. Downloading {repo_id}...")
            self._download_model(repo_id, local_path)
            return local_path
        else:
            raise FileNotFoundError(
                f"Model not found at {local_path} and auto-download is disabled"
            )
    
    def _download_model(self, repo_id: str, local_path: Path):
        """
        Download model from HuggingFace or ModelScope.
        
        Args:
            repo_id: Repository identifier
            local_path: Local destination path
        """
        local_path.mkdir(parents=True, exist_ok=True)
        
        if settings.MODEL_SOURCE == "huggingface":
            self._download_from_huggingface(repo_id, local_path)
        else:
            self._download_from_modelscope(repo_id, local_path)
    
    def _download_from_huggingface(self, repo_id: str, local_path: Path):
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading from HuggingFace: {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                token=settings.HF_TOKEN,
            )
            logger.info(f"Downloaded to {local_path}")
            
        except ImportError:
            raise ImportError(
                "huggingface-hub not installed. Run: pip install huggingface-hub"
            )
        except Exception as e:
            logger.error(f"Error downloading from HuggingFace: {e}")
            raise
    
    def _download_from_modelscope(self, repo_id: str, local_path: Path):
        """Download from ModelScope."""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            logger.info(f"Downloading from ModelScope: {repo_id}")
            snapshot_download(
                model_id=repo_id,
                local_dir=str(local_path),
            )
            logger.info(f"Downloaded to {local_path}")
            
        except ImportError:
            raise ImportError(
                "modelscope not installed. Run: pip install modelscope"
            )
        except Exception as e:
            logger.error(f"Error downloading from ModelScope: {e}")
            raise
    
    def _apply_patches(self, model):
        """Apply stability patches to the model."""
        import types
        import numpy as np
        
        def _safe_normalize(self, audios):
            """Patched normalize method with better error handling."""
            if isinstance(audios, list):
                items = audios
            elif (
                isinstance(audios, tuple)
                and len(audios) == 2
                and isinstance(audios[0], np.ndarray)
            ):
                items = [audios]
            else:
                items = [audios]
            
            out = []
            for a in items:
                if a is None:
                    continue
                if isinstance(a, str):
                    try:
                        wav, sr = self._load_audio_to_np(a)
                        out.append([wav.astype(np.float32), int(sr)])
                    except Exception as e:
                        logger.error(f"Failed to load audio file {a}: {e}")
                elif (
                    isinstance(a, (tuple, list))
                    and len(a) == 2
                    and isinstance(a[0], np.ndarray)
                ):
                    out.append([a[0].astype(np.float32), int(a[1])])
            
            for i in range(len(out)):
                wav, sr = out[i][0], out[i][1]
                if wav.ndim > 1:
                    out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
            
            return out
        
        model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
        
        # Apply SageAttention if requested
        if settings.ATTN_MODE == "sage_attention":
            self._apply_sage_attention(model)
        
        logger.info("Model patches applied successfully")
    
    def _apply_sage_attention(self, model):
        """Apply SageAttention for faster inference."""
        try:
            from sageattention import sageattn
            
            def make_sage_forward(orig_forward):
                def sage_forward(*args, **kwargs):
                    if len(args) >= 3:
                        q, k, v = args[0], args[1], args[2]
                        return sageattn(
                            q, k, v,
                            is_causal=False,
                            attn_mask=kwargs.get("attention_mask"),
                        )
                    return orig_forward(*args, **kwargs)
                return sage_forward
            
            patched_count = 0
            for name, m in model.model.named_modules():
                if "Attention" in type(m).__name__ or "attn" in name.lower():
                    if hasattr(m, "forward"):
                        m.forward = make_sage_forward(m.forward)
                        patched_count += 1
            
            logger.info(f"SageAttention patched on {patched_count} modules")
            
        except ImportError:
            logger.warning("sageattention not installed. Using SDPA.")
        except Exception as e:
            logger.warning(f"SageAttention patch failed: {e}")
    
    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
        
        Args:
            model_name: Model identifier
        """
        if model_name in self._models:
            logger.info(f"Unloading model: {model_name}")
            del self._models[model_name]
            self._clear_memory()
    
    def unload_all(self):
        """Unload all models from memory."""
        logger.info("Unloading all models")
        self._models.clear()
        self._clear_memory()
    
    def _clear_memory(self):
        """Clear GPU/CPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass
    
    def list_loaded_models(self) -> list:
        """List currently loaded models."""
        return list(self._models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        return model_name in self._models


# Global model manager instance
model_manager = ModelManager()
