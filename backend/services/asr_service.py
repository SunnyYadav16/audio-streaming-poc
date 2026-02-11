"""
ASR service using Faster-Whisper (large-v3-turbo).

This module provides a singleton-style ASRService that loads the
Faster-Whisper model once and exposes a simple `transcribe` method which
accepts in-memory PCM audio and returns a transcript plus language code.
"""

from typing import Optional, Tuple

import torch
import numpy as np
import os
from faster_whisper import WhisperModel


class ASRService:
    """
    Automatic Speech Recognition service powered by Faster-Whisper.

    The underlying WhisperModel is loaded once (large-v3-turbo) and reused
    across requests. Audio is expected to be 16 kHz mono float32 PCM in
    the [-1, 1] range, matching what the VAD pipeline already produces.
    """

    _instance = None
    _model: Optional[WhisperModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ASRService._model is None:
            self._load_model()

    def _load_model(self) -> None:
        """
        Load the Faster-Whisper model.

        large-v3-turbo is optimized for speed and quality. We explicitly
        select the device and compute type based on whether a CUDA GPU is
        available:
        - CUDA available  -> int8_float16 on GPU (fast, lower VRAM)
        - CPU only        -> int8 on CPU (good balance of speed/accuracy)
        """
        # This will download the model on first use and cache it under
        # ~/.cache/huggingface or a similar directory.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model_size = os.environ.get("WHISPER_MODEL", "small")

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        ASRService._model = model

    @property
    def model(self) -> WhisperModel:
        assert ASRService._model is not None, "ASR model not loaded"
        return ASRService._model

    def transcribe(
        self,
        audio_pcm: np.ndarray,
        language: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Transcribe a single utterance of PCM audio.

        Args:
            audio_pcm: 1D float32 numpy array of 16 kHz mono PCM samples
                normalized to [-1, 1].
            language: Optional BCP-47 language code ("en", "es", "pt", ...).
                If None, Faster-Whisper will attempt to detect the language.

        Returns:
            Tuple of (text, used_language):
                - text: the concatenated transcript for the utterance.
                - used_language: the language code chosen by the model
                  (detected or forced), if available.
        """
        if audio_pcm.size == 0:
            return "", language

        # Ensure correct dtype and 1D shape
        if not isinstance(audio_pcm, np.ndarray):
            audio_pcm = np.array(audio_pcm, dtype=np.float32)
        audio_pcm = audio_pcm.astype(np.float32).flatten()

        segments, info = self.model.transcribe(
            audio_pcm,
            language=language,
            task="transcribe",
            # beam_size can be tuned: 1–3 for lower latency, 5+ for quality.
            beam_size=3,
        )

        # Join all segment texts into a single utterance-level transcript
        pieces = [seg.text.strip() for seg in segments]
        text = " ".join(p for p in pieces if p)

        used_language: Optional[str] = language
        if getattr(info, "language", None):
            used_language = info.language

        return text.strip(), used_language


_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get or create the global ASR service instance."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service

