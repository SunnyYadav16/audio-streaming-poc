"""
Silero VAD Service for real-time voice activity detection.
Processes audio chunks and returns speech probability.
"""
import torch
import numpy as np
from typing import Tuple
from pathlib import Path


class VADService:
    """Voice Activity Detection using Silero VAD model."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to reuse loaded model."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if VADService._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        print("[VAD] Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        VADService._model = model
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils
        print("[VAD] Model loaded successfully")
    
    @property
    def model(self):
        return VADService._model
    
    def reset_states(self):
        """Reset model states for new audio stream."""
        self.model.reset_states()
    
    def process_chunk(
        self, 
        audio_chunk: np.ndarray, 
        sample_rate: int = 16000
    ) -> Tuple[float, bool]:
        """
        Process an audio chunk and return speech probability.
        
        Args:
            audio_chunk: Audio samples as float32 numpy array, normalized to [-1, 1]
            sample_rate: Sample rate (8000 or 16000 Hz)
            
        Returns:
            Tuple of (speech_probability, is_speech)
        """
        # Convert numpy to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk
        
        # Ensure 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, sample_rate).item()
        
        # Threshold for speech detection
        threshold = 0.5
        is_speech = speech_prob >= threshold
        
        return speech_prob, is_speech


class SpeechSegmentDetector:
    """
    Detects speech segments with silence-based boundary detection.
    Tracks when speech starts and ends based on VAD output.
    """
    
    def __init__(
        self,
        silence_threshold_ms: int = 500,
        sample_rate: int = 16000,
        chunk_size: int = 512
    ):
        """
        Args:
            silence_threshold_ms: Silence duration (ms) to mark end of utterance
            sample_rate: Audio sample rate
            chunk_size: Number of samples per chunk
        """
        self.silence_threshold_ms = silence_threshold_ms
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Calculate number of silent chunks needed
        chunk_duration_ms = (chunk_size / sample_rate) * 1000
        self.silence_chunks_threshold = int(silence_threshold_ms / chunk_duration_ms)
        
        # State tracking
        self.is_speaking = False
        self.silent_chunks = 0
        self.speech_start_time = None
        self.total_speech_chunks = 0
    
    def update(self, is_speech: bool) -> dict:
        """
        Update state with new VAD result.
        
        Args:
            is_speech: Whether current chunk contains speech
            
        Returns:
            Event dict with 'type' key: 'speech_start', 'speech_end', or None
        """
        event = {'type': None}
        
        if is_speech:
            self.silent_chunks = 0
            
            if not self.is_speaking:
                # Speech just started
                self.is_speaking = True
                self.speech_start_time = self.total_speech_chunks
                event = {
                    'type': 'speech_start',
                }
            
            self.total_speech_chunks += 1
            
        else:
            if self.is_speaking:
                self.silent_chunks += 1
                
                if self.silent_chunks >= self.silence_chunks_threshold:
                    # Speech ended (silence threshold reached)
                    speech_duration = (self.total_speech_chunks - self.speech_start_time) * \
                                     (self.chunk_size / self.sample_rate)
                    
                    event = {
                        'type': 'speech_end',
                        'duration': round(speech_duration, 2)
                    }
                    
                    self.is_speaking = False
                    self.silent_chunks = 0
        
        return event
    
    def reset(self):
        """Reset detector state for new session."""
        self.is_speaking = False
        self.silent_chunks = 0
        self.speech_start_time = None
        self.total_speech_chunks = 0


# Global VAD service instance
_vad_service: VADService = None


def get_vad_service() -> VADService:
    """Get or create the global VAD service instance."""
    global _vad_service
    if _vad_service is None:
        _vad_service = VADService()
    return _vad_service
