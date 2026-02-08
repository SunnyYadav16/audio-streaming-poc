"""VAD service package."""
from .vad_service import VADService, SpeechSegmentDetector, get_vad_service

__all__ = ['VADService', 'SpeechSegmentDetector', 'get_vad_service']
