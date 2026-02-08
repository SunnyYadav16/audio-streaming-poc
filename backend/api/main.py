"""
FastAPI WebSocket server for audio streaming with VAD.
Receives audio chunks from browser, performs voice activity detection,
and saves recordings as WAV files.
"""
import os
import io
import wave
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import av
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.vad import get_vad_service, SpeechSegmentDetector

app = FastAPI(title="Audio Streaming API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for saved recordings
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# VAD configuration
VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # ~32ms at 16kHz


class AudioStreamDecoder:
    """Decodes WebM/Opus audio stream in real-time."""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.buffer = b''
        self.audio_buffer = np.array([], dtype=np.float32)
        self.initialized = False
    
    def add_chunk(self, data: bytes) -> np.ndarray:
        """
        Add WebM chunk and return decoded PCM samples.
        Returns empty array if not enough data to decode.
        """
        self.buffer += data
        
        try:
            # Try to decode the accumulated buffer
            input_buffer = io.BytesIO(self.buffer)
            container = av.open(input_buffer, format='webm')
            
            samples = []
            for stream in container.streams:
                if stream.type == 'audio':
                    for frame in container.decode(stream):
                        # Get audio as float32 numpy array
                        audio = frame.to_ndarray()
                        
                        # Convert stereo to mono if needed
                        if audio.shape[0] > 1:
                            audio = audio.mean(axis=0)
                        else:
                            audio = audio[0]
                        
                        samples.append(audio)
            
            container.close()
            
            if samples:
                self.initialized = True
                audio_data = np.concatenate(samples).astype(np.float32)
                
                # Resample from 48kHz to 16kHz if needed
                if self.target_sample_rate != 48000:
                    # Simple decimation (48000 / 16000 = 3)
                    audio_data = audio_data[::3]
                
                return audio_data
                
        except Exception as e:
            # Not enough data yet to decode, or error
            pass
        
        return np.array([], dtype=np.float32)


class AudioSession:
    """Manages a single audio recording session with VAD."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chunks: list[bytes] = []
        self.started_at = datetime.now()
        
        # VAD components
        self.vad_service = get_vad_service()
        self.vad_service.reset_states()
        self.segment_detector = SpeechSegmentDetector(
            silence_threshold_ms=500,
            sample_rate=VAD_SAMPLE_RATE,
            chunk_size=VAD_CHUNK_SIZE
        )
        
        # Audio processing
        self.decoder = AudioStreamDecoder(target_sample_rate=VAD_SAMPLE_RATE)
        self.pcm_buffer = np.array([], dtype=np.float32)
    
    def add_chunk(self, data: bytes):
        """Add an audio chunk to the session."""
        self.chunks.append(data)
    
    def process_for_vad(self, data: bytes) -> list[dict]:
        """
        Process audio chunk for VAD and return any speech events.
        """
        events = []
        
        # Decode WebM chunk to PCM
        pcm_samples = self.decoder.add_chunk(data)
        
        if len(pcm_samples) == 0:
            return events
        
        # Add to PCM buffer
        self.pcm_buffer = np.concatenate([self.pcm_buffer, pcm_samples])
        
        # Process complete VAD chunks
        while len(self.pcm_buffer) >= VAD_CHUNK_SIZE:
            chunk = self.pcm_buffer[:VAD_CHUNK_SIZE]
            self.pcm_buffer = self.pcm_buffer[VAD_CHUNK_SIZE:]
            
            # Run VAD on chunk
            speech_prob, is_speech = self.vad_service.process_chunk(
                chunk, 
                sample_rate=VAD_SAMPLE_RATE
            )
            
            # Check for speech boundary events
            event = self.segment_detector.update(is_speech)
            if event['type']:
                events.append(event)
        
        return events
    
    def get_webm_data(self) -> bytes:
        """Combine all chunks into a single WebM blob."""
        return b''.join(self.chunks)
    
    def save_as_wav(self) -> Optional[Path]:
        """Convert WebM audio to WAV and save to disk."""
        if not self.chunks:
            return None
        
        webm_data = self.get_webm_data()
        output_path = RECORDINGS_DIR / f"{self.session_id}.wav"
        
        try:
            input_buffer = io.BytesIO(webm_data)
            container = av.open(input_buffer, format='webm')
            
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            
            samples = []
            for frame in container.decode(audio_stream):
                frame = frame.to_ndarray()
                samples.append(frame)
            
            container.close()
            
            if not samples:
                return None
            
            audio_data = np.concatenate(samples, axis=1)
            
            if audio_data.shape[0] > 1:
                audio_data = audio_data.mean(axis=0)
            else:
                audio_data = audio_data[0]
            
            audio_data = (audio_data * 32767).astype(np.int16)
            
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_data.tobytes())
            
            return output_path
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            debug_path = RECORDINGS_DIR / f"{self.session_id}.webm"
            debug_path.write_bytes(webm_data)
            print(f"Saved raw WebM to {debug_path}")
            return None


# Active sessions
sessions: dict[str, AudioSession] = {}


@app.on_event("startup")
async def startup_event():
    """Load VAD model on startup."""
    print("[API] Loading VAD model...")
    get_vad_service()
    print("[API] VAD model ready")


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """WebSocket endpoint for receiving audio streams with VAD."""
    await websocket.accept()
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = AudioSession(session_id)
    sessions[session_id] = session
    
    print(f"[{session_id}] Client connected")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            session.add_chunk(data)
            
            # Process for VAD and get events
            events = session.process_for_vad(data)
            
            for event in events:
                if event['type'] == 'speech_start':
                    print(f"[{session_id}] 🎤 Speech started")
                elif event['type'] == 'speech_end':
                    print(f"[{session_id}] 🔇 Speech ended (duration: {event['duration']}s)")
            
    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session.chunks:
            output_path = session.save_as_wav()
            if output_path:
                print(f"[{session_id}] Saved recording to {output_path}")
            else:
                print(f"[{session_id}] Failed to save recording")
        
        sessions.pop(session_id, None)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Audio Streaming API with VAD"}


@app.get("/recordings")
async def list_recordings():
    """List all saved recordings."""
    files = list(RECORDINGS_DIR.glob("*.wav"))
    return {
        "recordings": [
            {"name": f.name, "size": f.stat().st_size}
            for f in sorted(files, reverse=True)
        ]
    }
