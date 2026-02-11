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
from services.asr_service import get_asr_service
from services.mt_service import get_mt_service

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
    """
    Incrementally decodes a WebM/Opus audio byte stream into PCM samples.

    The browser sends small WebM/Opus chunks over the WebSocket. This class
    buffers those bytes and uses PyAV to decode whatever complete frames are
    available, returning 16 kHz mono float32 PCM suitable for VAD.
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self.buffer = b''
        self.initialized = False
        # Track how many resampled samples have already been returned so
        # that each call only yields the *new* portion of the audio.
        self._samples_returned = 0
    
    def add_chunk(self, data: bytes) -> np.ndarray:
        """
        Add a WebM/Opus chunk and return only the **newly decoded** PCM
        samples since the last successful call.

        Because PyAV needs the full WebM byte stream to decode (header +
        data), we re-open the entire buffer each time but slice off the
        samples that were already returned in previous calls.
        """
        self.buffer += data
        
        try:
            # Decode the full accumulated buffer
            input_buffer = io.BytesIO(self.buffer)
            container = av.open(input_buffer, format='webm')
            
            samples = []
            for stream in container.streams:
                if stream.type == 'audio':
                    for frame in container.decode(stream):
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
                    audio_data = audio_data[::3]
                
                # Return only the new samples
                new_samples = audio_data[self._samples_returned:]
                self._samples_returned = len(audio_data)
                return new_samples
                
        except Exception as e:
            # Not enough data yet to decode, or error
            pass
        
        return np.array([], dtype=np.float32)


class AudioSession:
    """
    Manages the lifetime of a single browser audio session.

    Responsibilities:
    - Collect raw WebM chunks so the full session can be saved as a WAV file.
    - Decode incoming chunks to 16 kHz mono PCM via AudioStreamDecoder.
    - Run Silero VAD on fixed-size PCM windows.
    - Use SpeechSegmentDetector to turn per-chunk VAD decisions into
      higher-level "speech_start"/"speech_end" events.
    """
    
    def __init__(self, session_id: str, language: Optional[str] = None):
        self.session_id = session_id
        self.chunks: list[bytes] = []
        self.started_at = datetime.now()
        self.language = language
        
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
        # Buffer for the current utterance (between speech_start and speech_end)
        self.current_utterance_pcm = np.array([], dtype=np.float32)
    
    def add_chunk(self, data: bytes):
        """Add an audio chunk to the session."""
        self.chunks.append(data)
    
    def process_for_vad(self, data: bytes) -> list[dict]:
        """
        Decode an incoming WebM chunk, run VAD over complete PCM windows,
        and return speech boundary events.

        This method is intentionally fast (no ASR). ASR is run separately
        in background threads by the WebSocket handler so the receive loop
        is never blocked.
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

            # Accumulate PCM for the current utterance while speaking
            if self.segment_detector.is_speaking:
                if self.current_utterance_pcm.size == 0:
                    self.current_utterance_pcm = chunk.copy()
                else:
                    self.current_utterance_pcm = np.concatenate(
                        [self.current_utterance_pcm, chunk]
                    )

            # Reset utterance buffer at speech_start
            if event["type"] == "speech_start":
                self.current_utterance_pcm = chunk.copy()

            # On speech_end, attach the accumulated utterance PCM so the
            # WebSocket handler can transcribe it in a background thread.
            if event["type"] == "speech_end":
                event["utterance_pcm"] = self.current_utterance_pcm.copy()
                self.current_utterance_pcm = np.array([], dtype=np.float32)

            if event["type"]:
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
    """Load VAD, ASR, and MT models on startup."""
    print("[API] Loading VAD model...")
    get_vad_service()
    print("[API] VAD model ready")

    print("[API] Loading ASR model (Faster-Whisper)...")
    get_asr_service()
    print("[API] ASR model ready")

    print("[API] Loading MT model (NLLB-200 via CTranslate2)...")
    get_mt_service()
    print("[API] MT model ready")


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket):
    """WebSocket endpoint for receiving audio streams with VAD and ASR.

    VAD runs inline (fast). ASR runs in background threads so the receive
    loop is never blocked and audio flows continuously.
    """
    await websocket.accept()

    # Optional language selection from query params (en, es, pt or auto-detect)
    lang_param = websocket.query_params.get("lang")
    if lang_param not in {"en", "es", "pt"}:
        lang_param = None

    # Optional target language for machine translation
    target_lang_param = websocket.query_params.get("target_lang")
    if target_lang_param not in {"en", "es", "pt"}:
        target_lang_param = None

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = AudioSession(session_id, language=lang_param)
    sessions[session_id] = session
    asr_service = get_asr_service()
    mt_service = get_mt_service() if target_lang_param else None

    # Track background ASR tasks so we can clean up on disconnect
    background_tasks: list[asyncio.Task] = []
    # Track the current partial-ASR task (only one at a time)
    partial_task: Optional[asyncio.Task] = None
    # Monotonic counter to discard stale partial results
    utterance_id = 0
    ws_open = True

    async def _send_json_safe(payload: dict):
        """Send JSON to the browser, swallowing errors if the socket closed."""
        nonlocal ws_open
        if not ws_open:
            return
        try:
            await websocket.send_json(payload)
        except Exception:
            ws_open = False

    async def _run_asr_and_send(
        pcm: np.ndarray,
        msg_type: str,
        utt_id: int,
        duration: Optional[float] = None,
    ):
        """Run Whisper in a worker thread, optionally translate, and send."""
        try:
            text, used_lang = await asyncio.to_thread(
                asr_service.transcribe, pcm, lang_param,
            )
            # Discard stale partials (utterance already ended)
            if msg_type == "transcript_partial" and utt_id != utterance_id:
                return
            if text:
                source_lang = used_lang or lang_param or "unknown"
                payload: dict = {
                    "type": msg_type,
                    "session_id": session_id,
                    "text": text,
                    "language": source_lang,
                }
                if duration is not None:
                    payload["duration"] = duration

                # --- Machine Translation (if target language is set) ---
                if (
                    mt_service is not None
                    and target_lang_param
                    and source_lang != target_lang_param
                    and source_lang != "unknown"
                ):
                    translated = await asyncio.to_thread(
                        mt_service.translate,
                        text,
                        source_lang,
                        target_lang_param,
                    )
                    if translated:
                        payload["translation"] = translated
                        payload["target_language"] = target_lang_param

                if msg_type == "transcript":
                    log_msg = (
                        f"[{session_id}] Speech ended (duration: {duration}s) "
                        f"lang={source_lang} text='{text}'"
                    )
                    if "translation" in payload:
                        log_msg += f" -> [{target_lang_param}] '{payload['translation']}'"
                    print(log_msg)

                await _send_json_safe(payload)
        except Exception as e:
            print(f"[{session_id}] ASR/MT ({msg_type}) error: {e}")

    print(
        f"[{session_id}] Client connected "
        f"(language={lang_param or 'auto'}, target={target_lang_param or 'none'})"
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            session.add_chunk(data)

            # VAD processing — fast, never blocks
            events = session.process_for_vad(data)

            for event in events:
                if event["type"] == "speech_start":
                    print(f"[{session_id}] 🎤 Speech started")
                    utterance_id += 1
                    # Cancel any in-flight partial for the previous utterance
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        partial_task = None

                elif event["type"] == "speech_end":
                    # Cancel any in-flight partial
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        partial_task = None

                    utterance_pcm = event.get("utterance_pcm")
                    duration = event.get("duration")
                    if utterance_pcm is not None and utterance_pcm.size > 0:
                        task = asyncio.create_task(
                            _run_asr_and_send(
                                utterance_pcm, "transcript", utterance_id, duration
                            )
                        )
                        background_tasks.append(task)

            # --- Periodic partial transcript while user is speaking ---
            # Fire a partial only when:
            #   - the user is actively speaking
            #   - we have at least ~1 second of utterance audio
            #   - no other partial is already in flight
            MIN_PARTIAL_SAMPLES = int(VAD_SAMPLE_RATE * 1.0)  # 1 second
            if (
                session.segment_detector.is_speaking
                and session.current_utterance_pcm.size >= MIN_PARTIAL_SAMPLES
                and (partial_task is None or partial_task.done())
            ):
                partial_task = asyncio.create_task(
                    _run_asr_and_send(
                        session.current_utterance_pcm.copy(),
                        "transcript_partial",
                        utterance_id,
                    )
                )
                background_tasks.append(partial_task)

    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ws_open = False
        # Cancel any in-flight ASR tasks
        for t in background_tasks:
            if not t.done():
                t.cancel()
        # Save the full recording
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
