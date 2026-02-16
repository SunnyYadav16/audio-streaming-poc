"""
FastAPI WebSocket server for audio streaming with VAD.
Receives audio chunks from browser, performs voice activity detection,
and saves recordings as WAV files.

Supports two modes:
  - Solo mode (/ws/audio): single user, self-transcription & translation.
  - Conversation mode (/ws/session): two users with bidirectional translation.
"""
import os
import io
import json
import wave
import asyncio
import string
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force line-buffered stdout so print() output appears immediately in
# log files when running under nohup / systemd / Docker.
sys.stdout.reconfigure(line_buffering=True)

import av
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.vad import get_vad_service, SpeechSegmentDetector
from services.asr_service import get_asr_service
from services.mt_service import get_mt_service
from services.tts_service import get_tts_service
from services.priority_router import PriorityRouter
from services.session_recorder import (
    SessionRecorder,
    TranscriptLogger,
    write_manifest,
    SESSIONS_DIR,
)

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

    The browser sends small WebM/Opus chunks over the WebSocket.  This
    class buffers those bytes and uses PyAV to decode whatever complete
    frames are available, returning 16 kHz mono float32 PCM suitable for
    VAD.

    To prevent **O(N²) degradation** (re-decoding the entire session on
    every chunk), the client periodically restarts its MediaRecorder,
    which sends a new WebM EBML header.  When we detect that header we
    reset the buffer so the decode window stays bounded.
    """

    # First 4 bytes of the EBML header that starts every WebM file.
    _EBML_SIGNATURE = bytes([0x1A, 0x45, 0xDF, 0xA3])

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

        If *data* starts with a fresh EBML header (new MediaRecorder
        segment), the accumulated buffer is reset so the decode window
        stays small.
        """
        # ── Detect fresh WebM header → reset decoder ──
        if (
            len(data) >= 4
            and data[:4] == self._EBML_SIGNATURE
            and self.initialized
        ):
            self.buffer = b''
            self._samples_returned = 0

        self.buffer += data

        try:
            # Decode the accumulated buffer
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

        except Exception:
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


# Active sessions (solo mode)
sessions: dict[str, AudioSession] = {}


# ------------------------------------------------------------------ #
#  Conversation Room (Phase 6 – Bidirectional Flow)                   #
# ------------------------------------------------------------------ #

class Participant:
    """
    Represents one user inside a ConversationRoom.

    Wraps the WebSocket, display name, spoken language, and the
    per-connection AudioSession used for VAD / decoding.
    """

    def __init__(
        self,
        ws: WebSocket,
        name: str,
        language: str,
        session: AudioSession,
        role: str = "a",
    ):
        self.ws = ws
        self.name = name
        self.language = language          # language this user speaks
        self.session = session
        self.role = role                  # 'a' (creator) or 'b' (joiner)
        self.ws_open = True

    async def send_json_safe(self, payload: dict):
        """Send a JSON text frame, swallowing errors if the socket closed."""
        if not self.ws_open:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            self.ws_open = False

    async def send_bytes_safe(self, data: bytes):
        """Send a binary frame, swallowing errors if the socket closed."""
        if not self.ws_open:
            return
        try:
            await self.ws.send_bytes(data)
        except Exception:
            self.ws_open = False


class ConversationRoom:
    """
    A conversation session between multiple participants.

    The room creator defines both languages up-front (e.g. English <-> Spanish).
    The creator is assigned ``language_a``; joiners are automatically
    assigned ``language_b``.  Supports up to ``max_participants`` users.
    """

    MAX_PARTICIPANTS = 6

    def __init__(self, room_id: str, language_a: str, language_b: str):
        self.room_id = room_id
        self.language_a = language_a   # creator's language
        self.language_b = language_b   # joiner's language
        self.participants: list[Participant] = []
        self.created_at = datetime.now()
        self.priority = PriorityRouter()  # Phase 11: priority routing
        # Phase 10: session recording & transcript
        self.recorder: Optional[SessionRecorder] = None
        self.transcript_logger: Optional[TranscriptLogger] = None
        self.session_start_time: Optional[datetime] = None
        self.session_name: Optional[str] = None
        self.session_active = False     # U.4: controlled by creator's session_start

    @property
    def is_full(self) -> bool:
        return len(self.participants) >= self.MAX_PARTICIPANTS

    def add_participant(self, participant: Participant):
        self.participants.append(participant)

    def remove_participant(self, participant: Participant):
        if participant in self.participants:
            self.participants.remove(participant)

    def get_partner(self, participant: Participant) -> Optional[Participant]:
        """Legacy helper — returns first other participant."""
        for p in self.participants:
            if p is not participant:
                return p
        return None

    def get_others(self, participant: Participant) -> list["Participant"]:
        """Return all participants except the given one."""
        return [p for p in self.participants if p is not participant]


# Active conversation rooms
conversation_rooms: dict[str, ConversationRoom] = {}

# Characters for room codes (excluding ambiguous: O/0, I/1/L)
_ROOM_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _generate_room_id() -> str:
    """Generate a short, human-friendly 6-character room code."""
    while True:
        code = "".join(random.choices(_ROOM_CHARS, k=6))
        if code not in conversation_rooms:
            return code


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

    print("[API] Loading TTS voices (Piper)...")
    get_tts_service()
    print("[API] TTS voices ready")

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

    # Optional TTS toggle (enabled by default when translation is active)
    tts_enabled = websocket.query_params.get("tts", "true").lower() != "false"

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = AudioSession(session_id, language=lang_param)
    sessions[session_id] = session
    asr_service = get_asr_service()
    mt_service = get_mt_service() if target_lang_param else None
    tts_service = get_tts_service() if (target_lang_param and tts_enabled) else None

    # Track background ASR tasks so we can clean up on disconnect
    background_tasks: list[asyncio.Task] = []
    # Track the current partial-ASR task (only one at a time)
    partial_task: Optional[asyncio.Task] = None
    # Monotonic counter to discard stale partial results
    utterance_id = 0
    ws_open = True

    # Accumulate TTS PCM across utterances for a single session-level file
    tts_pcm_chunks: list[bytes] = []
    tts_sample_rate: Optional[int] = None

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
        nonlocal tts_sample_rate
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

                # --- TTS (only for final transcripts with a translation) ---
                if (
                    msg_type == "transcript"
                    and tts_service is not None
                    and target_lang_param
                    and payload.get("translation")
                ):
                    tts_wav = await asyncio.to_thread(
                        tts_service.synthesize,
                        payload["translation"],
                        target_lang_param,
                    )
                    if tts_wav:
                        # Extract raw PCM from WAV and accumulate
                        wav_buf = io.BytesIO(tts_wav)
                        with wave.open(wav_buf, "rb") as wf:
                            if tts_sample_rate is None:
                                tts_sample_rate = wf.getframerate()
                            tts_pcm_chunks.append(wf.readframes(wf.getnframes()))
                        payload["has_tts"] = True

                if msg_type == "transcript":
                    log_msg = (
                        f"[{session_id}] Speech ended (duration: {duration}s) "
                        f"lang={source_lang} text='{text}'"
                    )
                    if "translation" in payload:
                        log_msg += f" -> [{target_lang_param}] '{payload['translation']}'"
                    if payload.get("has_tts"):
                        log_msg += f" [TTS buffered, {len(tts_pcm_chunks)} utterances total]"
                    print(log_msg)

                await _send_json_safe(payload)
        except Exception as e:
            print(f"[{session_id}] ASR/MT ({msg_type}) error: {e}")

    print(
        f"[{session_id}] Client connected "
        f"(language={lang_param or 'auto'}, target={target_lang_param or 'none'}, "
        f"tts={'on' if tts_service else 'off'})"
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

        # Save the full session recording (original audio)
        if session.chunks:
            output_path = session.save_as_wav()
            if output_path:
                print(f"[{session_id}] Saved recording to {output_path}")
            else:
                print(f"[{session_id}] Failed to save recording")

        # Save combined TTS audio as a single file for the entire session
        if tts_pcm_chunks and tts_sample_rate:
            tts_dir = RECORDINGS_DIR / "tts"
            tts_dir.mkdir(parents=True, exist_ok=True)
            tts_path = tts_dir / f"{session_id}_{target_lang_param or 'tts'}.wav"
            with wave.open(str(tts_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(tts_sample_rate)
                wf.writeframes(b"".join(tts_pcm_chunks))
            print(f"[{session_id}] Saved TTS audio to {tts_path}")

        sessions.pop(session_id, None)


# ------------------------------------------------------------------ #
#  Conversation session endpoint (Phase 6)                             #
# ------------------------------------------------------------------ #

@app.websocket("/ws/session")
async def session_websocket(websocket: WebSocket):
    """
    Bidirectional conversation WebSocket.

    Creating a room (no room_id)
    ----------------------------
    The creator defines the language pair for the whole session.

    Query params:
        name         – display name (default "User")
        my_lang      – creator's spoken language (en | es | pt)
        partner_lang – the other user's spoken language (en | es | pt)

    Joining a room (room_id provided)
    ---------------------------------
    The joiner's language is auto-assigned from the room config.
    No language selection needed — just code + name.

    Query params:
        room_id – 6-char room code
        name    – display name (default "User")

    Protocol (server → client)
    --------------------------
    JSON text frames:
        room_created       – you created a new room (includes room_id, languages)
        room_joined        – you joined an existing room (includes auto-assigned lang)
        partner_joined     – the other user connected
        partner_left       – the other user disconnected
        transcript         – final ASR result (speaker: "self" | "partner")
        transcript_partial – interim ASR result
        error              – something went wrong

    Binary frames:
        WAV audio (TTS of partner's translated speech)
    """
    await websocket.accept()

    VALID_LANGS = {"en", "es", "pt"}

    # --- parse query params -----------------------------------------------
    room_id_param = (websocket.query_params.get("room_id") or "").strip().upper()
    user_name = websocket.query_params.get("name", "User").strip() or "User"

    # --- create / join room -----------------------------------------------
    if room_id_param:
        # ---- JOINING an existing room ----
        room = conversation_rooms.get(room_id_param)
        if room is None:
            await websocket.send_json(
                {"type": "error", "message": f"Room {room_id_param} not found"}
            )
            await websocket.close()
            return
        if room.is_full:
            await websocket.send_json(
                {"type": "error", "message": f"Room {room_id_param} is full (max {room.MAX_PARTICIPANTS})"}
            )
            await websocket.close()
            return
        room_id = room_id_param
        # Auto-assign the joiner language
        user_lang = room.language_b
    else:
        # ---- CREATING a new room ----
        my_lang = (websocket.query_params.get("my_lang") or "en").strip().lower()
        partner_lang = (websocket.query_params.get("partner_lang") or "es").strip().lower()
        if my_lang not in VALID_LANGS:
            my_lang = "en"
        if partner_lang not in VALID_LANGS:
            partner_lang = "es"

        room_id = _generate_room_id()
        room = ConversationRoom(room_id, language_a=my_lang, language_b=partner_lang)
        conversation_rooms[room_id] = room
        user_lang = my_lang

    # --- set up participant -----------------------------------------------
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audio_session = AudioSession(session_id, language=user_lang)
    # Assign roles: a (creator), b, c, d, …
    _ROLES = "abcdefghij"
    role = _ROLES[len(room.participants)] if len(room.participants) < len(_ROLES) else f"p{len(room.participants)}"
    participant = Participant(
        websocket, user_name, user_lang, audio_session, role=role,
    )
    room.add_participant(participant)

    # Notify the new participant
    if len(room.participants) == 1:
        await participant.send_json_safe({
            "type": "room_created",
            "room_id": room_id,
            "user_name": user_name,
            "language": user_lang,
            "partner_language": room.language_b,
        })
        # U.5: session_status = waiting (only creator connected)
        await participant.send_json_safe({
            "type": "session_status",
            "status": "waiting",
        })
        print(
            f"[Room {room_id}] Created by {user_name} "
            f"({room.language_a} ↔ {room.language_b})"
        )
    else:
        others = room.get_others(participant)
        # Tell the joiner about their auto-assigned language & existing partners
        first_other = others[0] if others else None
        await participant.send_json_safe({
            "type": "room_joined",
            "room_id": room_id,
            "user_name": user_name,
            "language": user_lang,
            "partner_name": first_other.name if first_other else None,
            "partner_language": first_other.language if first_other else None,
        })
        # Tell all existing participants about the new joiner
        for other in others:
            if other.ws_open:
                await other.send_json_safe({
                    "type": "partner_joined",
                    "name": user_name,
                    "language": user_lang,
                })

        # U.5: session_status = ready (2+ connected, waiting for host)
        if len(room.participants) >= 2:
            for p in room.participants:
                if p.ws_open:
                    await p.send_json_safe({
                        "type": "session_status",
                        "status": "ready",
                    })

        print(
            f"[Room {room_id}] {user_name} joined as {user_lang} speaker – "
            f"{len(others)} other(s) in room"
        )

    # --- services ---------------------------------------------------------
    asr_service = get_asr_service()
    mt_service = get_mt_service()
    tts_service = get_tts_service()

    background_tasks: list[asyncio.Task] = []
    partial_task: Optional[asyncio.Task] = None
    utterance_id = 0

    # Phase 11 (A.2): serialise ASR calls to avoid thread-safety issues
    _asr_lock = asyncio.Lock()

    def _wav_duration_ms(wav_bytes: bytes) -> float:
        """Estimate duration of a WAV buffer in milliseconds."""
        try:
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                return (wf.getnframes() / wf.getframerate()) * 1000.0
        except Exception:
            return 2000.0  # safe fallback: assume 2 s

    # ------------------------------------------------------------------
    async def _process_speech(
        pcm: np.ndarray,
        msg_type: str,
        utt_id: int,
        duration: Optional[float] = None,
    ):
        """
        Run ASR → MT → TTS pipeline and route results to ALL other
        participants.  No turn-taking gating — concurrent speech is allowed.

        Priority:
          - Creator (role a) → tts_priority "high"  (interrupts others' TTS)
          - Everyone else    → tts_priority "normal"
        """
        try:
            # A.2: serialise ASR to avoid CTranslate2 thread-safety issues
            async with _asr_lock:
                text, used_lang = await asyncio.to_thread(
                    asr_service.transcribe, pcm, user_lang,
                )
            # Discard stale partials
            if msg_type == "transcript_partial" and utt_id != utterance_id:
                return
            if not text:
                return

            source_lang = used_lang or user_lang
            tts_priority = room.priority.get_priority(role)
            speaker_role = room.priority.get_speaker_role(role)

            # ---- payload for SELF (the speaker) ----
            self_payload: dict = {
                "type": msg_type,
                "speaker": "self",
                "speaker_role": speaker_role,
                "text": text,
                "language": source_lang,
            }
            if duration is not None:
                self_payload["duration"] = round(duration, 2)

            # ---- build payloads for each OTHER participant ----
            others = room.get_others(participant)
            first_translated: Optional[str] = None
            first_target_lang: Optional[str] = None
            tts_cache: dict[str, bytes] = {}  # lang -> wav bytes

            for other in others:
                if not other.ws_open:
                    continue

                target_lang = other.language
                other_payload: Optional[dict] = None
                tts_wav: bytes = b""

                if source_lang != target_lang and source_lang != "unknown":
                    translated = await asyncio.to_thread(
                        mt_service.translate, text, source_lang, target_lang,
                    )
                    if translated:
                        # Attach translation to self payload (first time)
                        if first_translated is None:
                            first_translated = translated
                            first_target_lang = target_lang
                            self_payload["translation"] = translated
                            self_payload["target_language"] = target_lang

                        other_payload = {
                            "type": msg_type,
                            "speaker": "partner",
                            "speaker_name": user_name,
                            "speaker_role": speaker_role,
                            "text": text,
                            "language": source_lang,
                            "translation": translated,
                            "target_language": target_lang,
                            "tts_priority": tts_priority,
                        }
                        if duration is not None:
                            other_payload["duration"] = round(duration, 2)

                        # TTS only for final transcripts (cache per language)
                        if msg_type == "transcript":
                            if target_lang in tts_cache:
                                tts_wav = tts_cache[target_lang]
                            else:
                                tts_wav = await asyncio.to_thread(
                                    tts_service.synthesize, translated, target_lang,
                                )
                                tts_cache[target_lang] = tts_wav
                            if tts_wav:
                                other_payload["has_tts"] = True
                else:
                    # Same language – relay untranslated
                    other_payload = {
                        "type": msg_type,
                        "speaker": "partner",
                        "speaker_name": user_name,
                        "speaker_role": speaker_role,
                        "text": text,
                        "language": source_lang,
                        "tts_priority": tts_priority,
                    }
                    if duration is not None:
                        other_payload["duration"] = round(duration, 2)

                # ---- send to this recipient ----
                if other_payload:
                    # A.5: high-priority → send tts_interrupt before audio
                    if tts_wav and tts_priority == "high":
                        await other.send_json_safe({"type": "tts_interrupt"})

                    await other.send_json_safe(other_payload)
                    if tts_wav:
                        await other.send_bytes_safe(tts_wav)

            # ---- send to SELF ----
            await participant.send_json_safe(self_payload)

            # ---- Phase 10: record audio & log transcript ----
            if msg_type == "transcript" and room.recorder is not None:
                # Record original speaker PCM
                room.recorder.add_speech_pcm(role, pcm, 16_000)

                # Record TTS audio into target language track
                for lang, wav_data in tts_cache.items():
                    if wav_data:
                        room.recorder.add_tts_pcm(lang, wav_data)

                # Log transcript entry
                if room.transcript_logger is not None:
                    room.transcript_logger.add_entry(
                        role=role,
                        text=text,
                        language=source_lang,
                        translation=first_translated,
                        target_language=first_target_lang,
                        duration=duration,
                    )

            # ---- log ----
            if msg_type == "transcript":
                log_msg = (
                    f"[Room {room_id}] [{user_name}] "
                    f"'{text}' ({source_lang})"
                )
                if first_translated:
                    log_msg += (
                        f" → '{first_translated}' "
                        f"({first_target_lang})"
                    )
                tts_total = sum(len(w) for w in tts_cache.values() if w)
                if tts_total:
                    log_msg += f" [TTS {tts_total} bytes → {len(others)} recipient(s)]"
                print(log_msg)

        except Exception as e:
            print(f"[Room {room_id}] [{user_name}] ASR/MT error: {e}")
            import traceback
            traceback.print_exc()

    # ── Binary control markers (4-byte signals from client) ─────────────
    SESSION_START_MARKER = b'STRT'   # U.3/U.4: creator starts session
    SESSION_END_MARKER   = b'ENDS'   # U.6:     creator ends session
    MIC_MUTE_MARKER      = b'MUTE'   # U.7:     user muted mic
    MIC_UNMUTE_MARKER    = b'UNMT'   # U.7:     user unmuted mic

    async def _handle_session_start():
        """U.4: Creator starts the session — validate role A, broadcast."""
        if role != "a":
            print(f"[Room {room_id}] [{user_name}] ⚠ session_start rejected (role={role})")
            return
        if not room.is_full:
            await participant.send_json_safe({
                "type": "error",
                "message": "Cannot start session — partner has not joined yet.",
            })
            return
        room.session_active = True
        room.session_start_time = datetime.now()

        # Phase 10: build session name and initialise recorder + logger
        ts = room.session_start_time.strftime("%Y%m%d_%H%M%S")
        room.session_name = f"{ts}_{room_id}_{room.language_a}-{room.language_b}"

        # Resolve participant names
        name_a = "User A"
        name_b = "User B"
        for p in room.participants:
            if p.role == "a":
                name_a = p.name
            elif p.role == "b":
                name_b = p.name

        room.recorder = SessionRecorder(
            session_name=room.session_name,
            language_a=room.language_a,
            language_b=room.language_b,
            name_a=name_a,
            name_b=name_b,
        )
        room.transcript_logger = TranscriptLogger(
            session_name=room.session_name,
            language_a=room.language_a,
            language_b=room.language_b,
            name_a=name_a,
            name_b=name_b,
        )
        print(
            f"[Room {room_id}] 📼 Session recorder initialised: "
            f"{room.session_name}"
        )

        # Broadcast session_status = active to both
        for p in room.participants:
            if p.ws_open:
                await p.send_json_safe({
                    "type": "session_status",
                    "status": "active",
                })
        print(f"[Room {room_id}] ▶ Session started by {user_name}")

    async def _handle_session_end():
        """U.6: Creator ends the session — stop processing, notify both."""
        if role != "a":
            print(f"[Room {room_id}] [{user_name}] ⚠ session_end rejected (role={role})")
            return
        room.session_active = False

        # Phase 10: finalise session recording & transcript
        await _finalize_session_recording()

        # Broadcast session_status = ready (back to room view)
        for p in room.participants:
            if p.ws_open:
                await p.send_json_safe({
                    "type": "session_status",
                    "status": "ready",
                })
        print(f"[Room {room_id}] ⏹ Session ended by {user_name}")

    async def _handle_mic_mute(muted: bool):
        """U.7: Relay mute status to all other participants."""
        nonlocal utterance_id

        if muted:
            # If user was actively speaking, treat mute as implicit speech_end —
            # process whatever audio was buffered so the utterance isn't lost.
            if (
                audio_session.segment_detector.is_speaking
                and audio_session.current_utterance_pcm.size > 0
            ):
                utterance_pcm = audio_session.current_utterance_pcm.copy()
                audio_session.current_utterance_pcm = np.array([], dtype=np.float32)

                # Estimate duration from PCM length
                dur = round(len(utterance_pcm) / VAD_SAMPLE_RATE, 2)

                utterance_id += 1
                task = asyncio.create_task(
                    _process_speech(utterance_pcm, "transcript", utterance_id, dur)
                )
                background_tasks.append(task)

            # A.9: no floor release needed — there's no floor concept anymore
            # Reset VAD so no stale state carries over
            audio_session.segment_detector.reset()
            audio_session.vad_service.reset_states()
        else:
            # Reset on unmute for a clean start
            audio_session.segment_detector.reset()
            audio_session.vad_service.reset_states()
            audio_session.current_utterance_pcm = np.array([], dtype=np.float32)

        # Notify all other participants
        for other in room.get_others(participant):
            if other.ws_open:
                await other.send_json_safe({
                    "type": "partner_muted" if muted else "partner_unmuted",
                    "name": user_name,
                })
        print(f"[Room {room_id}] [{user_name}] {'🔇 Muted' if muted else '🔊 Unmuted'}")

    async def _finalize_session_recording():
        """Phase 10: write WAVs, transcript, and manifest to disk."""
        if room.recorder is None or room.session_name is None:
            return

        session_dir = SESSIONS_DIR / room.session_name

        try:
            # Wait for any in-flight ASR/MT/TTS tasks to complete
            pending = [t for t in background_tasks if not t.done()]
            if pending:
                print(
                    f"[Room {room_id}] ⏳ Waiting for {len(pending)} "
                    f"background tasks before finalising…"
                )
                await asyncio.gather(*pending, return_exceptions=True)

            # Write audio tracks
            wav_paths = room.recorder.finalize(session_dir)

            # Write transcript
            transcript_path = None
            if room.transcript_logger is not None:
                transcript_path = room.transcript_logger.finalize(session_dir)

            # Write manifest
            end_time = datetime.now()
            name_a = "User A"
            name_b = "User B"
            for p in room.participants:
                if p.role == "a":
                    name_a = p.name
                elif p.role == "b":
                    name_b = p.name

            write_manifest(
                session_dir=session_dir,
                session_name=room.session_name,
                start_time=room.session_start_time or datetime.now(),
                end_time=end_time,
                language_a=room.language_a,
                language_b=room.language_b,
                name_a=name_a,
                name_b=name_b,
                room_id=room_id,
                transcript_entries=(
                    room.transcript_logger.entry_count
                    if room.transcript_logger else 0
                ),
            )

            # Notify participants that artifacts are ready
            for p in room.participants:
                if p.ws_open:
                    await p.send_json_safe({
                        "type": "session_artifacts",
                        "session_name": room.session_name,
                        "download_url": f"/api/sessions/{room.session_name}/download",
                    })

            print(
                f"[Room {room_id}] 📼 Session artifacts saved to "
                f"{session_dir}"
            )
        except Exception as e:
            print(f"[Room {room_id}] ❌ Failed to finalise session recording: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up recorder references
            room.recorder = None
            room.transcript_logger = None

    # --- main receive loop ------------------------------------------------
    print(
        f"[Room {room_id}] [{user_name}] Connected "
        f"(role={role}, session_active={room.session_active})"
    )

    try:
        while True:
            data = await websocket.receive_bytes()

            # ── Control signals: 4-byte binary markers ──
            if len(data) == 4:
                if data == SESSION_START_MARKER:
                    await _handle_session_start()
                    continue
                elif data == SESSION_END_MARKER:
                    await _handle_session_end()
                    continue
                elif data == MIC_MUTE_MARKER:
                    await _handle_mic_mute(True)
                    continue
                elif data == MIC_UNMUTE_MARKER:
                    await _handle_mic_mute(False)
                    continue

            # ── Skip audio processing if session is not active ──
            if not room.session_active:
                continue

            audio_session.add_chunk(data)

            events = audio_session.process_for_vad(data)

            for event in events:
                if event["type"] == "speech_start":
                    # Phase 11: no turn-taking gate — always accept
                    utterance_id += 1
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        partial_task = None

                elif event["type"] == "speech_end":
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        partial_task = None

                    utterance_pcm = event.get("utterance_pcm")
                    dur = event.get("duration")
                    if utterance_pcm is not None and utterance_pcm.size > 0:
                        task = asyncio.create_task(
                            _process_speech(
                                utterance_pcm, "transcript", utterance_id, dur,
                            )
                        )
                        background_tasks.append(task)

            # Periodic partial transcript — no floor check, always allowed
            MIN_PARTIAL_SAMPLES = int(VAD_SAMPLE_RATE * 1.0)
            if (
                room.session_active
                and audio_session.segment_detector.is_speaking
                and audio_session.current_utterance_pcm.size >= MIN_PARTIAL_SAMPLES
                and (partial_task is None or partial_task.done())
            ):
                partial_task = asyncio.create_task(
                    _process_speech(
                        audio_session.current_utterance_pcm.copy(),
                        "transcript_partial",
                        utterance_id,
                    )
                )
                background_tasks.append(partial_task)

    except WebSocketDisconnect:
        print(f"[Room {room_id}] [{user_name}] Disconnected")
    except RuntimeError as e:
        # Starlette raises RuntimeError after a disconnect message is received
        print(f"[Room {room_id}] [{user_name}] Disconnected (ws closed)")
    except Exception as e:
        print(f"[Room {room_id}] [{user_name}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        participant.ws_open = False
        for t in background_tasks:
            if not t.done():
                t.cancel()

        # If session was active and this user leaving kills it, end session
        if room.session_active:
            room.session_active = False
            # Phase 10: finalise recording on disconnect
            await _finalize_session_recording()

        # Notify all remaining participants of departure
        for other in room.get_others(participant):
            if other.ws_open:
                await other.send_json_safe({
                    "type": "partner_left",
                    "name": user_name,
                })
                # If only 1 left after this departure, signal ended
                if len(room.participants) <= 2:
                    await other.send_json_safe({
                        "type": "session_status",
                        "status": "ended",
                    })

        room.remove_participant(participant)
        if not room.participants:
            conversation_rooms.pop(room_id, None)
            print(f"[Room {room_id}] Room closed (empty)")

        # Save recording
        if audio_session.chunks:
            output_path = audio_session.save_as_wav()
            if output_path:
                print(f"[Room {room_id}] [{user_name}] Saved to {output_path}")


# ------------------------------------------------------------------ #
#  REST endpoints                                                      #
# ------------------------------------------------------------------ #

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Audio Streaming API with VAD"}


@app.get("/rooms")
async def list_rooms():
    """List active conversation rooms (for debugging / lobby)."""
    return {
        "rooms": [
            {
                "room_id": r.room_id,
                "language_a": r.language_a,
                "language_b": r.language_b,
                "participants": [
                    {"name": p.name, "language": p.language}
                    for p in r.participants
                ],
                "is_full": r.is_full,
                "created_at": r.created_at.isoformat(),
            }
            for r in conversation_rooms.values()
        ]
    }


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


@app.get("/api/sessions/{session_name}/download")
async def download_session(session_name: str):
    """
    Download all session artifacts as a zip archive.

    Returns a zip containing the two language WAV files, the transcript
    text file, and the SHA-256 manifest JSON.
    """
    import zipfile
    from fastapi.responses import StreamingResponse

    session_dir = SESSIONS_DIR / session_name
    if not session_dir.exists() or not session_dir.is_dir():
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"error": f"Session '{session_name}' not found"},
        )

    # Collect all files in the session directory
    files = sorted(f for f in session_dir.iterdir() if f.is_file())
    if not files:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"error": f"Session '{session_name}' has no files"},
        )

    # Build zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f"{session_name}/{f.name}")
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{session_name}.zip"'
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    """List all saved session recordings."""
    if not SESSIONS_DIR.exists():
        return {"sessions": []}

    sessions_list = []
    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            manifest_files = list(d.glob("*_manifest.json"))
            manifest_data = None
            if manifest_files:
                try:
                    manifest_data = json.loads(
                        manifest_files[0].read_text(encoding="utf-8")
                    )
                except Exception:
                    pass

            sessions_list.append({
                "session_name": d.name,
                "files": [f.name for f in sorted(d.iterdir()) if f.is_file()],
                "manifest": manifest_data,
            })

    return {"sessions": sessions_list}
