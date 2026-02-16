# Audio Streaming POC

Real-time browser-to-server audio streaming with speech recognition, translation, and text-to-speech — supporting multi-participant conversation sessions with priority-based routing.

## Features

- Browser audio capture with WebM/Opus encoding
- Real-time WebSocket streaming
- Voice activity detection (Silero VAD)
- Speech-to-text (Faster-Whisper)
- Machine translation (NLLB-200 via CTranslate2)
- Text-to-speech (Piper TTS)
- Live transcript and translation display
- **Conversation sessions** — multi-participant rooms with room codes
- **Session recording** — dual-language WAV tracks, per-turn transcript logs, and downloadable ZIP archives
- **Priority routing** — creator's speech always interrupts others' TTS playback
- **Concurrent speech** — all participants can speak simultaneously with independent ASR pipelines

## Architecture

```
Browser (x N)                         Server
+-----------------+                 +------------------------+
| MediaRecorder   |--- WebSocket -->| FastAPI                |
| (WebM/Opus)     |                 | PyAV Decoder           |
| Web Audio API   |<-- JSON/binary -| Silero VAD             |
| AudioContext    |                 | Faster-Whisper ASR     |
| (TTS playback)  |                 | NLLB-200 MT            |
| Priority Queue  |                 | Piper TTS              |
+-----------------+                 | PriorityRouter         |
                                    | SessionRecorder        |
                                    +------------------------+
```

### Conversation Pipeline

```
Participant A speaks ──> ASR ──> MT ──> TTS ──> broadcast to B, C, ... (priority: high)
Participant B speaks ──> ASR ──> MT ──> TTS ──> broadcast to A, C, ... (priority: normal)
Participant C speaks ──> ASR ──> MT ──> TTS ──> broadcast to A, B, ... (priority: normal)

All participants can speak simultaneously.
Creator (role A) has high priority — their TTS interrupts all in-progress playback.
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+
- pnpm

### Backend

```bash
cd backend
uv run uvicorn api.main:app --reload --port 8000
```

On first run, models are downloaded automatically:
- Faster-Whisper (ASR)
- NLLB-200-distilled-1.3B (translation, converted to CTranslate2 int8)
- Piper voice models for en_US, es_ES, pt_BR

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:5173. Choose **Solo** mode for self-transcription or **Conversation** to create/join a multi-participant room.

## Project Structure

```
audio-streaming-poc/
├── backend/
│   ├── api/
│   │   └── main.py                # FastAPI WebSocket server
│   ├── services/
│   │   ├── asr_service.py         # Faster-Whisper ASR
│   │   ├── mt_service.py          # NLLB-200 translation
│   │   ├── tts_service.py         # Piper TTS
│   │   ├── priority_router.py     # Role → TTS priority mapper
│   │   ├── session_recorder.py    # Dual-language WAV recording & transcript log
│   │   ├── turn_taking.py         # Legacy turn-taking (unused)
│   │   └── vad/
│   │       └── vad_service.py     # Silero VAD
│   ├── models/                    # Downloaded models (gitignored)
│   └── recordings/
│       ├── *.wav                  # Solo mode recordings
│       └── sessions/              # Conversation session archives
│           └── <session_name>/
│               ├── track_en.wav   # Language A audio track
│               ├── track_es.wav   # Language B audio track
│               ├── transcript.jsonl  # Per-turn transcript log
│               └── manifest.json  # SHA-256 checksums
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioRecorder.tsx       # Solo mode UI
│   │   │   └── ConversationSession.tsx # Multi-participant session UI
│   │   └── App.tsx
│   └── package.json
├── ARCHITECTURE.md
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, Python 3.11 |
| Audio | MediaRecorder API, PyAV |
| VAD | Silero VAD |
| ASR | Faster-Whisper (small) |
| Translation | NLLB-200-distilled-1.3B, CTranslate2 |
| TTS | Piper TTS (ONNX) |
| Transport | WebSocket (JSON + binary) |

## REST API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /rooms` | List active conversation rooms |
| `GET /recordings` | List solo mode recordings |
| `GET /api/sessions` | List saved session archives |
| `GET /api/sessions/{name}/download` | Download session ZIP (WAV tracks + transcript + manifest) |

## Conversation Sessions

### Creating a Room

1. Click **Conversation** on the landing page
2. Enter your name, select language pair (e.g. English ↔ Spanish)
3. Click **Create Room** — a 6-character room code is generated
4. Share the room code with participants (up to 10)

### Joining a Room

1. Enter your name and the room code
2. Click **Join** — language is auto-assigned based on room config

### Session Flow

- **Creator** clicks **Start Session** to begin recording and translation
- All participants can speak simultaneously — no turn-taking restrictions
- Creator's speech has **high priority** — TTS interrupts all recipients
- **Creator** clicks **End Session** to stop — recordings are finalized
- Muting still works: buffered speech is processed on mute

### Session Recording

When a session ends, the following are saved to `backend/recordings/sessions/<session_name>/`:
- `track_<lang>.wav` — 16kHz/16-bit mono WAV per language, silence-padded for alignment
- `transcript.jsonl` — one JSON object per turn with role, text, translation, timestamps
- `manifest.json` — SHA-256 checksums

### Priority Routing

| Speaker Role | TTS Priority | Behavior |
|---|---|---|
| Creator (role A) | `high` | `tts_interrupt` sent before audio; recipients' playback is stopped |
| Others (role B+) | `normal` | TTS queued normally on recipients |

On the frontend, high-priority TTS triggers a **"Court officer is speaking…"** animated banner.

## Phase Completion

### Phase 1: Audio Capture
- Browser microphone access
- MediaRecorder with WebM/Opus
- WebSocket streaming to server
- WAV file saving

### Phase 2: Voice Activity Detection
- Silero VAD integration
- Speech start/end detection
- 500ms silence threshold
- Client-side visual feedback

### Phase 3: Speech Recognition
- Faster-Whisper ASR (small model)
- Partial transcripts while speaking
- Final transcripts on utterance end
- Language auto-detection (en, es, pt)

### Phase 4: Machine Translation
- NLLB-200-distilled-1.3B via CTranslate2 (int8)
- Supports en, es, pt language pairs
- Translation runs in background threads
- Live translation display in UI

### Phase 5: Text-to-Speech
- Piper TTS with per-language voice models
- WAV audio sent as binary WebSocket frames
- Browser-side AudioContext playback queue
- Toggle to enable/disable TTS

### Phase 6: Conversation Sessions
- Room creation with 6-char codes
- Partner join via room code
- Bidirectional translation pipeline
- Per-participant audio sessions

### Phase 7: Session Controls & Echo Suppression
- Creator-controlled session start/end
- Mute/unmute with implicit speech_end processing
- Partner muted/unmuted UI indicators
- Barge-in TTS cancellation on own speech

### Phase 8: UI Polish
- Dark theme with glassmorphism sidebar
- Real-time chat bubbles with translation overlay
- Live partial transcript display
- Responsive layout for mobile

### Phase 9: Robustness
- Graceful WebSocket disconnect handling
- Connection cleanup on page unload
- Error recovery and state reset
- MediaRecorder periodic restart for stability

### Phase 10: Session Recording & Transcript Log
- Dual-language WAV recording (16kHz/16-bit mono)
- Silence padding for temporal alignment
- Per-turn JSONL transcript log
- SHA-256 manifest for integrity
- ZIP download via REST endpoint

### Phase 11: Independent Audio Channels with Priority Routing
- Removed strict turn-taking — concurrent speech supported
- Multi-participant rooms (up to 10)
- `asyncio.Lock` around ASR for thread safety
- Server-side echo suppression replaced with 300ms client-side cooldown
- Priority router: creator TTS always interrupts others
- `tts_interrupt` message + priority-aware frontend TTS queue
- "Court officer is speaking…" animated UI banner
