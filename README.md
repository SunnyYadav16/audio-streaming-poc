# Audio Streaming POC

Real-time browser-to-server audio streaming with speech recognition, translation, and text-to-speech.

## Features

- Browser audio capture with WebM/Opus encoding
- Real-time WebSocket streaming
- Voice activity detection (Silero VAD)
- Speech-to-text (Faster-Whisper)
- Machine translation (NLLB-200 via CTranslate2)
- Text-to-speech (Piper TTS)
- Live transcript and translation display

## Architecture

```
Browser                              Server
+-----------------+                 +--------------------+
| MediaRecorder   |--- WebSocket -->| FastAPI            |
| (WebM/Opus)     |                 | PyAV Decoder       |
| Web Audio API   |                 | Silero VAD         |
| AudioContext    |<-- JSON/binary --| Faster-Whisper ASR |
| (TTS playback)  |                 | NLLB-200 MT        |
+-----------------+                 | Piper TTS          |
                                    +--------------------+
```

Pipeline: Audio -> VAD -> ASR -> MT -> TTS -> Browser playback

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

Open http://localhost:5173 and click Start Recording.

## Project Structure

```
audio-streaming-poc/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI WebSocket server
│   ├── services/
│   │   ├── asr_service.py       # Faster-Whisper ASR
│   │   ├── mt_service.py        # NLLB-200 translation
│   │   ├── tts_service.py       # Piper TTS
│   │   └── vad/
│   │       └── vad_service.py   # Silero VAD
│   ├── models/                  # Downloaded models (gitignored)
│   ├── recordings/              # Saved WAV files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── AudioRecorder.tsx
│   │   └── App.tsx
│   └── package.json
└── ARCHITECTURE.md
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
