# 🎙️ Audio Streaming POC

Real-time browser-to-server audio streaming with Voice Activity Detection (VAD).

## ✨ Features

- **Browser Audio Capture** - MediaRecorder API with WebM/Opus encoding
- **Real-time Streaming** - WebSocket-based audio transmission
- **Server-side VAD** - Silero VAD for accurate speech detection
- **Client-side VAD** - Web Audio API for visual feedback
- **Speech Boundaries** - Automatic detection of utterance start/end
- **WAV Recording** - Recordings saved to disk for verification

## 🏗️ Architecture

```
Browser                          Server
┌─────────────────┐             ┌─────────────────┐
│ MediaRecorder   │─────────────│ FastAPI         │
│ (WebM/Opus)     │  WebSocket  │ WebSocket       │
├─────────────────┤             ├─────────────────┤
│ Web Audio API   │             │ PyAV Decoder    │
│ (Visual VAD)    │             │ (WebM → PCM)    │
└─────────────────┘             ├─────────────────┤
                                │ Silero VAD      │
                                │ (Speech Events) │
                                └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 22+ (use `nvm use 22`)
- pnpm

### Backend

```bash
cd backend
uv run uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
nvm use 22
pnpm install
pnpm dev
```

Open http://localhost:5173 and click **Start Recording**.

## 📁 Project Structure

```
audio-streaming-poc/
├── backend/
│   ├── api/
│   │   └── main.py           # FastAPI WebSocket server
│   ├── services/
│   │   └── vad/
│   │       └── vad_service.py # Silero VAD wrapper
│   ├── recordings/           # Saved WAV files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── AudioRecorder.tsx
│   │   └── lib/
│   │       └── useWebSocket.ts
│   └── package.json
└── docker-compose.yml
```

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, TypeScript, Vite |
| Backend | FastAPI, Python 3.11 |
| Audio | MediaRecorder API, PyAV |
| VAD | Silero VAD (PyTorch) |
| Transport | WebSocket |

## 📊 Phase Completion

### ✅ Phase 1: Audio Capture
- [x] Browser microphone access
- [x] MediaRecorder with WebM/Opus
- [x] WebSocket streaming to server
- [x] WAV file saving

### ✅ Phase 2: Voice Activity Detection
- [x] Silero VAD integration
- [x] Speech start/end detection
- [x] 500ms silence threshold
- [x] Client-side visual feedback
