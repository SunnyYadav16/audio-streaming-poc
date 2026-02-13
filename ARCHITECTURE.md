## Audio Streaming POC – Architecture Overview

This document explains how audio flows from the browser to the backend, how
Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), Machine
Translation (MT), and Text-to-Speech (TTS) are integrated, and where future
components (LLM, barge-in) will plug in.

### High-level flow

- **Frontend (React)**
  - Captures microphone audio using `getUserMedia`.
  - Encodes audio with `MediaRecorder` (WebM/Opus, ~250ms chunks).
  - Streams encoded chunks to the backend over a WebSocket.
  - Receives JSON messages (`transcript`, `transcript_partial`) back from the
    server over the same WebSocket and renders them in a live transcript panel.
  - Receives binary WebSocket frames containing TTS WAV audio and plays them
    back via `AudioContext` with a sequential queue.
  - Shows a client-side speaking indicator based on audio energy level.
  - Displays live translations alongside transcripts when a target language is
    selected.

- **Backend (FastAPI)**
  - Exposes a WebSocket endpoint at `/ws/audio`.
  - Receives WebM/Opus chunks from the browser.
  - Decodes chunks to 16 kHz mono PCM using PyAV (returning only **new**
    samples each call via a `_samples_returned` counter).
  - Runs Silero VAD on 512-sample PCM windows (~32 ms each).
  - Uses silence-based logic to detect speech start/end events.
  - On `speech_end`, runs Faster-Whisper ASR in a **background thread**
    (`asyncio.to_thread`) so the receive loop is never blocked.
  - After ASR, optionally translates the text via NLLB-200 (CTranslate2) when
    a `target_lang` is specified and differs from the source language.
  - After translation, optionally synthesizes speech via Piper TTS and sends
    the resulting WAV audio back as a binary WebSocket frame.
  - Sends partial and final transcripts (with translations) back to the browser
    as JSON.
  - Saves each WebSocket session as a `.wav` file for debugging/verification.

---

## 1. Frontend

### 1.1 Entry points

- `frontend/src/main.tsx` – React entrypoint, mounts `App` into `index.html`.
- `frontend/src/App.tsx` – Top-level UI component.
- `frontend/src/components/AudioRecorder.tsx` – Core audio capture, WebSocket
  client, and transcript display.

### 1.2 `AudioRecorder` responsibilities

- Request microphone access via
  `navigator.mediaDevices.getUserMedia({ audio: { echoCancellation, noiseSuppression, sampleRate } })`.
- Establish a WebSocket connection to the backend:
  - Default URL: `ws://localhost:8000/ws/audio`, configurable via `serverUrl` prop.
  - Optionally appends `?lang=en|es|pt` when a specific language is selected.
- Create a `MediaRecorder` instance:
  - `mimeType: 'audio/webm;codecs=opus'`, `audioBitsPerSecond: 128000`.
  - `mediaRecorder.start(250)` to emit ~250 ms chunks.
- For each `ondataavailable` event:
  - If the socket is open and `event.data.size > 0`, call `ws.send(event.data)`.
- Listen for server messages via `ws.onmessage`:
  - `type === "transcript"` (final) – clear live text, append a
    `TranscriptEntry` (id, text, language, duration, timestamp) to the
    finalized list.
  - `type === "transcript_partial"` (interim) – update `liveTranscript` state
    shown in italic at the bottom of the panel.
- Track recording state (idle / connecting / recording / error) and a
  human-readable duration counter for the UI.

### 1.3 UI layout

Split-screen design:

- **Left panel** – recording controls, status ring with timer, speaking
  indicator, start/stop button, source language selector, target language
  selector, and TTS toggle.
- **Right panel** – scrollable transcript panel with finalized utterance entries
  (original text, language badge, duration, timestamp). When translation is
  active, translated text is shown below each entry. A faded partial-text line
  (with live translation) appears while the user is still speaking.
  Auto-scrolls as new entries arrive. An animated "Playing audio..." indicator
  appears when TTS audio is being played.

### 1.4 Client-side speaking indicator

- Uses the Web Audio API (`AnalyserNode`, `getByteFrequencyData`).
- Computes average frequency energy and compares to `speechThreshold`.
- A few consecutive "loud" frames set `isSpeaking = true`; falling below
  gradually decrements and resets.
- This is **not** Silero VAD — just a lightweight heuristic for visual feedback.
  It can be swapped for a neural VAD (e.g., `@ricky0123/vad-web`) later.

---

## 2. Backend

### 2.1 FastAPI application

File: `backend/api/main.py`

- Creates the FastAPI app with permissive CORS for local development.
- On startup, eagerly loads all models:
  - `get_vad_service()` – Silero VAD.
  - `get_asr_service()` – Faster-Whisper (model set via `WHISPER_MODEL` env
    var, default `small`).
  - `get_mt_service()` – NLLB-200 via CTranslate2.
  - `get_tts_service()` – Piper TTS voices for en, es, pt.
- Provides:
  - `GET /` – Health check.
  - `GET /recordings` – Lists saved `.wav` recordings.
  - `WS /ws/audio` – Main audio WebSocket endpoint (bidirectional).

### 2.2 WebSocket endpoint `/ws/audio`

Function: `audio_websocket`

Accepts optional query parameters:
- `?lang=en|es|pt` – force a transcription language (otherwise Faster-Whisper
  auto-detects).
- `?target_lang=en|es|pt` – target language for translation. If omitted or
  same as source, translation is skipped.
- `?tts=true|false` – enable TTS synthesis of translated text (default false).

For each client connection:

1. Accept the WebSocket and generate a `session_id`.
2. Create an `AudioSession` instance.
3. In a loop (**never blocked by ASR**):
   - `await websocket.receive_bytes()` to get the next encoded audio chunk.
   - `session.add_chunk(data)` to buffer for later `.wav` saving.
   - `session.process_for_vad(data)` to run VAD (fast, inline).
   - For each returned event:
     - `speech_start` – log; increment `utterance_id`; cancel any in-flight
       partial ASR task.
     - `speech_end` – fire a background task
       (`asyncio.create_task(_run_asr_and_send(...))`) that runs Whisper in a
       worker thread and sends the final transcript JSON back to the browser.
   - After processing events, if the user is still speaking and at least 1 s
     of audio has accumulated and no partial task is in flight, fire a
     background partial-ASR task for live interim text.
4. On disconnect or error:
   - Cancel all in-flight background tasks.
   - Convert accumulated WebM data to `.wav` and save under
     `backend/recordings/{session_id}.wav`.

#### Messages sent to the browser

**JSON (text frames):**

| `type`               | Fields                                                                                    | When                        |
|----------------------|-------------------------------------------------------------------------------------------|-----------------------------|
| `transcript`         | `session_id`, `text`, `language`, `duration`, `translation`, `target_language`, `has_tts_audio` | After each utterance ends   |
| `transcript_partial` | `session_id`, `text`, `language`, `translation`, `target_language`                        | Periodically while speaking |

- `translation` and `target_language` are present only when a target language is
  configured and differs from the source.
- `has_tts_audio` is `true` when TTS is enabled and a binary WAV frame follows.

**Binary (binary frames):**

- WAV audio generated by Piper TTS. Sent immediately after the corresponding
  `transcript` JSON message when `has_tts_audio` is true. The frontend
  distinguishes JSON vs binary by checking `instanceof ArrayBuffer`.

### 2.3 `AudioSession`

File: `backend/api/main.py`

Per-connection state holder:

- Raw WebM chunks (for `.wav` conversion).
- `VADService` instance (Silero VAD).
- `SpeechSegmentDetector` (silence-based segmentation).
- `AudioStreamDecoder` (WebM/Opus → 16 kHz PCM).
- `pcm_buffer` – rolling buffer of decoded samples awaiting VAD.
- `current_utterance_pcm` – accumulates PCM while the user is speaking; copied
  and attached to `speech_end` events for the handler to transcribe.

Key methods:

- `add_chunk(data)` – Append a raw WebM chunk to the session.
- `process_for_vad(data) -> list[dict]` – **VAD-only, no ASR.** Decodes
  audio, runs Silero VAD on 512-sample windows, accumulates utterance PCM,
  emits `speech_start` / `speech_end` events. On `speech_end`, attaches
  `event["utterance_pcm"]` (a numpy copy) for the WebSocket handler to
  transcribe in a background thread.
- `save_as_wav()` – Convert all accumulated WebM chunks to a mono 48 kHz
  `.wav` file. Falls back to saving raw `.webm` on error.

### 2.4 `AudioStreamDecoder`

File: `backend/api/main.py`

Converts the incremental WebM/Opus byte stream into decoded PCM samples:

- Maintains an internal `buffer` of raw bytes and a `_samples_returned`
  counter.
- On each `add_chunk(data)`:
  - Appends `data` to the byte buffer.
  - Re-opens the full buffer with PyAV (`av.open`, format `"webm"`) and
    decodes all audio frames to `float32` numpy arrays.
  - Mixes down to mono if needed.
  - Resamples from 48 kHz to 16 kHz by simple decimation (`audio_data[::3]`).
  - **Returns only the new samples** by slicing at `_samples_returned` and
    updating the counter. This prevents the VAD from seeing duplicate audio
    (critical for correct Silero recurrent state).
- Returns an empty array if decoding is not yet possible.

---

## 3. VAD and speech segmentation

### 3.1 `VADService` (Silero VAD)

File: `backend/services/vad/vad_service.py`

- Loads the Silero VAD model once via `torch.hub.load`.
- Singleton via `get_vad_service()`.
- `reset_states()` – Reset recurrent model state between sessions.
- `process_chunk(audio_chunk, sample_rate=16000) -> (speech_prob, is_speech)`:
  - Accepts 1D float32 numpy array in \[-1, 1\].
  - Returns `speech_prob` (float) and `is_speech` (bool, threshold 0.5).

### 3.2 `SpeechSegmentDetector`

File: `backend/services/vad/vad_service.py`

Converts per-chunk `is_speech` flags into higher-level speech segments:

- Configured with `silence_threshold_ms=500`, `sample_rate=16000`,
  `chunk_size=512`.
- Computes `silence_chunks_threshold ≈ 15` chunks (~480 ms).
- State machine:
  - `is_speech=True` and not speaking → emit `speech_start`, set
    `is_speaking=True`.
  - `is_speech=False` while speaking → increment `silent_chunks`; when
    `>= silence_chunks_threshold` → emit `speech_end` with duration, reset.
- `update(is_speech) -> dict` with `type`: `"speech_start"`, `"speech_end"`,
  or `None`.

---

## 4. ASR (speech-to-text) — Phase 3, implemented

### 4.1 `ASRService` (Faster-Whisper)

File: `backend/services/asr_service.py`

- Singleton loaded on startup via `get_asr_service()`.
- Uses `WhisperModel` with explicit device selection:
  - CUDA available → `device="cuda"`, `compute_type="int8_float16"`.
  - CPU only → `device="cpu"`, `compute_type="int8"`.
- Model defaults to `small` (configurable via `WHISPER_MODEL` env var).
- `transcribe(audio_pcm, language=None) -> (text, used_language)`:
  - Accepts 1D float32 16 kHz mono PCM.
  - If `language` is provided, skips auto-detection (faster and more stable
    for short utterances).
  - Returns concatenated segment text and the language code.
  - `beam_size=3` for a good latency/quality balance.

### 4.2 How ASR integrates with the pipeline

ASR is **not** called inside `process_for_vad` (that would block the event
loop). Instead, the WebSocket handler:

1. Receives `speech_end` events with attached `utterance_pcm`.
2. Fires `asyncio.create_task(asyncio.to_thread(asr_service.transcribe, ...))`.
3. When the thread completes, sends JSON (`type: "transcript"`) back to the
   browser.
4. For interim results while the user is still speaking, fires periodic partial
   ASR tasks (`type: "transcript_partial"`) — at most one at a time, with a
   monotonic `utterance_id` to discard stale results.

This keeps the receive loop fast (~1 ms per chunk for VAD) and audio flowing
continuously.

### 4.3 Language selection

- The frontend language selector sets `?lang=en|es|pt` on the WebSocket URL.
- The backend passes this to `asr_service.transcribe(language=...)`.
- If set to "Auto-detect", `language=None` is passed and Faster-Whisper
  detects the language, returning it in the response.

---

## 5. Machine Translation — Phase 4, implemented

### 5.1 `MTService` (NLLB-200 via CTranslate2)

File: `backend/services/mt_service.py`

- Singleton loaded on startup via `get_mt_service()`.
- Uses the NLLB-200-distilled-1.3B model, converted to CTranslate2 format with
  int8 quantization for efficient CPU inference.
- On first load, downloads the Hugging Face model and runs
  `ct2-transformers-converter` to produce the CTranslate2 model directory under
  `backend/models/nllb-200-distilled-1.3B-ct2-int8/`.
- Maintains a language code mapping from ISO 639-1 (`en`, `es`, `pt`) to
  NLLB Flores-200 codes (`eng_Latn`, `spa_Latn`, `por_Latn`).
- `translate(text, source_lang, target_lang) -> str`:
  - Tokenizes input with the NLLB tokenizer, setting the source language as the
    `src_lang`.
  - Translates via `ctranslate2.Translator` with a target prefix token for the
    target language.
  - Filters special tokens (`</s>`, `<unk>`) before decoding the output.

### 5.2 How MT integrates with the pipeline

After ASR produces a transcript (partial or final), if a `target_lang` is
configured and differs from the detected source language:

1. `mt_service.translate(text, source_lang, target_lang)` runs in
   `asyncio.to_thread`.
2. The resulting translation is included in the JSON response alongside the
   original transcript text.
3. For partial transcripts, translation is sent as `translation` in the
   `transcript_partial` message for live display.

---

## 6. Text-to-Speech — Phase 5, implemented

### 6.1 `TTSService` (Piper TTS)

File: `backend/services/tts_service.py`

- Singleton loaded on startup via `get_tts_service()`.
- Uses Piper TTS with ONNX voice models downloaded from the `rhasspy/piper-voices`
  Hugging Face repository (v1.0.0 branch).
- Voices loaded per language:
  - `en` → `en_US-lessac-medium`
  - `es` → `es_ES-sharvard-medium`
  - `pt` → `pt_BR-faber-medium`
- `synthesize(text, language, length_scale, sentence_silence) -> bytes`:
  - Returns WAV audio bytes using `PiperVoice.synthesize_wav()`.
  - Configurable speaking rate (`length_scale`) and pause between sentences
    (`sentence_silence`).

### 6.2 How TTS integrates with the pipeline

On final transcripts, if TTS is enabled and a translation was produced:

1. `tts_service.synthesize(translated_text, target_lang)` runs in
   `asyncio.to_thread`.
2. The JSON transcript message is sent first with `has_tts_audio: true`.
3. The WAV bytes are sent as a binary WebSocket frame immediately after.

### 6.3 Frontend audio playback

- The frontend sets `binaryType = 'arraybuffer'` on the WebSocket.
- Binary frames are enqueued in a `ttsQueueRef`.
- `playNextTts()` decodes WAV data via `AudioContext.decodeAudioData`, creates a
  `BufferSourceNode`, and plays it. When playback ends, the next item in the
  queue is played automatically.
- A "Playing audio..." indicator is shown while TTS audio is active.
- The TTS toggle in the UI controls whether `?tts=true` is sent to the server.

---

## 7. Extension points for future phases

### 7.1 LLM-based conversational logic

After ASR + MT produce a transcript and translation:

- A conversation manager could maintain dialog state, call an LLM (local or
  remote), and produce a text response.
- This would plug in after translation, optionally replacing or augmenting the
  translated text before TTS.

### 7.2 Barge-in (interrupting TTS when the user speaks)

- Use **server-side** Silero VAD (existing) plus **client-side** VAD (to be
  upgraded from the current energy-based detector).
- While TTS is playing, if the user starts speaking:
  - Immediately stop TTS playback on the client and clear the audio queue.
  - Optionally send a control message to the server to cancel ongoing TTS.
- The current `isSpeaking` indicator and `ttsQueueRef` are natural hooks for
  this.

---

## 8. File index

- **Frontend**
  - `frontend/src/main.tsx` – React entrypoint.
  - `frontend/src/App.tsx` – App shell.
  - `frontend/src/components/AudioRecorder.tsx` – Microphone capture,
    MediaRecorder, WebSocket client, translation display, TTS audio playback,
    split-screen transcript UI.

- **Backend**
  - `backend/api/main.py`
    - FastAPI app and CORS.
    - WebSocket endpoint `/ws/audio` (bidirectional: audio + JSON in,
      JSON + binary out).
    - `AudioStreamDecoder` – WebM/Opus → PCM decoder (dedup via
      `_samples_returned`).
    - `AudioSession` – per-connection state (chunks, VAD, utterance PCM
      accumulation, WAV saving).
    - Non-blocking ASR/MT/TTS dispatch via `asyncio.create_task` +
      `asyncio.to_thread`.
    - Health and recordings listing endpoints.
  - `backend/services/vad/vad_service.py`
    - `VADService` – Silero VAD model wrapper (singleton).
    - `SpeechSegmentDetector` – silence-based segmentation.
  - `backend/services/asr_service.py`
    - `ASRService` – Faster-Whisper model wrapper (singleton).
    - `get_asr_service()` – global accessor.
  - `backend/services/mt_service.py`
    - `MTService` – NLLB-200 CTranslate2 translator (singleton).
    - `get_mt_service()` – global accessor.
    - Language code mapping (ISO 639-1 → NLLB Flores-200).
  - `backend/services/tts_service.py`
    - `TTSService` – Piper TTS voice manager (singleton).
    - `get_tts_service()` – global accessor.
    - Per-language voice model downloading and loading.
  - `backend/models/` – Downloaded model files (gitignored).
