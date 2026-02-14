/**
 * AudioRecorder
 *
 * Split-screen layout:
 *   LEFT  – Recording controls, status, language selector.
 *   RIGHT – Live transcript panel that scrolls as new text arrives.
 *
 * Message types from the server:
 *   "transcript_partial" – interim text while the user is still speaking.
 *   "transcript"         – final text once an utterance ends.
 *
 * The partial text is shown in a faded/italic style at the bottom of the
 * transcript list and is replaced by the final version once it arrives.
 */
import { useState, useRef, useCallback, useEffect } from 'react';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface AudioRecorderProps {
    serverUrl?: string;
    onBack?: () => void;
}

type RecordingState = 'idle' | 'recording' | 'connecting' | 'error';

interface TranscriptEntry {
    id: number;
    text: string;
    language: string;
    translation?: string;
    targetLanguage?: string;
    duration?: number;
    timestamp: Date;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function AudioRecorder({ serverUrl = 'ws://localhost:8000/ws/audio', onBack }: AudioRecorderProps) {
    /* ---- state ---- */
    const [state, setState] = useState<RecordingState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [duration, setDuration] = useState(0);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [language, setLanguage] = useState<'auto' | 'en' | 'es' | 'pt'>('auto');
    const [targetLanguage, setTargetLanguage] = useState<'none' | 'en' | 'es' | 'pt'>('es');
    const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
    const [liveTranscript, setLiveTranscript] = useState<string>('');
    const [liveTranslation, setLiveTranslation] = useState<string>('');

    const [ttsEnabled, setTtsEnabled] = useState(true);
    const [isPlayingTts, setIsPlayingTts] = useState(false);

    /* ---- refs ---- */
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<number | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const transcriptEndRef = useRef<HTMLDivElement | null>(null);
    const entryIdRef = useRef(0);
    const ttsAudioCtxRef = useRef<AudioContext | null>(null);
    const ttsQueueRef = useRef<ArrayBuffer[]>([]);
    const ttsPlayingRef = useRef(false);

    /* ---- TTS audio playback queue ---- */
    const playNextTts = useCallback(async () => {
        if (ttsPlayingRef.current) return;
        const buf = ttsQueueRef.current.shift();
        if (!buf) { setIsPlayingTts(false); return; }

        ttsPlayingRef.current = true;
        setIsPlayingTts(true);

        try {
            if (!ttsAudioCtxRef.current) {
                ttsAudioCtxRef.current = new AudioContext();
            }
            const ctx = ttsAudioCtxRef.current;
            if (ctx.state === 'suspended') await ctx.resume();

            const decoded = await ctx.decodeAudioData(buf.slice(0));
            const source = ctx.createBufferSource();
            source.buffer = decoded;
            source.connect(ctx.destination);
            source.onended = () => {
                ttsPlayingRef.current = false;
                playNextTts();           // play next in queue
            };
            source.start();
        } catch (e) {
            console.error('TTS playback error:', e);
            ttsPlayingRef.current = false;
            setIsPlayingTts(false);
            playNextTts();
        }
    }, []);

    const enqueueTtsAudio = useCallback((data: ArrayBuffer) => {
        ttsQueueRef.current.push(data);
        playNextTts();
    }, [playNextTts]);

    /* ---- auto-scroll transcript panel ---- */
    useEffect(() => {
        transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcripts, liveTranscript]);

    /* ---- cleanup on unmount ---- */
    useEffect(() => {
        return () => { stopRecording(); };
    }, []);

    /* ---- client-side energy-based speaking indicator ---- */
    const startAudioAnalysis = useCallback((stream: MediaStream) => {
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(analyser);
        analyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const speechThreshold = 30;
        let speechFrames = 0;
        const speechFrameThreshold = 3;

        const analyze = () => {
            if (!analyserRef.current) return;
            analyserRef.current.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

            if (average > speechThreshold) {
                speechFrames++;
                if (speechFrames >= speechFrameThreshold) setIsSpeaking(true);
            } else {
                speechFrames = Math.max(0, speechFrames - 1);
                if (speechFrames === 0) setIsSpeaking(false);
            }
            animationFrameRef.current = requestAnimationFrame(analyze);
        };
        analyze();
    }, []);

    const stopAudioAnalysis = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        analyserRef.current = null;
        setIsSpeaking(false);
    }, []);

    /* ---- recording lifecycle ---- */
    const startRecording = useCallback(async () => {
        try {
            setError(null);
            setState('connecting');

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 48000 },
            });
            streamRef.current = stream;
            startAudioAnalysis(stream);

            const params: string[] = [];
            if (language !== 'auto') params.push(`lang=${language}`);
            if (targetLanguage !== 'none') params.push(`target_lang=${targetLanguage}`);
            if (!ttsEnabled) params.push('tts=false');
            const queryStr = params.length > 0 ? params.join('&') : '';
            const wsUrl = queryStr
                ? `${serverUrl}${serverUrl.includes('?') ? '&' : '?'}${queryStr}`
                : serverUrl;
            const ws = new WebSocket(wsUrl);
            websocketRef.current = ws;

            ws.onopen = () => {
                setTranscripts([]);
                setLiveTranscript('');

                const mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 128000,
                });
                mediaRecorderRef.current = mediaRecorder;

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data);
                    }
                };
                mediaRecorder.onstop = () => console.log('MediaRecorder stopped');
                mediaRecorder.start(250);

                setState('recording');
                setDuration(0);
                timerRef.current = window.setInterval(() => setDuration(d => d + 1), 1000);
            };

            ws.onerror = () => {
                setError('Failed to connect to server');
                setState('error');
                cleanup();
            };

            ws.onclose = () => {
                if (state === 'recording') { cleanup(); setState('idle'); }
            };

            ws.binaryType = 'arraybuffer';
            ws.onmessage = (event) => {
                // Binary message = TTS audio (WAV)
                if (event.data instanceof ArrayBuffer) {
                    enqueueTtsAudio(event.data);
                    return;
                }

                // Text message = JSON transcript/translation
                if (typeof event.data !== 'string') return;
                try {
                    const msg = JSON.parse(event.data);
                    if (msg?.type === 'transcript' && typeof msg.text === 'string') {
                        setLiveTranscript('');
                        setLiveTranslation('');
                        entryIdRef.current += 1;
                        setTranscripts(prev => [
                            ...prev,
                            {
                                id: entryIdRef.current,
                                text: msg.text,
                                language: msg.language ?? 'unknown',
                                translation: msg.translation,
                                targetLanguage: msg.target_language,
                                duration: msg.duration,
                                timestamp: new Date(),
                            },
                        ]);
                    } else if (msg?.type === 'transcript_partial' && typeof msg.text === 'string') {
                        setLiveTranscript(msg.text);
                        setLiveTranslation(msg.translation ?? '');
                    }
                } catch { /* ignore bad JSON */ }
            };
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to access microphone');
            setState('error');
        }
    }, [serverUrl, state, startAudioAnalysis, language, targetLanguage, ttsEnabled, enqueueTtsAudio]);

    const cleanup = useCallback(() => {
        stopAudioAnalysis();
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current?.stop();
        mediaRecorderRef.current = null;
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
        websocketRef.current?.close();
        websocketRef.current = null;
    }, [stopAudioAnalysis]);

    const stopRecording = useCallback(() => {
        cleanup();
        setLiveTranscript('');
        setLiveTranslation('');
        // Clear TTS queue
        ttsQueueRef.current = [];
        ttsPlayingRef.current = false;
        setIsPlayingTts(false);
        setState('idle');
    }, [cleanup]);

    const formatDuration = (s: number) =>
        `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

    const langLabel: Record<string, string> = { en: 'EN', es: 'ES', pt: 'PT' };

    /* ---------------------------------------------------------------- */
    /*  Render                                                           */
    /* ---------------------------------------------------------------- */
    return (
        <>
            <style>{`
                /* ---------- import fonts ---------- */
                @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

                /* ---------- layout ---------- */
                .ca-root {
                    display: flex;
                    height: 100vh;
                    width: 100vw;
                    font-family: 'DM Sans', sans-serif;
                    background: #0a0a0f;
                    color: #e4e4e7;
                    overflow: hidden;
                }

                /* ---------- left panel ---------- */
                .ca-controls {
                    flex: 0 0 420px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 28px;
                    padding: 40px 32px;
                    background: linear-gradient(165deg, #111118 0%, #0a0a0f 100%);
                    border-right: 1px solid rgba(255,255,255,0.06);
                    position: relative;
                }
                .ca-controls::after {
                    content: '';
                    position: absolute;
                    top: 0; right: 0;
                    width: 1px; height: 100%;
                    background: linear-gradient(180deg, transparent, rgba(99,102,241,0.3), transparent);
                }

                .ca-logo {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px;
                    font-weight: 700;
                    letter-spacing: 3px;
                    text-transform: uppercase;
                    color: #6366f1;
                    margin-bottom: 8px;
                }
                .ca-tagline {
                    font-size: 13px;
                    color: #52525b;
                    margin-top: -20px;
                    margin-bottom: 12px;
                }

                /* status ring */
                .ca-ring-wrapper {
                    position: relative;
                    width: 180px; height: 180px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .ca-ring {
                    position: absolute;
                    inset: 0;
                    border-radius: 50%;
                    border: 2px solid rgba(255,255,255,0.06);
                    transition: border-color 0.3s, box-shadow 0.3s;
                }
                .ca-ring.recording {
                    border-color: rgba(239,68,68,0.5);
                    box-shadow: 0 0 40px rgba(239,68,68,0.12);
                }
                .ca-ring.speaking {
                    border-color: rgba(99,102,241,0.7);
                    box-shadow: 0 0 60px rgba(99,102,241,0.18);
                    animation: ca-pulse-ring 1.5s ease-in-out infinite;
                }
                @keyframes ca-pulse-ring {
                    0%, 100% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.06); opacity: 0.7; }
                }
                .ca-duration {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 36px;
                    font-weight: 500;
                    letter-spacing: 2px;
                    color: #fafafa;
                }
                .ca-status-label {
                    font-size: 12px;
                    letter-spacing: 2px;
                    text-transform: uppercase;
                    margin-top: 6px;
                }
                .ca-status-label.speaking { color: #6366f1; }
                .ca-status-label.recording { color: #ef4444; }
                .ca-status-label.idle { color: #52525b; }

                /* buttons */
                .ca-btn {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    padding: 14px 36px;
                    border: none;
                    border-radius: 12px;
                    font-family: 'DM Sans', sans-serif;
                    font-size: 15px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.15s, box-shadow 0.2s, background 0.2s;
                }
                .ca-btn:active { transform: scale(0.97); }
                .ca-btn.start {
                    background: #6366f1;
                    color: #fff;
                    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
                }
                .ca-btn.start:hover {
                    background: #818cf8;
                    box-shadow: 0 6px 32px rgba(99,102,241,0.45);
                }
                .ca-btn.stop {
                    background: rgba(239,68,68,0.15);
                    color: #ef4444;
                    border: 1px solid rgba(239,68,68,0.3);
                }
                .ca-btn.stop:hover { background: rgba(239,68,68,0.25); }
                .ca-btn.connecting {
                    background: rgba(255,255,255,0.05);
                    color: #71717a;
                    cursor: wait;
                }
                .ca-btn svg { width: 18px; height: 18px; }

                /* language picker */
                .ca-lang-picker {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 13px;
                    color: #71717a;
                }
                .ca-lang-picker select {
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: #e4e4e7;
                    font-family: 'DM Sans', sans-serif;
                    font-size: 13px;
                    padding: 8px 12px;
                    cursor: pointer;
                    outline: none;
                    transition: border-color 0.2s;
                }
                .ca-lang-picker select:hover { border-color: rgba(99,102,241,0.4); }
                .ca-lang-picker select:disabled { opacity: 0.4; cursor: not-allowed; }

                .ca-error {
                    background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.25);
                    color: #fca5a5;
                    font-size: 13px;
                    padding: 10px 16px;
                    border-radius: 10px;
                    max-width: 300px;
                    text-align: center;
                }

                /* ---------- right panel (transcript) ---------- */
                .ca-transcript-panel {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    min-width: 0;
                    position: relative;
                }
                .ca-transcript-header {
                    flex: 0 0 auto;
                    padding: 20px 28px 14px;
                    border-bottom: 1px solid rgba(255,255,255,0.06);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                .ca-transcript-title {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 2px;
                    text-transform: uppercase;
                    color: #52525b;
                }
                .ca-transcript-count {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px;
                    color: #3f3f46;
                }

                .ca-transcript-body {
                    flex: 1;
                    overflow-y: auto;
                    padding: 24px 28px;
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                }
                /* custom scrollbar */
                .ca-transcript-body::-webkit-scrollbar { width: 6px; }
                .ca-transcript-body::-webkit-scrollbar-track { background: transparent; }
                .ca-transcript-body::-webkit-scrollbar-thumb {
                    background: rgba(255,255,255,0.08);
                    border-radius: 3px;
                }

                .ca-empty {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 12px;
                    color: #3f3f46;
                    text-align: center;
                }
                .ca-empty-icon { font-size: 40px; opacity: 0.4; }
                .ca-empty-text { font-size: 14px; line-height: 1.6; max-width: 280px; }

                /* transcript entries */
                .ca-entry {
                    display: flex;
                    gap: 14px;
                    animation: ca-fade-in 0.3s ease-out;
                }
                @keyframes ca-fade-in {
                    from { opacity: 0; transform: translateY(8px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                .ca-entry-dot {
                    flex: 0 0 8px;
                    width: 8px; height: 8px;
                    border-radius: 50%;
                    background: #6366f1;
                    margin-top: 7px;
                }
                .ca-entry-content { flex: 1; min-width: 0; }
                .ca-entry-text {
                    font-size: 15px;
                    line-height: 1.65;
                    color: #fafafa;
                    word-break: break-word;
                }
                .ca-entry-meta {
                    display: flex;
                    gap: 12px;
                    margin-top: 4px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px;
                    color: #52525b;
                }

                /* translation row beneath transcript entry */
                .ca-entry-translation {
                    margin-top: 6px;
                    padding: 8px 12px;
                    background: rgba(99,102,241,0.08);
                    border-left: 3px solid rgba(99,102,241,0.4);
                    border-radius: 6px;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #a5b4fc;
                    word-break: break-word;
                }
                .ca-entry-translation-label {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 1.5px;
                    text-transform: uppercase;
                    color: #6366f1;
                    margin-bottom: 2px;
                }

                /* live / partial transcript */
                .ca-live {
                    display: flex;
                    gap: 14px;
                    animation: ca-fade-in 0.2s ease-out;
                }
                .ca-live-dot {
                    flex: 0 0 8px;
                    width: 8px; height: 8px;
                    border-radius: 50%;
                    background: #6366f1;
                    margin-top: 7px;
                    animation: ca-blink 1s ease-in-out infinite;
                }
                @keyframes ca-blink {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.3; }
                }
                .ca-live-text {
                    flex: 1;
                    font-size: 15px;
                    line-height: 1.65;
                    color: rgba(250,250,250,0.5);
                    font-style: italic;
                    word-break: break-word;
                }
                .ca-live-translation {
                    margin-top: 4px;
                    padding: 6px 10px;
                    background: rgba(99,102,241,0.06);
                    border-left: 3px solid rgba(99,102,241,0.25);
                    border-radius: 6px;
                    font-size: 13px;
                    line-height: 1.55;
                    color: rgba(165,180,252,0.6);
                    font-style: italic;
                    word-break: break-word;
                }

                /* target language picker row */
                .ca-lang-row {
                    display: flex;
                    gap: 16px;
                    flex-wrap: wrap;
                    align-items: center;
                    justify-content: center;
                }

                /* TTS toggle */
                .ca-tts-toggle {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 13px;
                    color: #71717a;
                }
                .ca-tts-toggle label {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                }
                .ca-tts-toggle input[type="checkbox"] {
                    appearance: none;
                    width: 36px; height: 20px;
                    background: rgba(255,255,255,0.08);
                    border-radius: 10px;
                    position: relative;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .ca-tts-toggle input[type="checkbox"]::after {
                    content: '';
                    position: absolute;
                    top: 2px; left: 2px;
                    width: 16px; height: 16px;
                    background: #71717a;
                    border-radius: 50%;
                    transition: transform 0.2s, background 0.2s;
                }
                .ca-tts-toggle input[type="checkbox"]:checked {
                    background: rgba(99,102,241,0.3);
                }
                .ca-tts-toggle input[type="checkbox"]:checked::after {
                    transform: translateX(16px);
                    background: #6366f1;
                }
                .ca-tts-toggle input:disabled {
                    opacity: 0.4;
                    cursor: not-allowed;
                }

                /* TTS playing indicator */
                .ca-tts-playing {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px;
                    color: #22d3ee;
                    letter-spacing: 1px;
                    animation: ca-fade-in 0.2s ease-out;
                }
                .ca-tts-bars {
                    display: flex;
                    gap: 2px;
                    align-items: flex-end;
                    height: 14px;
                }
                .ca-tts-bar {
                    width: 3px;
                    background: #22d3ee;
                    border-radius: 1px;
                    animation: ca-bar-bounce 0.6s ease-in-out infinite;
                }
                .ca-tts-bar:nth-child(1) { height: 6px; animation-delay: 0s; }
                .ca-tts-bar:nth-child(2) { height: 10px; animation-delay: 0.15s; }
                .ca-tts-bar:nth-child(3) { height: 8px; animation-delay: 0.3s; }
                @keyframes ca-bar-bounce {
                    0%, 100% { transform: scaleY(0.5); }
                    50% { transform: scaleY(1.2); }
                }

                /* ---------- responsive ---------- */
                @media (max-width: 820px) {
                    .ca-root { flex-direction: column; }
                    .ca-controls {
                        flex: 0 0 auto;
                        border-right: none;
                        border-bottom: 1px solid rgba(255,255,255,0.06);
                        padding: 24px 20px;
                    }
                    .ca-controls::after { display: none; }
                    .ca-ring-wrapper { width: 120px; height: 120px; }
                    .ca-duration { font-size: 24px; }
                    .ca-transcript-panel { min-height: 300px; }
                }
            `}</style>

            <div className="ca-root">
                {/* ==================== LEFT: Controls ==================== */}
                <div className="ca-controls">
                    {onBack && (
                        <button
                            onClick={onBack}
                            style={{
                                position: 'absolute', top: 16, left: 16,
                                background: 'none', border: 'none', color: '#71717a',
                                fontSize: 13, cursor: 'pointer', display: 'flex',
                                alignItems: 'center', gap: 4, padding: '6px 10px',
                                borderRadius: 6, transition: 'all 0.2s',
                            }}
                            onMouseEnter={e => { e.currentTarget.style.color = '#e4e4e7'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
                            onMouseLeave={e => { e.currentTarget.style.color = '#71717a'; e.currentTarget.style.background = 'none'; }}
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M19 12H5" /><path d="M12 19l-7-7 7-7" />
                            </svg>
                            Back
                        </button>
                    )}
                    <div className="ca-logo">CourtAccess AI</div>
                    <div className="ca-tagline">Real-time speech transcription &amp; translation</div>

                    {/* Status ring + timer */}
                    <div className="ca-ring-wrapper">
                        <div className={`ca-ring ${state === 'recording' ? (isSpeaking ? 'speaking' : 'recording') : ''}`} />
                        <div className="ca-duration">
                            {state === 'recording' ? formatDuration(duration) : '00:00'}
                        </div>
                    </div>
                    <div className={`ca-status-label ${isSpeaking ? 'speaking' : state === 'recording' ? 'recording' : 'idle'}`}>
                        {isSpeaking ? 'Speaking...' : state === 'recording' ? 'Listening' : state === 'connecting' ? 'Connecting...' : 'Ready'}
                    </div>

                    {/* Error */}
                    {error && <div className="ca-error">{error}</div>}

                    {/* Action button */}
                    {state === 'idle' || state === 'error' ? (
                        <button className="ca-btn start" onClick={startRecording}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                <line x1="12" y1="19" x2="12" y2="22" />
                            </svg>
                            Start Recording
                        </button>
                    ) : state === 'recording' ? (
                        <button className="ca-btn stop" onClick={stopRecording}>
                            <svg viewBox="0 0 24 24" fill="currentColor">
                                <rect x="6" y="6" width="12" height="12" rx="2" />
                            </svg>
                            Stop Recording
                        </button>
                    ) : (
                        <button className="ca-btn connecting" disabled>Connecting...</button>
                    )}

                    {/* Language pickers */}
                    <div className="ca-lang-row">
                        <div className="ca-lang-picker">
                            <span>Source</span>
                            <select
                                value={language}
                                onChange={(e) => setLanguage(e.target.value as 'auto' | 'en' | 'es' | 'pt')}
                                disabled={state === 'recording' || state === 'connecting'}
                            >
                                <option value="auto">Auto-detect</option>
                                <option value="en">English</option>
                                <option value="es">Spanish</option>
                                <option value="pt">Portuguese</option>
                            </select>
                        </div>
                        <div className="ca-lang-picker">
                            <span>Translate to</span>
                            <select
                                value={targetLanguage}
                                onChange={(e) => setTargetLanguage(e.target.value as 'none' | 'en' | 'es' | 'pt')}
                                disabled={state === 'recording' || state === 'connecting'}
                            >
                                <option value="none">Off</option>
                                <option value="en">English</option>
                                <option value="es">Spanish</option>
                                <option value="pt">Portuguese</option>
                            </select>
                        </div>
                    </div>

                    {/* TTS toggle */}
                    {targetLanguage !== 'none' && (
                        <div className="ca-tts-toggle">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={ttsEnabled}
                                    onChange={(e) => setTtsEnabled(e.target.checked)}
                                    disabled={state === 'recording' || state === 'connecting'}
                                />
                                Speak translation
                            </label>
                        </div>
                    )}

                    {/* TTS playing indicator */}
                    {isPlayingTts && (
                        <div className="ca-tts-playing">
                            <div className="ca-tts-bars">
                                <div className="ca-tts-bar" />
                                <div className="ca-tts-bar" />
                                <div className="ca-tts-bar" />
                            </div>
                            Playing audio...
                        </div>
                    )}
                </div>

                {/* ==================== RIGHT: Transcript ==================== */}
                <div className="ca-transcript-panel">
                    <div className="ca-transcript-header">
                        <span className="ca-transcript-title">
                            {targetLanguage !== 'none' ? 'Transcript & Translation' : 'Live Transcript'}
                        </span>
                        {transcripts.length > 0 && (
                            <span className="ca-transcript-count">
                                {transcripts.length} utterance{transcripts.length !== 1 ? 's' : ''}
                            </span>
                        )}
                    </div>

                    <div className="ca-transcript-body">
                        {transcripts.length === 0 && !liveTranscript ? (
                            <div className="ca-empty">
                                <div className="ca-empty-icon">🎙</div>
                                <div className="ca-empty-text">
                                    {state === 'recording'
                                        ? 'Start speaking — your words will appear here in real time.'
                                        : 'Press Start Recording and speak into your microphone.'}
                                </div>
                            </div>
                        ) : (
                            <>
                                {/* Finalized transcript entries */}
                                {transcripts.map((entry) => (
                                    <div key={entry.id} className="ca-entry">
                                        <div className="ca-entry-dot" />
                                        <div className="ca-entry-content">
                                            <div className="ca-entry-text">{entry.text}</div>
                                            {entry.translation && (
                                                <div className="ca-entry-translation">
                                                    <div className="ca-entry-translation-label">
                                                        {langLabel[entry.targetLanguage ?? ''] ?? entry.targetLanguage?.toUpperCase() ?? 'Translation'}
                                                    </div>
                                                    {entry.translation}
                                                </div>
                                            )}
                                            <div className="ca-entry-meta">
                                                <span>{langLabel[entry.language] ?? entry.language.toUpperCase()}</span>
                                                {entry.targetLanguage && <span>→ {langLabel[entry.targetLanguage] ?? entry.targetLanguage.toUpperCase()}</span>}
                                                {entry.duration != null && <span>{entry.duration}s</span>}
                                                <span>
                                                    {entry.timestamp.toLocaleTimeString([], {
                                                        hour: '2-digit',
                                                        minute: '2-digit',
                                                        second: '2-digit',
                                                    })}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}

                                {/* Live partial transcript */}
                                {liveTranscript && (
                                    <div className="ca-live">
                                        <div className="ca-live-dot" />
                                        <div style={{ flex: 1 }}>
                                            <div className="ca-live-text">{liveTranscript}</div>
                                            {liveTranslation && (
                                                <div className="ca-live-translation">{liveTranslation}</div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                        <div ref={transcriptEndRef} />
                    </div>
                </div>
            </div>
        </>
    );
}