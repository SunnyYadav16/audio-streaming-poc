/**
 * ConversationSession
 *
 * Bidirectional translated conversation between two users.
 *
 * Session lifecycle (U.2–U.6):
 *   lobby   → create or join a room (no audio)
 *   waiting → room created, waiting for partner (no audio)
 *   ready   → both connected, host can start session (no audio)
 *   active  → session live, audio flowing
 *   ended   → partner disconnected
 *
 * Mute (U.1/U.7):
 *   During active session, either user can mute/unmute.
 *   Muting stops sending audio chunks entirely.
 *   Partner sees a visual indicator.
 */
import { useState, useRef, useCallback, useEffect } from 'react';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type SessionPhase =
    | 'lobby'       // choosing to create or join
    | 'waiting'     // created room, waiting for partner (no audio)
    | 'ready'       // both users connected, host can start (no audio)
    | 'active'      // session live, audio flowing
    | 'ended';      // partner left

interface PartnerInfo {
    name: string;
    language: string;
}

interface ChatMessage {
    id: number;
    speaker: 'self' | 'partner';
    speakerName?: string;
    text: string;
    language: string;
    translation?: string;
    targetLanguage?: string;
    duration?: number;
    timestamp: Date;
}

/* ------------------------------------------------------------------ */
/*  Binary control markers (must match backend)                        */
/* ------------------------------------------------------------------ */

const MARKER_SESSION_START = new Uint8Array([0x53, 0x54, 0x52, 0x54]); // b'STRT'
const MARKER_SESSION_END   = new Uint8Array([0x45, 0x4E, 0x44, 0x53]); // b'ENDS'
const MARKER_MIC_MUTE      = new Uint8Array([0x4D, 0x55, 0x54, 0x45]); // b'MUTE'
const MARKER_MIC_UNMUTE    = new Uint8Array([0x55, 0x4E, 0x4D, 0x54]); // b'UNMT'

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function ConversationSession({ onBack }: { onBack: () => void }) {
    /* ---- lobby state ---- */
    const [phase, setPhase] = useState<SessionPhase>('lobby');
    const [myName, setMyName] = useState('');
    const [joinName, setJoinName] = useState('');
    const [roomCode, setRoomCode] = useState('');
    const [joinCode, setJoinCode] = useState('');
    const [error, setError] = useState<string | null>(null);

    // Creator picks both languages; joiner's language is auto-assigned
    const [langA, setLangA] = useState<'en' | 'es' | 'pt'>('en');   // creator speaks
    const [langB, setLangB] = useState<'en' | 'es' | 'pt'>('es');   // partner speaks
    const [myLanguage, setMyLanguage] = useState<string>('en');       // set after connect

    /* ---- session state ---- */
    const [partner, setPartner] = useState<PartnerInfo | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [duration, setDuration] = useState(0);
    const [micLocked, setMicLocked] = useState(false);  // echo-suppression lockout
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [livePartial, setLivePartial] = useState<{ speaker: 'self' | 'partner'; text: string; translation?: string } | null>(null);
    const [isPlayingTts, setIsPlayingTts] = useState(false);

    /* ---- U.1–U.7 state ---- */
    const [isCreator, setIsCreator] = useState(false);   // U.3: role A flag
    const [isMuted, setIsMuted] = useState(false);        // U.1: local mute
    const [partnerMuted, setPartnerMuted] = useState(false); // U.7: partner mute indicator

    /* ---- refs ---- */
    const wsRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<number | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const chatEndRef = useRef<HTMLDivElement | null>(null);
    const msgIdRef = useRef(0);
    const ttsAudioCtxRef = useRef<AudioContext | null>(null);
    const ttsQueueRef = useRef<ArrayBuffer[]>([]);
    const ttsPlayingRef = useRef(false);
    const isMutedRef = useRef(false);  // mirror for use in ondataavailable closure
    const recorderRestartRef = useRef<number | null>(null); // periodic MediaRecorder restart

    // Keep ref in sync with state
    useEffect(() => { isMutedRef.current = isMuted; }, [isMuted]);

    /* ---- TTS playback queue ---- */
    const playNextTts = useCallback(async () => {
        if (ttsPlayingRef.current) return;
        const buf = ttsQueueRef.current.shift();
        if (!buf) { setIsPlayingTts(false); return; }

        ttsPlayingRef.current = true;
        setIsPlayingTts(true);

        try {
            if (!ttsAudioCtxRef.current) ttsAudioCtxRef.current = new AudioContext();
            const ctx = ttsAudioCtxRef.current;
            if (ctx.state === 'suspended') await ctx.resume();

            const decoded = await ctx.decodeAudioData(buf.slice(0));
            const source = ctx.createBufferSource();
            source.buffer = decoded;
            source.connect(ctx.destination);
            source.onended = () => {
                ttsPlayingRef.current = false;
                playNextTts();
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

    /* ---- auto-scroll ---- */
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, livePartial]);

    /* ---- cleanup on unmount ---- */
    useEffect(() => {
        return () => { disconnectAll(); };
    }, []);

    /* ---- speaking indicator (energy-based VAD) ---- */
    const startAudioAnalysis = useCallback((stream: MediaStream) => {
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(analyser);
        analyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        let speechFrames = 0;

        const analyze = () => {
            if (!analyserRef.current) return;
            analyserRef.current.getByteFrequencyData(dataArray);
            const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
            if (avg > 30) {
                speechFrames++;
                if (speechFrames >= 3) setIsSpeaking(true);
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

    /* ---- U.2: start mic capture (called when session becomes active) ---- */
    const startAudioCapture = useCallback(async () => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 48000 },
            });
            streamRef.current = stream;
            startAudioAnalysis(stream);

            const recorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 128000,
            });
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = (e) => {
                // U.1: skip send entirely when muted — no server processing
                if (e.data.size > 0 && ws.readyState === WebSocket.OPEN && !isMutedRef.current) {
                    ws.send(e.data);
                }
            };

            recorder.start(250);
            setIsRecording(true);
            setIsMuted(false);
            setDuration(0);
            timerRef.current = window.setInterval(() => setDuration(d => d + 1), 1000);

            // Restart MediaRecorder every 30s to send a fresh WebM header.
            // This prevents the server's AudioStreamDecoder buffer from
            // growing unbounded (O(N²) re-decode on every chunk).
            const RESTART_INTERVAL_MS = 30_000;
            recorderRestartRef.current = window.setInterval(() => {
                const rec = mediaRecorderRef.current;
                if (rec && rec.state === 'recording') {
                    rec.stop();
                    rec.start(250);
                }
            }, RESTART_INTERVAL_MS);
        } catch (micErr) {
            setError(micErr instanceof Error ? micErr.message : 'Microphone access denied');
        }
    }, [startAudioAnalysis]);

    /* ---- U.2/U.6: stop mic capture (session ended / paused) ---- */
    const stopAudioCapture = useCallback(() => {
        stopAudioAnalysis();
        if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
        if (recorderRestartRef.current) { clearInterval(recorderRestartRef.current); recorderRestartRef.current = null; }
        if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current?.stop();
        mediaRecorderRef.current = null;
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
        ttsQueueRef.current = [];
        ttsPlayingRef.current = false;
        setIsPlayingTts(false);
        setIsRecording(false);
        setIsSpeaking(false);
        setIsMuted(false);
        setMicLocked(false);
    }, [stopAudioAnalysis]);

    /* ---- disconnect everything ---- */
    const disconnectAll = useCallback(() => {
        stopAudioCapture();
        wsRef.current?.close();
        wsRef.current = null;
    }, [stopAudioCapture]);

    /* ---- U.3: Start Session (creator only) ---- */
    const startSession = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(MARKER_SESSION_START);
        }
    }, []);

    /* ---- U.6: End Session (creator only) ---- */
    const endSession = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(MARKER_SESSION_END);
        }
    }, []);

    /* ---- U.1/U.7: Toggle mute ---- */
    const toggleMute = useCallback(() => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const willMute = !isMutedRef.current;
        setIsMuted(willMute);
        // Send MUTE/UNMT to server so partner gets visual indicator
        ws.send(willMute ? MARKER_MIC_MUTE : MARKER_MIC_UNMUTE);
    }, []);

    /* ---- connect WebSocket (NO mic — U.2) ---- */
    const connectToRoom = useCallback(async (opts: {
        roomId?: string;
        name: string;
        myLang?: string;       // only for creator
        partnerLang?: string;  // only for creator
    }) => {
        try {
            setError(null);

            // Build URL with protocol-aware WebSocket
            const params = new URLSearchParams();
            if (opts.roomId) {
                // JOINING – no language params; server auto-assigns
                params.set('room_id', opts.roomId);
                setIsCreator(false);
            } else {
                // CREATING – send both languages
                params.set('my_lang', opts.myLang || 'en');
                params.set('partner_lang', opts.partnerLang || 'es');
                setIsCreator(true);
            }
            params.set('name', opts.name || 'User');

            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws/session?${params.toString()}`;

            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;
            ws.binaryType = 'arraybuffer';

            // U.2: on open we do NOT request mic — just confirm connection
            ws.onopen = () => {
                console.log('[WS] Connected to room');
            };

            ws.onerror = () => {
                setError('Connection failed. Is the server running?');
                disconnectAll();
                setPhase('lobby');
            };

            ws.onclose = () => {
                stopAudioCapture();
                wsRef.current = null;
            };

            ws.onmessage = (event) => {
                // Binary = TTS audio
                if (event.data instanceof ArrayBuffer) {
                    enqueueTtsAudio(event.data);
                    return;
                }

                if (typeof event.data !== 'string') return;
                try {
                    const msg = JSON.parse(event.data);

                    switch (msg.type) {
                        case 'room_created':
                            setRoomCode(msg.room_id);
                            setMyLanguage(msg.language);
                            // Phase set by session_status
                            break;

                        case 'room_joined':
                            setRoomCode(msg.room_id);
                            setMyLanguage(msg.language);
                            if (msg.partner_name) {
                                setPartner({ name: msg.partner_name, language: msg.partner_language });
                            }
                            // Phase set by session_status
                            break;

                        case 'partner_joined':
                            setPartner({ name: msg.name, language: msg.language });
                            setPartnerMuted(false);
                            // Phase set by session_status
                            break;

                        case 'partner_left':
                            setPartner(null);
                            setPartnerMuted(false);
                            // Phase set by session_status
                            break;

                        // U.5: Session status — drives phase transitions
                        case 'session_status':
                            switch (msg.status) {
                                case 'waiting':
                                    setPhase('waiting');
                                    break;
                                case 'ready':
                                    // Stop audio if we were active (End Session)
                                    stopAudioCapture();
                                    setPhase('ready');
                                    break;
                                case 'active':
                                    setPhase('active');
                                    // Start mic capture now
                                    startAudioCapture();
                                    break;
                                case 'ended':
                                    stopAudioCapture();
                                    setPhase('ended');
                                    break;
                            }
                            break;

                        // U.7: Partner mute indicators
                        case 'partner_muted':
                            setPartnerMuted(true);
                            break;
                        case 'partner_unmuted':
                            setPartnerMuted(false);
                            break;

                        case 'transcript':
                            if (typeof msg.text === 'string') {
                                setLivePartial(null);
                                msgIdRef.current += 1;
                                setMessages(prev => [...prev, {
                                    id: msgIdRef.current,
                                    speaker: msg.speaker || 'self',
                                    speakerName: msg.speaker_name,
                                    text: msg.text,
                                    language: msg.language ?? 'unknown',
                                    translation: msg.translation,
                                    targetLanguage: msg.target_language,
                                    duration: msg.duration,
                                    timestamp: new Date(),
                                }]);
                            }
                            break;

                        case 'transcript_partial':
                            if (typeof msg.text === 'string') {
                                setLivePartial({
                                    speaker: msg.speaker || 'self',
                                    text: msg.text,
                                    translation: msg.translation,
                                });
                            }
                            break;

                        case 'mic_locked': {
                            // Server locked our mic (echo suppression after TTS)
                            setMicLocked(true);
                            const lockDuration = msg.duration_ms || 2000;
                            setTimeout(() => setMicLocked(false), lockDuration);
                            break;
                        }

                        case 'error':
                            setError(msg.message || 'Unknown error');
                            disconnectAll();
                            setPhase('lobby');
                            break;
                    }
                } catch { /* bad JSON */ }
            };
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Connection failed');
            setPhase('lobby');
        }
    }, [startAudioCapture, stopAudioCapture, disconnectAll, enqueueTtsAudio]);

    const handleCreate = useCallback(() => {
        if (langA === langB) { setError('Languages must be different'); return; }
        setMessages([]);
        setPartner(null);
        setLivePartial(null);
        setPartnerMuted(false);
        connectToRoom({ name: myName, myLang: langA, partnerLang: langB });
    }, [connectToRoom, myName, langA, langB]);

    const handleJoin = useCallback(() => {
        const code = joinCode.trim().toUpperCase();
        if (!code) { setError('Please enter a room code'); return; }
        setMessages([]);
        setPartner(null);
        setLivePartial(null);
        setPartnerMuted(false);
        connectToRoom({ roomId: code, name: joinName });
    }, [joinCode, joinName, connectToRoom]);

    const handleLeave = useCallback(() => {
        disconnectAll();
        setPhase('lobby');
        setRoomCode('');
        setPartner(null);
        setMessages([]);
        setLivePartial(null);
        setDuration(0);
        setMyLanguage('en');
        setMicLocked(false);
        setIsMuted(false);
        setPartnerMuted(false);
        setIsCreator(false);
    }, [disconnectAll]);

    const formatDuration = (s: number) =>
        `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

    const langLabel: Record<string, string> = { en: 'English', es: 'Spanish', pt: 'Portuguese' };
    const langFlag: Record<string, string> = { en: 'EN', es: 'ES', pt: 'PT' };

    // Display name — whichever field was used to connect
    const displayName = myName || joinName || 'You';

    /* ---------------------------------------------------------------- */
    /*  Render                                                           */
    /* ---------------------------------------------------------------- */
    return (
        <>
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

                .cs-root {
                    display: flex; height: 100vh; width: 100vw;
                    font-family: 'DM Sans', sans-serif;
                    background: #0a0a0f; color: #e4e4e7; overflow: hidden;
                }

                /* ---- left panel ---- */
                .cs-sidebar {
                    flex: 0 0 380px; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; gap: 20px;
                    padding: 32px 28px;
                    background: linear-gradient(165deg, #111118 0%, #0a0a0f 100%);
                    border-right: 1px solid rgba(255,255,255,0.06);
                    position: relative;
                }
                .cs-sidebar::after {
                    content: ''; position: absolute; top: 0; right: 0;
                    width: 1px; height: 100%;
                    background: linear-gradient(180deg, transparent, rgba(99,102,241,0.3), transparent);
                }

                .cs-back {
                    position: absolute; top: 16px; left: 16px;
                    background: none; border: none; color: #71717a;
                    font-size: 13px; cursor: pointer; display: flex;
                    align-items: center; gap: 4px; padding: 6px 10px;
                    border-radius: 6px; transition: all 0.2s;
                }
                .cs-back:hover { color: #e4e4e7; background: rgba(255,255,255,0.05); }

                .cs-logo {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px; font-weight: 700;
                    letter-spacing: 3px; text-transform: uppercase; color: #6366f1;
                }
                .cs-subtitle {
                    font-size: 13px; color: #52525b; margin-top: -12px;
                }

                /* room code display */
                .cs-room-code-box {
                    background: rgba(99,102,241,0.08);
                    border: 1px dashed rgba(99,102,241,0.3);
                    border-radius: 12px; padding: 16px 28px; text-align: center;
                }
                .cs-room-code-label {
                    font-size: 11px; color: #6366f1; letter-spacing: 2px;
                    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
                    margin-bottom: 6px;
                }
                .cs-room-code {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 32px; font-weight: 700; letter-spacing: 6px;
                    color: #fafafa;
                }
                .cs-room-code-hint {
                    font-size: 12px; color: #52525b; margin-top: 6px;
                }

                /* partner info */
                .cs-partner-badge {
                    display: flex; align-items: center; gap: 10px;
                    background: rgba(34,211,238,0.08);
                    border: 1px solid rgba(34,211,238,0.2);
                    border-radius: 10px; padding: 12px 18px;
                }
                .cs-partner-dot {
                    width: 8px; height: 8px; border-radius: 50%;
                    background: #22d3ee;
                    box-shadow: 0 0 8px rgba(34,211,238,0.5);
                }
                .cs-partner-dot.muted {
                    background: #f59e0b;
                    box-shadow: 0 0 8px rgba(245,158,11,0.5);
                }
                .cs-partner-name {
                    font-size: 14px; font-weight: 600; color: #e4e4e7;
                }
                .cs-partner-lang {
                    font-size: 12px; color: #22d3ee;
                    font-family: 'JetBrains Mono', monospace;
                }
                .cs-partner-muted-label {
                    font-size: 10px; color: #f59e0b;
                    font-family: 'JetBrains Mono', monospace;
                    letter-spacing: 1px;
                }

                /* status ring */
                .cs-ring-wrap {
                    position: relative; width: 140px; height: 140px;
                    display: flex; align-items: center; justify-content: center;
                }
                .cs-ring {
                    position: absolute; inset: 0; border-radius: 50%;
                    border: 2px solid rgba(255,255,255,0.06);
                    transition: all 0.3s;
                }
                .cs-ring.speaking {
                    border-color: rgba(99,102,241,0.7);
                    box-shadow: 0 0 50px rgba(99,102,241,0.15);
                    animation: cs-pulse 1.5s ease-in-out infinite;
                }
                .cs-ring.recording {
                    border-color: rgba(34,211,238,0.4);
                    box-shadow: 0 0 30px rgba(34,211,238,0.1);
                }
                .cs-ring.muted-ring {
                    border-color: rgba(239,68,68,0.4);
                    box-shadow: 0 0 30px rgba(239,68,68,0.1);
                }
                @keyframes cs-pulse {
                    0%, 100% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.05); opacity: 0.7; }
                }
                .cs-timer {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 28px; font-weight: 500; letter-spacing: 2px; color: #fafafa;
                }
                .cs-status {
                    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
                    margin-top: 4px;
                }
                .cs-status.speaking { color: #6366f1; }
                .cs-status.recording { color: #22d3ee; }
                .cs-status.locked { color: #f59e0b; }
                .cs-status.muted-status { color: #ef4444; }

                .cs-ring.locked {
                    border-color: rgba(245,158,11,0.5);
                    box-shadow: 0 0 30px rgba(245,158,11,0.1);
                    animation: cs-pulse-lock 2s ease-in-out infinite;
                }
                @keyframes cs-pulse-lock {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                /* lobby inputs */
                .cs-field {
                    display: flex; flex-direction: column; gap: 6px; width: 100%; max-width: 280px;
                }
                .cs-field label {
                    font-size: 12px; color: #71717a; letter-spacing: 1px;
                    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
                }
                .cs-field input, .cs-field select {
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px; color: #e4e4e7;
                    font-family: 'DM Sans', sans-serif; font-size: 14px;
                    padding: 10px 14px; outline: none; transition: border-color 0.2s;
                }
                .cs-field input:focus, .cs-field select:focus {
                    border-color: rgba(99,102,241,0.5);
                }
                .cs-field input::placeholder { color: #3f3f46; }

                .cs-divider {
                    width: 100%; max-width: 280px; text-align: center;
                    font-size: 12px; color: #3f3f46; position: relative;
                }
                .cs-divider::before, .cs-divider::after {
                    content: ''; position: absolute; top: 50%;
                    width: 38%; height: 1px;
                    background: rgba(255,255,255,0.06);
                }
                .cs-divider::before { left: 0; }
                .cs-divider::after { right: 0; }

                /* join input row */
                .cs-join-row {
                    display: flex; gap: 8px; width: 100%; max-width: 280px;
                }
                .cs-join-row input {
                    flex: 1; background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px; color: #e4e4e7;
                    font-family: 'JetBrains Mono', monospace; font-size: 16px;
                    padding: 10px 14px; outline: none; text-align: center;
                    letter-spacing: 4px; text-transform: uppercase;
                    transition: border-color 0.2s;
                }
                .cs-join-row input:focus { border-color: rgba(99,102,241,0.5); }
                .cs-join-row input::placeholder { letter-spacing: 1px; font-size: 13px; color: #3f3f46; }

                /* buttons */
                .cs-btn {
                    display: flex; align-items: center; justify-content: center;
                    gap: 8px; padding: 12px 28px; border: none;
                    border-radius: 10px; font-family: 'DM Sans', sans-serif;
                    font-size: 14px; font-weight: 600; cursor: pointer;
                    transition: all 0.15s; width: 100%; max-width: 280px;
                }
                .cs-btn:active { transform: scale(0.97); }
                .cs-btn.primary {
                    background: #6366f1; color: #fff;
                    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
                }
                .cs-btn.primary:hover { background: #818cf8; }
                .cs-btn.secondary {
                    background: rgba(34,211,238,0.12); color: #22d3ee;
                    border: 1px solid rgba(34,211,238,0.25);
                }
                .cs-btn.secondary:hover { background: rgba(34,211,238,0.2); }
                .cs-btn.danger {
                    background: rgba(239,68,68,0.12); color: #ef4444;
                    border: 1px solid rgba(239,68,68,0.25);
                }
                .cs-btn.danger:hover { background: rgba(239,68,68,0.2); }
                .cs-btn.small { width: auto; max-width: none; padding: 10px 18px; font-size: 13px; }
                .cs-btn.success {
                    background: rgba(34,197,94,0.15); color: #22c55e;
                    border: 1px solid rgba(34,197,94,0.3);
                    box-shadow: 0 4px 20px rgba(34,197,94,0.15);
                }
                .cs-btn.success:hover { background: rgba(34,197,94,0.25); }
                .cs-btn.warning {
                    background: rgba(245,158,11,0.12); color: #f59e0b;
                    border: 1px solid rgba(245,158,11,0.25);
                }
                .cs-btn.warning:hover { background: rgba(245,158,11,0.2); }
                .cs-btn.mute-btn {
                    background: rgba(239,68,68,0.12); color: #ef4444;
                    border: 1px solid rgba(239,68,68,0.25);
                    max-width: 280px;
                }
                .cs-btn.mute-btn:hover { background: rgba(239,68,68,0.2); }
                .cs-btn.mute-btn.active {
                    background: rgba(239,68,68,0.25); color: #fca5a5;
                    border-color: rgba(239,68,68,0.5);
                }

                .cs-error {
                    background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.25);
                    color: #fca5a5; font-size: 13px; padding: 10px 16px;
                    border-radius: 10px; max-width: 280px; text-align: center;
                }

                /* waiting animation */
                .cs-waiting-dots {
                    display: flex; gap: 6px; align-items: center;
                    justify-content: center; padding: 8px 0;
                }
                .cs-waiting-dots span {
                    width: 8px; height: 8px; border-radius: 50%;
                    background: #6366f1; animation: cs-bounce 1.2s ease-in-out infinite;
                }
                .cs-waiting-dots span:nth-child(2) { animation-delay: 0.15s; }
                .cs-waiting-dots span:nth-child(3) { animation-delay: 0.3s; }
                @keyframes cs-bounce {
                    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
                    40% { transform: scale(1); opacity: 1; }
                }

                /* TTS indicator */
                .cs-tts-playing {
                    display: flex; align-items: center; gap: 6px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px; color: #22d3ee; letter-spacing: 1px;
                }
                .cs-tts-bars { display: flex; gap: 2px; align-items: flex-end; height: 14px; }
                .cs-tts-bar {
                    width: 3px; background: #22d3ee; border-radius: 1px;
                    animation: cs-bar 0.6s ease-in-out infinite;
                }
                .cs-tts-bar:nth-child(1) { height: 6px; animation-delay: 0s; }
                .cs-tts-bar:nth-child(2) { height: 10px; animation-delay: 0.15s; }
                .cs-tts-bar:nth-child(3) { height: 8px; animation-delay: 0.3s; }
                @keyframes cs-bar {
                    0%, 100% { transform: scaleY(0.5); }
                    50% { transform: scaleY(1.2); }
                }

                /* ready-stage info box */
                .cs-ready-info {
                    background: rgba(34,197,94,0.08);
                    border: 1px solid rgba(34,197,94,0.2);
                    border-radius: 10px; padding: 14px 20px;
                    text-align: center; max-width: 280px;
                }
                .cs-ready-info .label {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px; color: #22c55e; letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 6px;
                }
                .cs-ready-info .text {
                    font-size: 13px; color: #a1a1aa; line-height: 1.5;
                }

                /* ---- right panel: chat ---- */
                .cs-chat-panel {
                    flex: 1; display: flex; flex-direction: column; min-width: 0;
                }
                .cs-chat-header {
                    flex: 0 0 auto; padding: 18px 28px 14px;
                    border-bottom: 1px solid rgba(255,255,255,0.06);
                    display: flex; align-items: center; justify-content: space-between;
                }
                .cs-chat-title {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 12px; font-weight: 700; letter-spacing: 2px;
                    text-transform: uppercase; color: #52525b;
                }
                .cs-chat-count {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px; color: #3f3f46;
                }

                .cs-chat-body {
                    flex: 1; overflow-y: auto; padding: 20px 28px;
                    display: flex; flex-direction: column; gap: 14px;
                }
                .cs-chat-body::-webkit-scrollbar { width: 6px; }
                .cs-chat-body::-webkit-scrollbar-track { background: transparent; }
                .cs-chat-body::-webkit-scrollbar-thumb {
                    background: rgba(255,255,255,0.08); border-radius: 3px;
                }

                .cs-empty {
                    flex: 1; display: flex; flex-direction: column;
                    align-items: center; justify-content: center; gap: 12px;
                    color: #3f3f46; text-align: center;
                }
                .cs-empty-icon { font-size: 40px; opacity: 0.4; }
                .cs-empty-text { font-size: 14px; line-height: 1.6; max-width: 320px; }

                /* chat bubbles */
                .cs-msg { display: flex; flex-direction: column; max-width: 75%;
                    animation: cs-fadein 0.25s ease-out; }
                @keyframes cs-fadein {
                    from { opacity: 0; transform: translateY(6px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                .cs-msg.self { align-self: flex-end; align-items: flex-end; }
                .cs-msg.partner { align-self: flex-start; align-items: flex-start; }

                .cs-msg-speaker {
                    font-size: 11px; font-weight: 600; margin-bottom: 4px;
                    font-family: 'JetBrains Mono', monospace; letter-spacing: 1px;
                }
                .cs-msg.self .cs-msg-speaker { color: #818cf8; }
                .cs-msg.partner .cs-msg-speaker { color: #22d3ee; }

                .cs-msg-bubble {
                    padding: 12px 16px; border-radius: 14px;
                    font-size: 15px; line-height: 1.6; word-break: break-word;
                }
                .cs-msg.self .cs-msg-bubble {
                    background: rgba(99,102,241,0.15);
                    border: 1px solid rgba(99,102,241,0.2);
                    color: #e4e4e7;
                    border-bottom-right-radius: 4px;
                }
                .cs-msg.partner .cs-msg-bubble {
                    background: rgba(34,211,238,0.08);
                    border: 1px solid rgba(34,211,238,0.15);
                    color: #e4e4e7;
                    border-bottom-left-radius: 4px;
                }

                .cs-msg-translation {
                    margin-top: 6px; padding: 8px 12px;
                    border-radius: 10px; font-size: 14px; line-height: 1.55;
                    word-break: break-word;
                }
                .cs-msg.self .cs-msg-translation {
                    background: rgba(99,102,241,0.06);
                    border-left: 3px solid rgba(99,102,241,0.3);
                    color: #a5b4fc;
                }
                .cs-msg.partner .cs-msg-translation {
                    background: rgba(34,211,238,0.06);
                    border-left: 3px solid rgba(34,211,238,0.3);
                    color: #67e8f9;
                }
                .cs-msg-translation-label {
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
                    text-transform: uppercase; margin-bottom: 2px;
                }
                .cs-msg.self .cs-msg-translation-label { color: #6366f1; }
                .cs-msg.partner .cs-msg-translation-label { color: #22d3ee; }

                .cs-msg-meta {
                    display: flex; gap: 10px; margin-top: 4px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 10px; color: #52525b;
                }

                /* live partial */
                .cs-live-bubble {
                    padding: 10px 14px; border-radius: 14px;
                    font-size: 14px; line-height: 1.55;
                    font-style: italic; opacity: 0.6;
                    animation: cs-fadein 0.2s ease-out;
                }
                .cs-msg.self .cs-live-bubble {
                    background: rgba(99,102,241,0.08);
                    border: 1px dashed rgba(99,102,241,0.2);
                    color: #a5b4fc;
                    border-bottom-right-radius: 4px;
                }
                .cs-msg.partner .cs-live-bubble {
                    background: rgba(34,211,238,0.05);
                    border: 1px dashed rgba(34,211,238,0.15);
                    color: #67e8f9;
                    border-bottom-left-radius: 4px;
                }

                /* responsive */
                @media (max-width: 820px) {
                    .cs-root { flex-direction: column; }
                    .cs-sidebar {
                        flex: 0 0 auto; border-right: none;
                        border-bottom: 1px solid rgba(255,255,255,0.06);
                        padding: 20px 16px;
                    }
                    .cs-sidebar::after { display: none; }
                    .cs-ring-wrap { width: 100px; height: 100px; }
                    .cs-timer { font-size: 20px; }
                    .cs-chat-panel { min-height: 300px; }
                }
            `}</style>

            <div className="cs-root">
                {/* ==================== LEFT: Sidebar ==================== */}
                <div className="cs-sidebar">
                    <button className="cs-back" onClick={phase === 'lobby' ? onBack : handleLeave}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M19 12H5" /><path d="M12 19l-7-7 7-7" />
                        </svg>
                        {phase === 'lobby' ? 'Back' : 'Leave Room'}
                    </button>

                    <div className="cs-logo">Conversation</div>
                    <div className="cs-subtitle">Bidirectional translated chat</div>

                    {error && <div className="cs-error">{error}</div>}

                    {/* ---- LOBBY ---- */}
                    {phase === 'lobby' && (
                        <>
                            {/* ---- CREATE SECTION ---- */}
                            <div style={{
                                fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
                                letterSpacing: 2, color: '#6366f1', textTransform: 'uppercase' as const,
                            }}>
                                Create a room
                            </div>

                            <div className="cs-field">
                                <label>Your name</label>
                                <input
                                    value={myName}
                                    onChange={e => setMyName(e.target.value)}
                                    placeholder="Enter your name"
                                    maxLength={20}
                                />
                            </div>

                            {/* Language pair picker */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', maxWidth: 280 }}>
                                <div className="cs-field" style={{ flex: 1 }}>
                                    <label>You speak</label>
                                    <select value={langA} onChange={e => setLangA(e.target.value as 'en' | 'es' | 'pt')}>
                                        <option value="en">English</option>
                                        <option value="es">Spanish</option>
                                        <option value="pt">Portuguese</option>
                                    </select>
                                </div>
                                <div style={{
                                    marginTop: 18, fontSize: 16, color: '#52525b', fontWeight: 600,
                                }}>
                                    ↔
                                </div>
                                <div className="cs-field" style={{ flex: 1 }}>
                                    <label>They speak</label>
                                    <select value={langB} onChange={e => setLangB(e.target.value as 'en' | 'es' | 'pt')}>
                                        <option value="en">English</option>
                                        <option value="es">Spanish</option>
                                        <option value="pt">Portuguese</option>
                                    </select>
                                </div>
                            </div>

                            <button className="cs-btn primary" onClick={handleCreate}>
                                Create Room
                            </button>

                            <div className="cs-divider">or join an existing room</div>

                            {/* ---- JOIN SECTION ---- */}
                            <div className="cs-field">
                                <label>Your name</label>
                                <input
                                    value={joinName}
                                    onChange={e => setJoinName(e.target.value)}
                                    placeholder="Enter your name"
                                    maxLength={20}
                                />
                            </div>
                            <div className="cs-join-row">
                                <input
                                    value={joinCode}
                                    onChange={e => setJoinCode(e.target.value.toUpperCase())}
                                    placeholder="Room code"
                                    maxLength={6}
                                />
                                <button className="cs-btn secondary small" onClick={handleJoin}>
                                    Join
                                </button>
                            </div>
                        </>
                    )}

                    {/* ---- WAITING (room created, no partner yet) ---- */}
                    {phase === 'waiting' && (
                        <>
                            <div className="cs-room-code-box">
                                <div className="cs-room-code-label">Room Code</div>
                                <div className="cs-room-code">{roomCode}</div>
                                <div className="cs-room-code-hint">Share this code with your partner</div>
                            </div>

                            {/* Language pair badge */}
                            <div style={{
                                display: 'flex', alignItems: 'center', gap: 8,
                                fontSize: 13, color: '#a1a1aa',
                                fontFamily: "'JetBrains Mono', monospace",
                            }}>
                                <span style={{ color: '#818cf8' }}>{langLabel[langA]}</span>
                                <span style={{ color: '#52525b' }}>↔</span>
                                <span style={{ color: '#22d3ee' }}>{langLabel[langB]}</span>
                            </div>

                            <div className="cs-waiting-dots">
                                <span /><span /><span />
                            </div>
                            <div style={{ fontSize: 13, color: '#71717a' }}>
                                Waiting for partner to join...
                            </div>
                            <button className="cs-btn danger" onClick={handleLeave}>
                                Cancel
                            </button>
                        </>
                    )}

                    {/* ---- READY (both connected, no session yet — U.2) ---- */}
                    {phase === 'ready' && (
                        <>
                            <div className="cs-room-code-box">
                                <div className="cs-room-code-label">Room Code</div>
                                <div className="cs-room-code">{roomCode}</div>
                            </div>

                            {/* Your role */}
                            <div style={{
                                background: 'rgba(99,102,241,0.08)',
                                border: '1px solid rgba(99,102,241,0.2)',
                                borderRadius: 8, padding: '8px 16px',
                                fontSize: 13, color: '#a5b4fc',
                                fontFamily: "'JetBrains Mono', monospace",
                                textAlign: 'center' as const,
                            }}>
                                You speak <strong style={{ color: '#818cf8' }}>{langLabel[myLanguage] ?? myLanguage}</strong>
                            </div>

                            {/* Partner badge */}
                            {partner && (
                                <div className="cs-partner-badge">
                                    <div className="cs-partner-dot" />
                                    <div>
                                        <div className="cs-partner-name">{partner.name}</div>
                                        <div className="cs-partner-lang">
                                            speaks {langLabel[partner.language] ?? partner.language}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* U.3: Start Session button — creator only */}
                            {isCreator ? (
                                <>
                                    <div className="cs-ready-info">
                                        <div className="label">Both users connected</div>
                                        <div className="text">
                                            Start the session when you're ready to begin translation.
                                        </div>
                                    </div>
                                    <button className="cs-btn success" onClick={startSession}>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                            <path d="M8 5v14l11-7z"/>
                                        </svg>
                                        Start Session
                                    </button>
                                </>
                            ) : (
                                <>
                                    <div className="cs-waiting-dots">
                                        <span /><span /><span />
                                    </div>
                                    <div style={{ fontSize: 13, color: '#71717a', textAlign: 'center' }}>
                                        Waiting for host to start session...
                                    </div>
                                </>
                            )}

                            <button className="cs-btn danger" onClick={handleLeave}>
                                Leave Room
                            </button>
                        </>
                    )}

                    {/* ---- ACTIVE (session live) ---- */}
                    {phase === 'active' && (
                        <>
                            {/* Room code + language pair */}
                            <div style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: 11, color: '#52525b', letterSpacing: 2,
                                textAlign: 'center' as const,
                            }}>
                                ROOM {roomCode}
                            </div>

                            {/* Your role */}
                            <div style={{
                                background: 'rgba(99,102,241,0.08)',
                                border: '1px solid rgba(99,102,241,0.2)',
                                borderRadius: 8, padding: '8px 16px',
                                fontSize: 13, color: '#a5b4fc',
                                fontFamily: "'JetBrains Mono', monospace",
                                textAlign: 'center' as const,
                            }}>
                                You speak <strong style={{ color: '#818cf8' }}>{langLabel[myLanguage] ?? myLanguage}</strong>
                            </div>

                            {/* Partner badge (U.7: shows muted state) */}
                            {partner && (
                                <div className="cs-partner-badge">
                                    <div className={`cs-partner-dot ${partnerMuted ? 'muted' : ''}`} />
                                    <div>
                                        <div className="cs-partner-name">{partner.name}</div>
                                        <div className="cs-partner-lang">
                                            speaks {langLabel[partner.language] ?? partner.language}
                                        </div>
                                        {partnerMuted && (
                                            <div className="cs-partner-muted-label">MUTED</div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Status ring */}
                            <div className="cs-ring-wrap">
                                <div className={`cs-ring ${
                                    isMuted ? 'muted-ring'
                                    : micLocked ? 'locked'
                                    : isSpeaking ? 'speaking'
                                    : isRecording ? 'recording' : ''
                                }`} />
                                <div className="cs-timer">
                                    {isRecording ? formatDuration(duration) : '--:--'}
                                </div>
                            </div>
                            <div className={`cs-status ${
                                isMuted ? 'muted-status'
                                : micLocked ? 'locked'
                                : isSpeaking ? 'speaking'
                                : isRecording ? 'recording' : ''
                            }`}>
                                {isMuted
                                    ? 'Muted'
                                    : micLocked ? 'Mic paused — listening'
                                    : isSpeaking ? 'Speaking...'
                                    : isRecording ? 'Listening' : 'Inactive'}
                            </div>

                            {/* TTS / locked indicator */}
                            {(isPlayingTts || micLocked) && (
                                <div className="cs-tts-playing">
                                    <div className="cs-tts-bars">
                                        <div className="cs-tts-bar" />
                                        <div className="cs-tts-bar" />
                                        <div className="cs-tts-bar" />
                                    </div>
                                    {micLocked ? 'Listening to translation...' : 'Partner speaking...'}
                                </div>
                            )}

                            {/* U.1/U.7: Mute button — available during active session */}
                            <button
                                className={`cs-btn mute-btn ${isMuted ? 'active' : ''}`}
                                onClick={toggleMute}
                            >
                                {isMuted ? (
                                    <>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <line x1="1" y1="1" x2="23" y2="23"/>
                                            <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/>
                                            <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2c0 .76-.13 1.49-.35 2.17"/>
                                            <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
                                        </svg>
                                        Unmute Mic
                                    </>
                                ) : (
                                    <>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                            <line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>
                                        </svg>
                                        Mute Mic
                                    </>
                                )}
                            </button>

                            {/* U.6: End Session — creator only */}
                            {isCreator && (
                                <button className="cs-btn warning" onClick={endSession}>
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                        <rect x="6" y="6" width="12" height="12" rx="1"/>
                                    </svg>
                                    End Session
                                </button>
                            )}

                            <button className="cs-btn danger" onClick={handleLeave}>
                                Leave Room
                            </button>
                        </>
                    )}

                    {/* ---- ENDED (partner disconnected) ---- */}
                    {phase === 'ended' && (
                        <>
                            <div style={{
                                fontFamily: "'JetBrains Mono', monospace",
                                fontSize: 11, color: '#52525b', letterSpacing: 2,
                                textAlign: 'center' as const,
                            }}>
                                ROOM {roomCode}
                            </div>

                            <div style={{ fontSize: 13, color: '#ef4444', textAlign: 'center' }}>
                                Partner disconnected
                            </div>

                            <button className="cs-btn danger" onClick={handleLeave}>
                                Leave Room
                            </button>
                        </>
                    )}
                </div>

                {/* ==================== RIGHT: Chat ==================== */}
                <div className="cs-chat-panel">
                    <div className="cs-chat-header">
                        <span className="cs-chat-title">Conversation</span>
                        {messages.length > 0 && (
                            <span className="cs-chat-count">
                                {messages.length} message{messages.length !== 1 ? 's' : ''}
                            </span>
                        )}
                    </div>

                    <div className="cs-chat-body">
                        {messages.length === 0 && !livePartial ? (
                            <div className="cs-empty">
                                <div className="cs-empty-icon">
                                    {phase === 'lobby' ? '🔗'
                                        : phase === 'waiting' ? '⏳'
                                        : phase === 'ready' ? '✋'
                                        : phase === 'ended' ? '👋'
                                        : '💬'}
                                </div>
                                <div className="cs-empty-text">
                                    {phase === 'lobby'
                                        ? 'Create or join a room to start a translated conversation.'
                                        : phase === 'waiting'
                                        ? 'Waiting for your partner to join. Share the room code!'
                                        : phase === 'ready'
                                        ? isCreator
                                            ? 'Your partner has joined. Click "Start Session" to begin.'
                                            : 'Connected! Waiting for the host to start the session.'
                                        : phase === 'ended'
                                        ? 'The conversation has ended.'
                                        : 'Start speaking — your conversation will appear here in real time.'}
                                </div>
                            </div>
                        ) : (
                            <>
                                {messages.map((msg) => (
                                    <div key={msg.id} className={`cs-msg ${msg.speaker}`}>
                                        <div className="cs-msg-speaker">
                                            {msg.speaker === 'self'
                                                ? displayName
                                                : (msg.speakerName || partner?.name || 'Partner')}
                                        </div>
                                        <div className="cs-msg-bubble">{msg.text}</div>
                                        {msg.translation && (
                                            <div className="cs-msg-translation">
                                                <div className="cs-msg-translation-label">
                                                    {langFlag[msg.targetLanguage ?? ''] ?? msg.targetLanguage?.toUpperCase() ?? 'Translation'}
                                                </div>
                                                {msg.translation}
                                            </div>
                                        )}
                                        <div className="cs-msg-meta">
                                            <span>{langFlag[msg.language] ?? msg.language.toUpperCase()}</span>
                                            {msg.targetLanguage && <span>→ {langFlag[msg.targetLanguage]}</span>}
                                            {msg.duration != null && <span>{msg.duration}s</span>}
                                            <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                                        </div>
                                    </div>
                                ))}

                                {/* Live partial */}
                                {livePartial && (
                                    <div className={`cs-msg ${livePartial.speaker}`}>
                                        <div className="cs-msg-speaker">
                                            {livePartial.speaker === 'self'
                                                ? displayName
                                                : (partner?.name || 'Partner')}
                                        </div>
                                        <div className="cs-live-bubble">
                                            {livePartial.text}
                                            {livePartial.translation && (
                                                <div style={{ marginTop: 4, fontSize: 12, opacity: 0.7 }}>
                                                    {livePartial.translation}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                        <div ref={chatEndRef} />
                    </div>
                </div>
            </div>
        </>
    );
}
