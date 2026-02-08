import { useState, useRef, useCallback, useEffect } from 'react';

interface AudioRecorderProps {
    serverUrl?: string;
}

type RecordingState = 'idle' | 'recording' | 'connecting' | 'error';

export function AudioRecorder({ serverUrl = 'ws://localhost:8000/ws/audio' }: AudioRecorderProps) {
    const [state, setState] = useState<RecordingState>('idle');
    const [error, setError] = useState<string | null>(null);
    const [duration, setDuration] = useState(0);
    const [isSpeaking, setIsSpeaking] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const websocketRef = useRef<WebSocket | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const timerRef = useRef<number | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopRecording();
        };
    }, []);

    // Simple audio level-based voice detection
    const startAudioAnalysis = useCallback((stream: MediaStream) => {
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(analyser);
        analyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const speechThreshold = 30; // Adjust based on testing
        let speechFrames = 0;
        const speechFrameThreshold = 3;

        const analyze = () => {
            if (!analyserRef.current) return;

            analyserRef.current.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

            if (average > speechThreshold) {
                speechFrames++;
                if (speechFrames >= speechFrameThreshold) {
                    setIsSpeaking(true);
                }
            } else {
                speechFrames = Math.max(0, speechFrames - 1);
                if (speechFrames === 0) {
                    setIsSpeaking(false);
                }
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

    const startRecording = useCallback(async () => {
        try {
            setError(null);
            setState('connecting');

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 48000,
                }
            });
            streamRef.current = stream;

            // Start audio level analysis for voice detection
            startAudioAnalysis(stream);

            // Connect to WebSocket server
            const ws = new WebSocket(serverUrl);
            websocketRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');

                // Create MediaRecorder with WebM/Opus format
                const mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus',
                    audioBitsPerSecond: 128000,
                });
                mediaRecorderRef.current = mediaRecorder;

                // Send audio chunks to server
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    console.log('MediaRecorder stopped');
                };

                // Start recording with 250ms chunks for real-time streaming
                mediaRecorder.start(250);
                setState('recording');
                setDuration(0);

                // Start duration timer
                timerRef.current = window.setInterval(() => {
                    setDuration(d => d + 1);
                }, 1000);
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                setError('Failed to connect to server');
                setState('error');
                cleanup();
            };

            ws.onclose = () => {
                console.log('WebSocket closed');
                if (state === 'recording') {
                    cleanup();
                    setState('idle');
                }
            };

        } catch (err) {
            console.error('Error starting recording:', err);
            setError(err instanceof Error ? err.message : 'Failed to access microphone');
            setState('error');
        }
    }, [serverUrl, state, startAudioAnalysis]);

    const cleanup = useCallback(() => {
        // Stop audio analysis
        stopAudioAnalysis();

        // Stop timer
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }

        // Stop MediaRecorder
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        mediaRecorderRef.current = null;

        // Stop media tracks
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
        }
        streamRef.current = null;

        // Close WebSocket
        if (websocketRef.current) {
            websocketRef.current.close();
        }
        websocketRef.current = null;
    }, [stopAudioAnalysis]);

    const stopRecording = useCallback(() => {
        cleanup();
        setState('idle');
    }, [cleanup]);

    const formatDuration = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="audio-recorder">
            <div className="recorder-status">
                {state === 'recording' && (
                    <div className="recording-indicator">
                        <span className={`pulse ${isSpeaking ? 'speaking' : ''}`}></span>
                        <span className="duration">{formatDuration(duration)}</span>
                        {isSpeaking && <span className="speaking-label">Speaking...</span>}
                    </div>
                )}
                {state === 'connecting' && (
                    <div className="connecting-indicator">Connecting...</div>
                )}
                {error && (
                    <div className="error-message">{error}</div>
                )}
            </div>

            <div className="recorder-controls">
                {state === 'idle' || state === 'error' ? (
                    <button
                        className="record-button start"
                        onClick={startRecording}
                    >
                        <svg viewBox="0 0 24 24" width="24" height="24">
                            <circle cx="12" cy="12" r="10" fill="currentColor" />
                        </svg>
                        Start Recording
                    </button>
                ) : state === 'recording' ? (
                    <button
                        className="record-button stop"
                        onClick={stopRecording}
                    >
                        <svg viewBox="0 0 24 24" width="24" height="24">
                            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                        </svg>
                        Stop Recording
                    </button>
                ) : (
                    <button className="record-button connecting" disabled>
                        Connecting...
                    </button>
                )}
            </div>
        </div>
    );
}
