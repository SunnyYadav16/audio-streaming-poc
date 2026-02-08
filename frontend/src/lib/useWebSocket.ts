import { useState, useEffect, useRef, useCallback } from 'react';

interface UseWebSocketOptions {
    url: string;
    onMessage?: (data: MessageEvent) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
    reconnect?: boolean;
    reconnectInterval?: number;
}

interface UseWebSocketReturn {
    socket: WebSocket | null;
    isConnected: boolean;
    send: (data: string | ArrayBuffer | Blob) => void;
    connect: () => void;
    disconnect: () => void;
}

export function useWebSocket({
    url,
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnect = false,
    reconnectInterval = 3000,
}: UseWebSocketOptions): UseWebSocketReturn {
    const [isConnected, setIsConnected] = useState(false);
    const socketRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);

    const connect = useCallback(() => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        const ws = new WebSocket(url);
        socketRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            onOpen?.();
        };

        ws.onclose = () => {
            setIsConnected(false);
            socketRef.current = null;
            onClose?.();

            if (reconnect) {
                reconnectTimeoutRef.current = window.setTimeout(connect, reconnectInterval);
            }
        };

        ws.onerror = (error) => {
            onError?.(error);
        };

        ws.onmessage = (event) => {
            onMessage?.(event);
        };
    }, [url, onOpen, onClose, onError, onMessage, reconnect, reconnectInterval]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
    }, []);

    const send = useCallback((data: string | ArrayBuffer | Blob) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(data);
        }
    }, []);

    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return {
        socket: socketRef.current,
        isConnected,
        send,
        connect,
        disconnect,
    };
}
