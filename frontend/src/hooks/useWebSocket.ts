import { useState, useEffect, useCallback, useRef } from 'react';

interface WebSocketState {
    isConnected: boolean;
    lastMessage: any | null;
    error: string | null;
}

interface UseWebSocketOptions {
    onMessage?: (data: any) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    reconnectInterval?: number;
    maxRetries?: number;
}

export const useWebSocket = (url: string, options: UseWebSocketOptions = {}) => {
    const {
        onMessage,
        onConnect,
        onDisconnect,
        reconnectInterval = 3000,
        maxRetries = 5,
    } = options;

    const [state, setState] = useState<WebSocketState>({
        isConnected: false,
        lastMessage: null,
        error: null,
    });

    const wsRef = useRef<WebSocket | null>(null);
    const retriesRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                retriesRef.current = 0;
                setState(prev => ({ ...prev, isConnected: true, error: null }));
                onConnect?.();

                // Start ping interval to keep connection alive
                pingIntervalRef.current = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('ping');
                    }
                }, 30000);
            };

            ws.onmessage = (event) => {
                try {
                    // Ignore pong responses
                    if (event.data === 'pong') return;

                    const data = JSON.parse(event.data);
                    setState(prev => ({ ...prev, lastMessage: data }));
                    onMessage?.(data);
                } catch {
                    // Handle non-JSON messages
                    setState(prev => ({ ...prev, lastMessage: event.data }));
                }
            };

            ws.onclose = () => {
                setState(prev => ({ ...prev, isConnected: false }));
                onDisconnect?.();

                if (pingIntervalRef.current) {
                    clearInterval(pingIntervalRef.current);
                }

                // Attempt reconnection
                if (retriesRef.current < maxRetries) {
                    retriesRef.current += 1;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, reconnectInterval);
                } else {
                    setState(prev => ({
                        ...prev,
                        error: 'Max reconnection attempts reached',
                    }));
                }
            };

            ws.onerror = () => {
                setState(prev => ({ ...prev, error: 'WebSocket error' }));
            };

            wsRef.current = ws;
        } catch (error) {
            setState(prev => ({ ...prev, error: 'Failed to connect' }));
        }
    }, [url, onMessage, onConnect, onDisconnect, reconnectInterval, maxRetries]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        retriesRef.current = maxRetries; // Prevent auto-reconnect
    }, [maxRetries]);

    const sendMessage = useCallback((data: any) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    }, []);

    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        ...state,
        sendMessage,
        reconnect: connect,
        disconnect,
    };
};
