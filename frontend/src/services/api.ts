import axios from 'axios';
import axiosRetry from 'axios-retry';

/**
 * Centralized API client for Signal.Engine.
 * - Base URL via VITE_API_URL env
 * - API key via VITE_API_KEY env
 * - Automatic retry on 5xx / network errors (3x exponential backoff)
 * - Request ID tracking (X-Request-ID header)
 * - Global error logging with request ID
 */
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
        ...(import.meta.env.VITE_API_KEY && {
            'X-API-Key': import.meta.env.VITE_API_KEY,
        }),
    },
});

// ─── Retry: 3 attempts with exponential backoff on 5xx / network errors ─────
axiosRetry(api, {
    retries: 3,
    retryDelay: axiosRetry.exponentialDelay,
    retryCondition: (error) =>
        axiosRetry.isNetworkOrIdempotentRequestError(error) ||
        (error.response?.status ? error.response.status >= 500 : false),
});

// ─── Request interceptor: attach X-Request-ID for end-to-end tracing ────────
api.interceptors.request.use((config) => {
    config.headers['X-Request-ID'] = crypto.randomUUID();
    return config;
});

// ─── Response interceptor: global error logging ─────────────────────────────
api.interceptors.response.use(
    (res) => res,
    (error) => {
        const status = error.response?.status;
        const msg = error.response?.data?.detail || error.response?.data?.error || error.message;
        const reqId = error.response?.headers?.['x-request-id'];
        console.error(
            `[API ${status || 'NETWORK'}]${reqId ? ` req=${reqId}` : ''} ${msg}`
        );
        return Promise.reject(error);
    }
);

export default api;
