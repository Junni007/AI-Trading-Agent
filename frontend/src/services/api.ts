import axios from 'axios';

/**
 * Centralized API client for Signal.Engine.
 * Base URL is configurable via VITE_API_URL environment variable.
 */
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
});

export default api;
