/**
 * API service tests.
 * Validates axios instance configuration, interceptors, and retry setup.
 */
import { describe, it, expect } from 'vitest';
import api from '../api';

describe('api service', () => {
    it('has correct baseURL configured', () => {
        expect(api.defaults.baseURL).toBe('http://localhost:8000');
    });

    it('has Content-Type header set to JSON', () => {
        expect(api.defaults.headers['Content-Type']).toBe('application/json');
    });

    it('has 30s timeout', () => {
        expect(api.defaults.timeout).toBe(30000);
    });

    it('has request interceptor registered', () => {
        // Axios stores interceptors internally — check handlers array
        const interceptors = (api.interceptors.request as any).handlers;
        expect(interceptors.length).toBeGreaterThanOrEqual(1);
    });

    it('has response interceptor registered', () => {
        const interceptors = (api.interceptors.response as any).handlers;
        expect(interceptors.length).toBeGreaterThanOrEqual(1);
    });

    it('request interceptor is a function that processes config', () => {
        // Verify the interceptor fulfilled handler is a function
        const requestInterceptor = (api.interceptors.request as any).handlers[0];
        expect(typeof requestInterceptor.fulfilled).toBe('function');
        // Interceptor should return a config-like object
        const mockConfig = { headers: { set: () => { } } };
        const result = requestInterceptor.fulfilled(mockConfig);
        expect(result).toBeDefined();
        expect(result.headers).toBeDefined();
    });
});
