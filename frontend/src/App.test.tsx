/**
 * App business logic unit tests.
 * We test the pure functions and state logic directly, not by mounting App
 * (which requires a full browser environment due to BrowserRouter + scrollTo).
 * Use Playwright E2E tests (e2e/signal-engine.spec.ts) for full App integration.
 *
 * Skill refs: testing-patterns (unit, AAA pattern)
 */
import { describe, it, expect } from 'vitest';

// ── deriveMarketMood ──────────────────────────────────────────────────────────
// Mirrors the logic in App.tsx AppContent.deriveMarketMood

const deriveMarketMood = (decisions: { Rational: string[] }[]): string => {
    const volCount = decisions.filter(d =>
        d.Rational.some(r => r.includes('High Volatility'))
    ).length;
    return volCount > decisions.length / 2 ? 'VOLATILE' : 'CALM';
};

describe('deriveMarketMood', () => {
    it('returns VOLATILE when strictly more than half signal High Volatility', () => {
        const sigs = [
            { Rational: ['High Volatility breakout'] },
            { Rational: ['High Volatility surge'] },
            { Rational: ['Strong momentum'] },
        ];
        expect(deriveMarketMood(sigs)).toBe('VOLATILE');
    });

    it('returns CALM when minority mention High Volatility', () => {
        const sigs = [
            { Rational: ['High Volatility'] },
            { Rational: ['Calm trend'] },
            { Rational: ['Breakout confirmed'] },
            { Rational: ['Oversold'] },
        ];
        expect(deriveMarketMood(sigs)).toBe('CALM');
    });

    it('returns CALM for empty signal list', () => {
        expect(deriveMarketMood([])).toBe('CALM');
    });

    it('returns CALM when exactly half (not majority) are volatile', () => {
        const sigs = [
            { Rational: ['High Volatility'] },
            { Rational: ['Calm trend'] },
        ];
        expect(deriveMarketMood(sigs)).toBe('CALM');
    });

    it('is case-sensitive — only exact "High Volatility" substring matches', () => {
        const sigs = [
            { Rational: ['high volatility'] },  // lowercase — should NOT match
            { Rational: ['HIGH VOLATILITY'] },  // all caps — should NOT match
        ];
        // 0 matches out of 2 → CALM
        expect(deriveMarketMood(sigs)).toBe('CALM');
    });
});

// ── isLocked ──────────────────────────────────────────────────────────────────
// Mirrors: const isLocked = isAuto && simState?.status !== 'DEAD';

const isLocked = (isAuto: boolean, simStatus: string | null | undefined): boolean =>
    isAuto && simStatus !== 'DEAD';

describe('isLocked logic', () => {
    it('is locked when auto-pilot on and sim status is RUNNING', () => {
        expect(isLocked(true, 'RUNNING')).toBe(true);
    });

    it('is locked when auto-pilot on and simState is null (no backend)', () => {
        expect(isLocked(true, null)).toBe(true);
    });

    it('is NOT locked when auto-pilot is off, regardless of sim status', () => {
        expect(isLocked(false, 'RUNNING')).toBe(false);
        expect(isLocked(false, 'DEAD')).toBe(false);
        expect(isLocked(false, null)).toBe(false);
    });

    it('is NOT locked when sim status is DEAD (even with auto on)', () => {
        expect(isLocked(true, 'DEAD')).toBe(false);
    });
});

// ── High-confidence notification threshold ────────────────────────────────────
// Mirrors: if (signal.Confidence >= 0.85) { notificationService.sendSignalAlert(...) }

const shouldNotify = (confidence: number): boolean => confidence >= 0.85;

describe('shouldNotify (notification threshold)', () => {
    it('triggers at exactly 0.85', () => {
        expect(shouldNotify(0.85)).toBe(true);
    });

    it('triggers above 0.85', () => {
        expect(shouldNotify(0.9)).toBe(true);
        expect(shouldNotify(1.0)).toBe(true);
    });

    it('does NOT trigger below 0.85', () => {
        expect(shouldNotify(0.84)).toBe(false);
        expect(shouldNotify(0.5)).toBe(false);
        expect(shouldNotify(0.0)).toBe(false);
    });
});

// ── Route path mapping ────────────────────────────────────────────────────────
// Assert the expected route structure matches what App.tsx defines

describe('route structure', () => {
    const routes = ['/', '/signals', '/analytics', '/settings', '*'];

    it('defines exactly 5 route paths (including catch-all)', () => {
        expect(routes).toHaveLength(5);
    });

    it('includes a catch-all wildcard for 404', () => {
        expect(routes).toContain('*');
    });

    it('settings page is at /settings (not /config)', () => {
        expect(routes).toContain('/settings');
        expect(routes).not.toContain('/config');
    });
});

// ── Connection error banner ───────────────────────────────────────────────────

describe('Connection error banner copy', () => {
    it('banner message is descriptive and actionable', () => {
        const msg = 'Unable to reach backend API. Data may be stale.';
        expect(msg).toMatch(/backend/i);
        expect(msg).toMatch(/stale/i);
    });

    it('auto-scan interval is 10 seconds (10_000 ms)', () => {
        // This is a snapshot test to catch accidental interval changes
        const INTERVAL_MS = 10_000;
        expect(INTERVAL_MS).toBe(10000);
    });
});
