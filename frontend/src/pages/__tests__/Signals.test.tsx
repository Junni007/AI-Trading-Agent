/**
 * Signals page tests — final version.
 * Note: In jsdom there are no CSS breakpoints, so the desktop table (hidden md:block)
 * is never visible. Tests target the mobile card view (md:hidden div) which is always
 * rendered in jsdom, OR use data that appears in both views.
 *
 * Skill refs: testing-patterns (AAA), webapp-testing (component-level)
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { Signals } from '../Signals';

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...p }: any) => React.createElement('div', p, children),
        tr: ({ children, ...p }: any) => React.createElement('tr', p, children),
    },
    AnimatePresence: ({ children }: any) => React.createElement(React.Fragment, null, children),
}));

// ── Fixtures ──────────────────────────────────────────────────────────────────

const makeSig = (ticker: string, action: string, confidence: number) => ({
    Ticker: ticker,
    Action: action,
    Confidence: confidence,
    Rational: ['Test reason A', 'Test reason B'],
    History: [],
    QuantRisk: { WinRate: 0.6, EV: 1.2, VaR95: 0.03 },
});

const SIGNALS = [
    makeSig('RELIANCE.NS', 'BUY', 0.85),
    makeSig('TCS.NS', 'WAIT', 0.55),
    makeSig('INFY.NS', 'WATCH_FOR_BREAKOUT', 0.62),
    makeSig('HDFCBANK.NS', 'IRON_CONDOR', 0.78),
];

function renderSigs(data = SIGNALS, props: any = {}) {
    return render(React.createElement(Signals, { data, ...props }));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Signals page', () => {

    describe('heading and subtitle', () => {
        it('renders "Active Signals" heading', () => {
            renderSigs();
            expect(screen.getByRole('heading', { name: /active signals/i })).toBeInTheDocument();
        });

        it('renders subtitle', () => {
            renderSigs();
            expect(screen.getByText(/real-time trading signals/i)).toBeInTheDocument();
        });
    });

    describe('empty state', () => {
        it('shows "No Active Signals" when data is empty', () => {
            renderSigs([]);
            expect(screen.getAllByText('No Active Signals').length).toBeGreaterThan(0);
        });

        it('shows skeleton rows when initial loading with no data', () => {
            renderSigs([], { loading: true });
            expect(document.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
        });
    });

    describe('filter tab counts', () => {
        it('All Signals tab shows total signal count (4)', () => {
            renderSigs(SIGNALS);
            const buttons = screen.getAllByRole('button');
            const allBtn = buttons.find(b => /all signals/i.test(b.textContent || ''));
            // Count badge is a sibling span with just '4'
            expect(allBtn?.textContent).toContain('4');
        });

        it('Active tab shows count 2 (BUY + IRON_CONDOR)', () => {
            renderSigs(SIGNALS);
            const buttons = screen.getAllByRole('button');
            const activeBtn = buttons.find(b => /\bactive\b/i.test(b.textContent || '') && !/all/i.test(b.textContent || ''));
            expect(activeBtn?.textContent).toContain('2');
        });

        it('Watching tab shows count 2 (WAIT + WATCH_FOR_BREAKOUT)', () => {
            renderSigs(SIGNALS);
            const buttons = screen.getAllByRole('button');
            const watchBtn = buttons.find(b => /watching/i.test(b.textContent || ''));
            expect(watchBtn?.textContent).toContain('2');
        });
    });

    describe('filter tab behaviour', () => {
        const findTab = (label: RegExp) => {
            const buttons = screen.getAllByRole('button');
            return buttons.find(b => label.test(b.textContent || ''));
        };

        it('Active filter hides non-actionable tickers', () => {
            renderSigs();
            fireEvent.click(findTab(/\bactive\b/i)!);
            // Tickers shown in either mobile card or desktop table — check both
            expect(screen.queryAllByText('TCS').length).toBe(0);
            expect(screen.queryAllByText('INFY').length).toBe(0);
        });

        it('Active filter keeps BUY and IRON_CONDOR visible', () => {
            renderSigs();
            fireEvent.click(findTab(/\bactive\b/i)!);
            // At least one of desktop table or mobile card renders the ticker
            expect(screen.queryAllByText('RELIANCE').length).toBeGreaterThan(0);
            expect(screen.queryAllByText('HDFCBANK').length).toBeGreaterThan(0);
        });

        it('Watching filter hides BUY and IRON_CONDOR', () => {
            renderSigs();
            fireEvent.click(findTab(/watching/i)!);
            expect(screen.queryAllByText('RELIANCE').length).toBe(0);
            expect(screen.queryAllByText('HDFCBANK').length).toBe(0);
        });

        it('Watching filter shows WAIT and WATCH_FOR_BREAKOUT tickers', () => {
            renderSigs();
            fireEvent.click(findTab(/watching/i)!);
            expect(screen.queryAllByText('TCS').length).toBeGreaterThan(0);
            expect(screen.queryAllByText('INFY').length).toBeGreaterThan(0);
        });

        it('shows "No Matches" when filtered tab has no results', () => {
            renderSigs([makeSig('ONLY.NS', 'BUY', 0.9)]);
            fireEvent.click(findTab(/watching/i)!);
            expect(screen.getAllByText('No Matches').length).toBeGreaterThan(0);
        });
    });

    describe('search', () => {
        it('search input exists and accepts text', () => {
            renderSigs();
            const input = screen.getByPlaceholderText(/search ticker/i);
            fireEvent.change(input, { target: { value: 'TCS' } });
            expect(input).toHaveValue('TCS');
        });

        it('search hides non-matching tickers', () => {
            renderSigs();
            fireEvent.change(screen.getByPlaceholderText(/search ticker/i), { target: { value: 'TCS' } });
            expect(screen.queryAllByText('RELIANCE').length).toBe(0);
        });

        it('search keeps matching ticker visible', () => {
            renderSigs();
            fireEvent.change(screen.getByPlaceholderText(/search ticker/i), { target: { value: 'TCS' } });
            expect(screen.queryAllByText('TCS').length).toBeGreaterThan(0);
        });

        it('shows No Matches on unmatched search', () => {
            renderSigs();
            fireEvent.change(screen.getByPlaceholderText(/search ticker/i), { target: { value: 'ZZZZ' } });
            expect(screen.getAllByText('No Matches').length).toBeGreaterThan(0);
        });

        it('clearing search restores all tickers', () => {
            renderSigs();
            const input = screen.getByPlaceholderText(/search ticker/i);
            fireEvent.change(input, { target: { value: 'TCS' } });
            fireEvent.change(input, { target: { value: '' } });
            expect(screen.queryAllByText('RELIANCE').length).toBeGreaterThan(0);
        });
    });

    describe('column sort', () => {
        it('default sort (Confidence DESC) puts highest-confidence first in mobile cards', () => {
            renderSigs();
            // Mobile cards: the confidence % appears in all cards
            const percentages = screen.getAllByText(/^\d+%$/);
            // First percent should be 85 (RELIANCE, highest)
            expect(percentages[0].textContent).toBe('85%');
        });

        it('clicking Confidence header toggles to ASC (lowest first)', () => {
            renderSigs();
            fireEvent.click(screen.getByText('Confidence'));
            const percentages = screen.getAllByText(/^\d+%$/);
            // After ASC, first should be 55% (TCS)
            expect(percentages[0].textContent).toBe('55%');
        });
    });

    describe('details callback', () => {
        it('calls onDetails when view-details button is clicked', () => {
            const onDetails = vi.fn();
            renderSigs(SIGNALS, { onDetails });
            const detailBtns = screen.getAllByRole('button', { name: /view details/i });
            fireEvent.click(detailBtns[0]);
            expect(onDetails).toHaveBeenCalledOnce();
        });

        it('passes the correct ticker to onDetails', () => {
            const onDetails = vi.fn();
            renderSigs(SIGNALS, { onDetails });
            const detailBtns = screen.getAllByRole('button', { name: /view details/i });
            fireEvent.click(detailBtns[0]);
            const [ticker] = onDetails.mock.calls[0];
            expect(ticker).toBe('RELIANCE.NS');
        });
    });
});
