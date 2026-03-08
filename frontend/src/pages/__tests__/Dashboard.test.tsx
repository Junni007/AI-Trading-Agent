/**
 * Dashboard page tests.
 * Covers: empty state, loading skeletons, filter tabs, search, signal card rendering.
 *
 * Skill refs: testing-patterns (AAA + pyramid), webapp-testing (component-level)
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Dashboard } from '../Dashboard';

// ── Mocks ────────────────────────────────────────────────────────────────────

vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...p }: any) => <div {...p}>{children}</div>,
    },
    AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock('../../components/Hero', () => ({
    Hero: ({ marketMood }: any) => <div data-testid="hero">MOOD:{marketMood}</div>,
}));
vi.mock('../../components/SimulationPanel', () => ({
    SimulationPanel: () => <div data-testid="sim-panel" />,
}));
vi.mock('../../components/Terminal', () => ({
    Terminal: () => <div data-testid="terminal" />,
}));
vi.mock('../../components/ThinkingNode', () => ({
    ThinkingNode: ({ ticker }: any) => <div data-testid="thinking-node">{ticker}</div>,
}));

// ── Fixtures ──────────────────────────────────────────────────────────────────

const defaultProps = {
    data: [],
    loading: false,
    marketMood: 'NEUTRAL',
    lastUpdated: '',
    serverStartTime: null,
    simState: null,
    onResetSim: vi.fn(),
    onScan: vi.fn(),
    isAuto: true,
    isLocked: true,
    onDetails: vi.fn(),
    logs: [],
};

const makeDashboard = (overrides = {}) =>
    render(<Dashboard {...defaultProps} {...overrides} />);

const makeSignal = (ticker: string, action = 'BUY') => ({
    Ticker: ticker,
    Action: action,
    Confidence: 0.8,
    Rational: ['Test reason'],
    History: [],
    QuantRisk: { WinRate: 0.6, EV: 1.2, VaR95: 0.02 },
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('Dashboard page', () => {

    it('renders the Hero component', () => {
        makeDashboard();
        expect(screen.getByTestId('hero')).toBeInTheDocument();
    });

    it('passes marketMood from props to Hero', () => {
        makeDashboard({ marketMood: 'VOLATILE' });
        expect(screen.getByTestId('hero')).toHaveTextContent('MOOD:VOLATILE');
    });

    it('renders Terminal component', () => {
        makeDashboard();
        expect(screen.getByTestId('terminal')).toBeInTheDocument();
    });

    // ── Empty State ───────────────────────────────────────────────────────────

    describe('empty state', () => {
        it('shows "No Signals Yet" when data is empty and not loading', () => {
            makeDashboard({ data: [], loading: false });
            expect(screen.getByText('No Signals Yet')).toBeInTheDocument();
        });

        it('shows "Initialize Scan" action button in empty state', () => {
            makeDashboard({ data: [], loading: false });
            expect(screen.getByRole('button', { name: /initialize scan/i })).toBeInTheDocument();
        });

        it('calls onScan when Initialize Scan is clicked', () => {
            const onScan = vi.fn();
            makeDashboard({ data: [], loading: false, onScan });
            fireEvent.click(screen.getByRole('button', { name: /initialize scan/i }));
            expect(onScan).toHaveBeenCalledOnce();
        });
    });

    // ── Loading Skeletons ─────────────────────────────────────────────────────

    describe('loading state', () => {
        it('shows scanning message during initial load', () => {
            makeDashboard({ data: [], loading: true });
            expect(screen.getByText(/scanning markets/i)).toBeInTheDocument();
        });

        it('does not show empty state while loading', () => {
            makeDashboard({ data: [], loading: true });
            expect(screen.queryByText('No Signals Yet')).not.toBeInTheDocument();
        });
    });

    // ── With Data ─────────────────────────────────────────────────────────────

    describe('with signals data', () => {
        const signals = [
            makeSignal('RELIANCE.NS', 'BUY'),
            makeSignal('TCS.NS', 'WAIT'),
            makeSignal('INFY.NS', 'WATCH_FOR_BREAKOUT'),
        ];

        it('does not show empty state when data exists', () => {
            makeDashboard({ data: signals });
            expect(screen.queryByText('No Signals Yet')).not.toBeInTheDocument();
        });

        it('renders ThinkingNode for each signal', () => {
            makeDashboard({ data: signals });
            const nodes = screen.getAllByTestId('thinking-node');
            expect(nodes).toHaveLength(3);
        });

        it('renders filter tabs when data is present', () => {
            makeDashboard({ data: signals });
            expect(screen.getByText('All Signals')).toBeInTheDocument();
            expect(screen.getByText('Active')).toBeInTheDocument();
            expect(screen.getByText('Watching')).toBeInTheDocument();
        });

        it('renders search input when data is present', () => {
            makeDashboard({ data: signals });
            expect(screen.getByPlaceholderText(/search ticker/i)).toBeInTheDocument();
        });

        // ── Filter ────────────────────────────────────────────────────────────

        it('Active tab hides WAIT and WATCH_FOR_BREAKOUT signals', () => {
            makeDashboard({ data: signals });
            fireEvent.click(screen.getByText('Active'));
            const nodes = screen.getAllByTestId('thinking-node');
            expect(nodes).toHaveLength(1);
            expect(nodes[0]).toHaveTextContent('RELIANCE.NS');
        });

        it('Watching tab shows only non-actionable signals', () => {
            makeDashboard({ data: signals });
            fireEvent.click(screen.getByText('Watching'));
            const nodes = screen.getAllByTestId('thinking-node');
            expect(nodes).toHaveLength(2);
        });

        it('shows "No Matches" EmptyState when filter produces no results', () => {
            const buyOnly = [makeSignal('RELIANCE.NS', 'BUY')];
            makeDashboard({ data: buyOnly });
            fireEvent.click(screen.getByText('Watching'));
            expect(screen.getByText('No Matches')).toBeInTheDocument();
        });

        // ── Search ────────────────────────────────────────────────────────────

        it('search filters signal cards by ticker', () => {
            makeDashboard({ data: signals });
            fireEvent.change(screen.getByPlaceholderText(/search ticker/i), {
                target: { value: 'RELIANCE' },
            });
            expect(screen.getAllByTestId('thinking-node')).toHaveLength(1);
        });

        it('search clears restores all signals', () => {
            makeDashboard({ data: signals });
            const input = screen.getByPlaceholderText(/search ticker/i);
            fireEvent.change(input, { target: { value: 'TCS' } });
            fireEvent.change(input, { target: { value: '' } });
            expect(screen.getAllByTestId('thinking-node')).toHaveLength(3);
        });
    });
});
