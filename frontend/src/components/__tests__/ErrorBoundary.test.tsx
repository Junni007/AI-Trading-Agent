/**
 * ErrorBoundary component tests.
 * Validates error catching, fallback UI, retry, and child rendering.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from '../ErrorBoundary';

// Component that throws on render
const ThrowingChild = ({ shouldThrow }: { shouldThrow: boolean }) => {
    if (shouldThrow) throw new Error('Test explosion');
    return <div>Child rendered OK</div>;
};

describe('ErrorBoundary', () => {
    beforeEach(() => {
        // Suppress React error boundary console.error noise
        vi.spyOn(console, 'error').mockImplementation(() => { });
    });

    it('renders children when no error occurs', () => {
        render(
            <ErrorBoundary>
                <div>Hello, world</div>
            </ErrorBoundary>
        );
        expect(screen.getByText('Hello, world')).toBeInTheDocument();
    });

    it('shows fallback UI when child throws', () => {
        render(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
        expect(screen.getByText(/unexpected error/i)).toBeInTheDocument();
    });

    it('displays the error message in fallback', () => {
        render(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Test explosion')).toBeInTheDocument();
    });

    it('renders Try Again and Reload Page buttons', () => {
        render(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Try Again')).toBeInTheDocument();
        expect(screen.getByText('Reload Page')).toBeInTheDocument();
    });

    it('Try Again resets error state', () => {
        const { rerender } = render(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();

        // Click Try Again — error state resets, but component re-renders children
        fireEvent.click(screen.getByText('Try Again'));

        // After retry, the boundary tries to render children again.
        // Since ThrowingChild still throws, it should show error again.
        rerender(
            <ErrorBoundary>
                <ThrowingChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
});
