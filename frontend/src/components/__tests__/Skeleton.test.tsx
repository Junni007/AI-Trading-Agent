/**
 * Skeleton & EmptyState component tests.
 * Validates rendering, variants, and interactive behavior.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Skeleton, CardSkeleton, EmptyState } from '../Skeleton';
import { Search } from 'lucide-react';

describe('Skeleton', () => {
    it('renders with default rect variant', () => {
        const { container } = render(<Skeleton className="w-full h-10" />);
        const skeleton = container.firstChild as HTMLElement;
        expect(skeleton).toBeInTheDocument();
        expect(skeleton.className).toContain('rounded-xl');
    });

    it('renders circle variant with rounded-full', () => {
        const { container } = render(<Skeleton variant="circle" className="w-10 h-10" />);
        const skeleton = container.firstChild as HTMLElement;
        expect(skeleton.className).toContain('rounded-full');
    });

    it('renders text variant with rounded-md', () => {
        const { container } = render(<Skeleton variant="text" />);
        const skeleton = container.firstChild as HTMLElement;
        expect(skeleton.className).toContain('rounded-md');
    });
});

describe('CardSkeleton', () => {
    it('renders multiple skeleton elements inside a card', () => {
        const { container } = render(<CardSkeleton />);
        // Should have a panel wrapper with skeleton children
        const panel = container.firstChild as HTMLElement;
        expect(panel).toBeInTheDocument();
        expect(panel.className).toContain('panel');
    });
});

describe('EmptyState', () => {
    it('renders title and description', () => {
        render(
            <EmptyState
                icon={Search}
                title="No results found"
                description="Try a different search query."
            />
        );
        expect(screen.getByText('No results found')).toBeInTheDocument();
        expect(screen.getByText('Try a different search query.')).toBeInTheDocument();
    });

    it('renders action button when action + onAction provided', () => {
        const handleAction = vi.fn();
        render(
            <EmptyState
                icon={Search}
                title="No data"
                description="Start a scan."
                action="Run Scan"
                onAction={handleAction}
            />
        );
        const button = screen.getByText('Run Scan');
        expect(button).toBeInTheDocument();
        fireEvent.click(button);
        expect(handleAction).toHaveBeenCalledOnce();
    });

    it('does not render action button without onAction', () => {
        render(
            <EmptyState
                icon={Search}
                title="No data"
                description="Nothing here."
            />
        );
        expect(screen.queryByRole('button')).toBeNull();
    });
});
