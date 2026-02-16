import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface ErrorBoundaryProps {
    children: React.ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { hasError: true, error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error('[ErrorBoundary] Caught error:', error, errorInfo);
    }

    handleReload = () => {
        this.setState({ hasError: false, error: null });
        window.location.reload();
    };

    handleRetry = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-void flex items-center justify-center p-6">
                    <div className="panel p-8 md:p-12 max-w-lg w-full text-center">
                        {/* Icon */}
                        <div className="w-14 h-14 rounded-2xl bg-crimson/10 border border-crimson/20 flex items-center justify-center mx-auto mb-6">
                            <AlertTriangle size={24} className="text-crimson" />
                        </div>

                        {/* Title */}
                        <h2 className="font-display text-2xl font-bold text-chalk mb-2">
                            Something went wrong
                        </h2>
                        <p className="text-smoke text-sm mb-6">
                            An unexpected error occurred. This has been logged for debugging.
                        </p>

                        {/* Error details (dev only) */}
                        {this.state.error && (
                            <div className="p-3 bg-void/60 rounded-xl border border-graphite/30 text-left mb-6">
                                <p className="text-[11px] text-smoke/50 uppercase tracking-wider mb-1 font-semibold">Error</p>
                                <code className="text-xs text-crimson/80 font-mono break-all">
                                    {this.state.error.message}
                                </code>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="flex items-center justify-center gap-3">
                            <button
                                onClick={this.handleRetry}
                                className="btn-ghost flex items-center gap-2 text-sm cursor-pointer"
                            >
                                <RotateCcw size={14} />
                                Try Again
                            </button>
                            <button
                                onClick={this.handleReload}
                                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-b from-amber to-amber/90 text-void shadow-lg shadow-amber/20 cursor-pointer transition-all hover:shadow-xl"
                            >
                                Reload Page
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
