import React from 'react';
import { motion } from 'framer-motion';
import { ArrowUpRight, Shield, Eye, Zap } from 'lucide-react';

interface ThinkingNodeProps {
    ticker: string;
    regime: string;
    action: string;
    confidence: number;
    steps: string[];
    history?: any[];
    quant?: { WinRate: number };
    onDetails: (ticker: string, action: string, steps: string[], confidence: number, history: any[]) => void;
}

export const ThinkingNode: React.FC<ThinkingNodeProps> = ({ ticker, regime, action, confidence, steps, history, quant, onDetails }) => {

    // Action classification
    const isIncome = action.includes('CONDOR') || action.includes('SPREAD');
    const isWatch = action === 'WATCH_FOR_BREAKOUT' || action === 'WAIT';

    // Signal styling based on action type
    const getSignalStyles = () => {
        if (isWatch) {
            return {
                accent: 'amber',
                badgeClass: 'signal-badge bg-amber/10 text-amber border-amber/20',
                glowClass: 'hover:shadow-amber/10',
                icon: <Eye size={12} />,
            };
        }
        if (isIncome) {
            return {
                accent: 'sage',
                badgeClass: 'signal-badge bg-sage/10 text-sage border-sage/20',
                glowClass: 'hover:shadow-sage/10',
                icon: <Shield size={12} />,
            };
        }
        return {
            accent: 'ember',
            badgeClass: 'signal-badge bg-ember/10 text-ember border-ember/20',
            glowClass: 'hover:shadow-ember/10',
            icon: <Zap size={12} />,
        };
    };

    const signal = getSignalStyles();

    // Confidence arc calculation
    const confidencePercent = Math.round(confidence * 100);
    const arcLength = 75.4; // Circumference of r=12 circle
    const arcOffset = arcLength - (arcLength * confidence);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className={`group relative flex flex-col justify-between panel p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl ${signal.glowClass}`}
        >
            {/* High Confidence Indicator */}
            {confidence >= 0.8 && (
                <motion.div
                    className={`absolute top-3 right-3 w-2 h-2 rounded-full bg-${signal.accent}`}
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                />
            )}

            {/* Header */}
            <div className="flex justify-between items-start mb-5">
                <div>
                    <h3 className="font-display text-2xl font-bold text-chalk tracking-tight leading-none group-hover:text-amber transition-colors">
                        {ticker.replace('.NS', '')}
                    </h3>
                    <div className="flex items-center gap-2 mt-2.5">
                        <span className="signal-badge-muted">
                            {regime.split(" ")[0]}
                        </span>
                        {steps.some(s => s.includes("RL Agent")) && (
                            <span className="signal-badge bg-violet-500/10 text-violet-400 border border-violet-500/20">
                                AI
                            </span>
                        )}
                        {quant && (
                            <span className={`signal-badge ${quant.WinRate > 0.5 ? 'bg-sage/10 text-sage border-sage/20' : 'bg-crimson/10 text-crimson border-crimson/20'}`}>
                                {(quant.WinRate * 100).toFixed(0)}%
                            </span>
                        )}
                    </div>
                </div>

                {/* Confidence Arc */}
                <div className="relative flex items-center justify-center w-14 h-14">
                    <svg className="absolute w-full h-full -rotate-90" viewBox="0 0 28 28">
                        <circle
                            cx="14"
                            cy="14"
                            r="12"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className="text-graphite"
                        />
                        <circle
                            cx="14"
                            cy="14"
                            r="12"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeDasharray={arcLength}
                            strokeDashoffset={arcOffset}
                            strokeLinecap="round"
                            className={`text-${signal.accent}`}
                        />
                    </svg>
                    <span className="font-mono text-sm font-bold text-chalk">{confidencePercent}</span>
                </div>
            </div>

            {/* Action Badge */}
            <div className={`${signal.badgeClass} w-full justify-center py-2 mb-4`}>
                {signal.icon}
                <span className="ml-1">{action.replace(/_/g, ' ')}</span>
            </div>

            {/* Rational Snippets */}
            <div className="space-y-2 mb-5 flex-grow">
                {steps.slice(-2).map((step, idx) => (
                    <div key={idx} className="flex items-start gap-2.5">
                        <div className={`mt-1.5 w-1 h-1 rounded-full bg-${signal.accent}/60`} />
                        <p className="text-xs text-ash leading-relaxed line-clamp-2 font-body">
                            {step}
                        </p>
                    </div>
                ))}
            </div>

            {/* Action Button */}
            <button
                onClick={() => onDetails(ticker, action, steps, confidence, history || [])}
                className="w-full py-3 rounded-xl btn-ghost flex items-center justify-center gap-2 group/btn"
            >
                <span className="text-xs font-semibold uppercase tracking-widest">Deep Dive</span>
                <ArrowUpRight size={14} className="text-smoke group-hover/btn:text-amber transition-colors" />
            </button>
        </motion.div>
    );
};
