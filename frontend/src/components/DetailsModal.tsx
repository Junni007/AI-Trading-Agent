import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle2, Activity } from 'lucide-react';
import { AreaChart, Area, Tooltip, ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';

// Typewriter Component
const TypewriterText = ({ text, delay }: { text: string; delay: number }) => {
    const [currentText, setCurrentText] = useState('');
    const [startTyping, setStartTyping] = useState(false);

    useEffect(() => {
        const timeout = setTimeout(() => {
            setStartTyping(true);
        }, delay);
        return () => clearTimeout(timeout);
    }, [delay]);

    useEffect(() => {
        if (!startTyping) return;

        let index = 0;
        const interval = setInterval(() => {
            if (index < text.length) {
                setCurrentText((prev) => prev + text.charAt(index));
                index++;
            } else {
                clearInterval(interval);
            }
        }, 15);

        return () => clearInterval(interval);
    }, [startTyping, text]);

    return (
        <span>
            {currentText}
            {startTyping && currentText.length < text.length && (
                <span className="text-amber animate-pulse">▊</span>
            )}
        </span>
    );
};

interface HistoryPoint {
    Time: string;
    Close: number;
    Volume: number;
}

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    ticker: string;
    action: string;
    rational: string[];
    confidence: number;
    history?: HistoryPoint[];
}

export const DetailsModal = ({ isOpen, onClose, ticker, action, rational, confidence, history }: ModalProps) => {

    const isActionable = action !== 'WAIT' && action !== 'WATCH_FOR_BREAKOUT';

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] flex items-center justify-center p-4"
                >
                    {/* Backdrop */}
                    <div
                        className="absolute inset-0 bg-void/80 backdrop-blur-2xl"
                        onClick={onClose}
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                        onClick={(e) => e.stopPropagation()}
                        className="relative w-full max-w-2xl max-h-[85vh] panel overflow-hidden flex flex-col"
                    >
                        {/* Header */}
                        <div className="p-6 md:p-8 border-b border-graphite/50 flex justify-between items-start">
                            <div>
                                <h2 className="font-display text-3xl md:text-4xl font-bold text-chalk tracking-tight">
                                    {ticker.replace('.NS', '')}
                                </h2>
                                <div className="flex items-center gap-3 mt-2">
                                    <span className={`signal-badge ${isActionable
                                            ? 'bg-ember/10 text-ember border-ember/20'
                                            : 'bg-amber/10 text-amber border-amber/20'
                                        }`}>
                                        {action.replace(/_/g, ' ')}
                                    </span>
                                    <span className="text-sm text-smoke font-mono">
                                        {(confidence * 100).toFixed(0)}% conf.
                                    </span>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2.5 rounded-xl bg-obsidian hover:bg-slate text-smoke hover:text-chalk transition-colors border border-graphite/50"
                            >
                                <X size={18} />
                            </button>
                        </div>

                        {/* Chart Section */}
                        <div className="w-full h-56 bg-void/50 border-b border-graphite/50 relative">
                            {history && history.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={history} margin={{ top: 20, right: 20, bottom: 0, left: 20 }}>
                                        <defs>
                                            <linearGradient id="modalChartGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#141418',
                                                borderColor: '#2A2A32',
                                                borderRadius: '12px',
                                                color: '#F5F5F5',
                                                fontFamily: 'JetBrains Mono, monospace',
                                                fontSize: '12px'
                                            }}
                                            itemStyle={{ color: '#F59E0B' }}
                                            labelStyle={{ display: 'none' }}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="Close"
                                            stroke="#F59E0B"
                                            strokeWidth={2}
                                            fill="url(#modalChartGradient)"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex items-center justify-center h-full text-smoke text-sm gap-2">
                                    <Activity size={16} className="text-amber/50" />
                                    No price history available
                                </div>
                            )}
                        </div>

                        {/* Reasoning Section */}
                        <div className="p-6 md:p-8 space-y-5 overflow-y-auto flex-grow">
                            <h3 className="text-[10px] font-bold text-smoke uppercase tracking-[0.25em] font-mono">
                                Analysis Chain
                            </h3>
                            <div className="space-y-4">
                                {rational.map((step, idx) => {
                                    const isSolution = step.includes("Solution");
                                    return (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: idx * 1.2 }}
                                            className="flex items-start gap-4"
                                        >
                                            <div className={`mt-1 ${isSolution ? 'text-sage' : 'text-graphite'}`}>
                                                {isSolution ? (
                                                    <CheckCircle2 size={16} />
                                                ) : (
                                                    <div className="w-1.5 h-1.5 rounded-full bg-amber/40 mt-1.5 ml-0.5" />
                                                )}
                                            </div>
                                            <p className={`text-sm leading-relaxed font-mono ${isSolution ? 'text-sage font-semibold' : 'text-ash'
                                                }`}>
                                                <TypewriterText text={step} delay={idx * 1200} />
                                            </p>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
