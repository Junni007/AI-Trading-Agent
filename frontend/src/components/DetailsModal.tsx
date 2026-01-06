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
        }, 20); // Typing speed

        return () => clearInterval(interval);
    }, [startTyping, text]);

    return <span>{currentText}{startTyping && currentText.length < text.length ? <span className="animate-pulse">|</span> : ''}</span>;
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
    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Apple-style Backdrop (Ultra Blur) */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-black/40 backdrop-blur-xl z-[100] flex items-center justify-center p-4 cursor-default"
                    >
                        {/* Modal Content (Glass Card) */}
                        <motion.div
                            initial={{ scale: 0.95, opacity: 0, y: 10 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.95, opacity: 0, y: 10 }}
                            onClick={(e) => e.stopPropagation()}
                            className="w-full max-w-2xl bg-gunmetal/80 backdrop-blur-2xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden relative flex flex-col max-h-[90vh]"
                        >
                            {/* Header */}
                            <div className="p-8 border-b border-white/5 flex justify-between items-start">
                                <div>
                                    <h2 className="text-4xl font-bold text-white tracking-tight mb-2">{ticker}</h2>
                                    <div className="flex items-center gap-3">
                                        <span className={`px-3 py-1 rounded-full text-xs font-semibold tracking-wide border ${action === 'WAIT'
                                            ? 'bg-gray-500/10 border-gray-500/20 text-gray-400'
                                            : 'bg-teal/10 border-teal/20 text-teal shadow-[0_0_10px_rgba(0,173,181,0.2)]'
                                            }`}>
                                            {action.replace(/_/g, ' ')}
                                        </span>
                                        <span className="text-sm text-gray-400 font-medium">Confidence: {(confidence * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="p-2 bg-white/5 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-all"
                                >
                                    <X size={20} />
                                </button>
                            </div>

                            {/* Chart Section (New Features) */}
                            <div className="w-full h-64 bg-marine/30 relative">
                                {history && history.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={history}>
                                            <defs>
                                                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#00ADB5" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#00ADB5" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#222831', borderColor: '#333', borderRadius: '8px', color: '#fff' }}
                                                itemStyle={{ color: '#00ADB5' }}
                                                labelStyle={{ display: 'none' }}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="Close"
                                                stroke="#00ADB5"
                                                fillOpacity={1}
                                                fill="url(#colorPrice)"
                                                strokeWidth={2}
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                                        <Activity size={16} className="mr-2" /> No Chart Data Available
                                    </div>
                                )}
                            </div>

                            {/* Logic Section */}
                            <div className="p-8 space-y-6 overflow-y-auto">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-[0.2em]">Thinking Process</h3>
                                <div className="space-y-4">
                                    {rational.map((step, idx) => (
                                        <motion.div
                                            key={idx}
                                            // Make the container fade in slightly before typing
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: idx * 1.5 }} // Stagger blocks
                                            className="flex items-start gap-4"
                                        >
                                            <div className={`mt-1 min-w-[20px] ${step.includes("Solution") ? "text-teal" : "text-gray-600"}`}>
                                                {step.includes("Solution") ? <CheckCircle2 size={20} /> : <div className="w-2 h-2 rounded-full bg-gray-600 mt-2 ml-1" />}
                                            </div>
                                            <p className={`text-sm leading-relaxed font-mono ${step.includes("Solution") ? "text-teal font-bold" : "text-gray-300"}`}>
                                                <TypewriterText text={step} delay={idx * 1500} />
                                            </p>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};
