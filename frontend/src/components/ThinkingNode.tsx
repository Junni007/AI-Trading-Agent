import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ShieldCheck, AlertTriangle, Zap } from 'lucide-react';

interface ThinkingNodeProps {
    ticker: string;
    regime: string;
    action: string;
    confidence: number;
    steps: string[];
    history?: any[];
    onDetails: (ticker: string, action: string, steps: string[], confidence: number, history: any[]) => void;
}

export const ThinkingNode: React.FC<ThinkingNodeProps> = ({ ticker, regime, action, confidence, steps, history, onDetails }) => {

    // Dynamic Styles based on Action
    const isBullish = action.includes('LONG') || action.includes('AMZN'); // Basic heuristic
    const isIncome = action.includes('CONDOR') || action.includes('SPREAD');
    const isWatch = action === 'WATCH_FOR_BREAKOUT' || action === 'WAIT';

    let accentColor = "text-teal";
    let badgeStyle = "bg-teal/10 text-teal border-teal/20";
    let glowStyle = "group-hover:shadow-[0_0_30px_rgba(0,173,181,0.15)]";

    if (isWatch) {
        accentColor = "text-yellow-400";
        badgeStyle = "bg-yellow-400/10 text-yellow-400 border-yellow-400/20";
        glowStyle = "group-hover:shadow-[0_0_30px_rgba(250,204,21,0.15)]";
    } else if (isIncome) {
        accentColor = "text-indigo-400";
        badgeStyle = "bg-indigo-400/10 text-indigo-400 border-indigo-400/20";
        glowStyle = "group-hover:shadow-[0_0_30px_rgba(129,140,248,0.15)]";
    }

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`group relative flex flex-col justify-between bg-gunmetal/40 backdrop-blur-xl rounded-3xl border border-white/5 p-6 transition-all duration-300 hover:-translate-y-1 hover:bg-gunmetal/60 ${glowStyle} overflow-hidden`}
        >
            {/* Holographic Border for High Confidence */}
            {confidence >= 0.8 && (
                <>
                    <motion.div
                        className="absolute inset-0 rounded-3xl border-2 border-transparent"
                        style={{ background: `linear-gradient(45deg, ${isBullish ? '#00ADB5' : '#818CF8'}, transparent 40%) border-box` }}
                        animate={{ rotate: [0, 360] }}
                        transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
                    />
                    <div className="absolute inset-0 rounded-3xl border border-white/10 shadow-[inner_0_0_20px_rgba(255,255,255,0.05)] pointer-events-none" />
                </>
            )}
            {/* Header: Ticker & Score */}
            <div className="flex justify-between items-start mb-6">
                <div>
                    <h3 className="text-3xl font-bold text-white tracking-tight leading-none group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-gray-400 transition-all">
                        {ticker.replace('.NS', '')}
                    </h3>
                    <div className="flex items-center gap-2 mt-2">
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider border ${badgeStyle}`}>
                            {regime.split(" ")[0]}
                        </span>
                    </div>
                </div>

                {/* Confidence Dial (Visual) */}
                <div className="relative flex flex-col items-center">
                    <span className={`text-3xl font-bold tracking-tighter ${accentColor}`}>
                        {(confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-[10px] text-gray-500 font-medium uppercase tracking-widest">Conf.</span>
                </div>
            </div>

            {/* Body: Rational Snippet */}
            <div className="space-y-3 mb-6 flex-grow">
                {/* Main Action Badge */}
                <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-opacity-5 ${badgeStyle} w-full justify-center`}>
                    {isIncome && <ShieldCheck size={14} />}
                    {isWatch && <AlertTriangle size={14} />}
                    {isBullish && <Zap size={14} />}
                    <span className="text-xs font-bold uppercase tracking-wide">
                        {action.replace(/_/g, ' ')}
                    </span>
                </div>

                {/* Snippets */}
                <div className="space-y-2 pt-2">
                    {steps.slice(-2).map((step, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                            <div className={`mt-1.5 w-1 h-1 rounded-full ${isWatch ? 'bg-yellow-500' : 'bg-teal'}`} />
                            <p className="text-xs text-gray-400 leading-relaxed font-light line-clamp-2">
                                {step}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer: Action Button */}
            <button
                onClick={() => onDetails(ticker, action, steps, confidence, history || [])}
                className="w-full py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/20 transition-all flex items-center justify-center gap-2 group/btn"
            >
                <span className="text-xs font-semibold text-gray-300 group-hover/btn:text-white uppercase tracking-wider">Analyze</span>
                <ArrowRight size={14} className="text-gray-500 group-hover/btn:text-white transition-colors" />
            </button>
        </motion.div>
    );
};
