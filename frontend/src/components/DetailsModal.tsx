import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle2, Activity, TrendingUp, TrendingDown, BarChart3, Shield, Target, ChevronDown, ChevronUp } from 'lucide-react';
import { Area, Bar, Tooltip, ResponsiveContainer, XAxis, YAxis, ComposedChart } from 'recharts';
import { useState } from 'react';

interface HistoryPoint {
    Time: string;
    Close: number;
    Volume: number;
}

interface QuantRisk {
    WinRate: number;
    EV: number;
    VaR95?: number;
    MaxDrawdown?: number;
}

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    ticker: string;
    action: string;
    rational: string[];
    confidence: number;
    history?: HistoryPoint[];
    quant?: QuantRisk;
}

// Risk gauge component
const RiskGauge = ({ label, value, maxValue, unit, color, description }: {
    label: string;
    value: number;
    maxValue: number;
    unit: string;
    color: 'sage' | 'amber' | 'crimson' | 'emerald';
    description: string;
}) => {
    const percentage = Math.min(100, (Math.abs(value) / maxValue) * 100);
    const colorMap = {
        sage: { bar: 'bg-sage', text: 'text-sage', glow: 'bg-sage/10' },
        amber: { bar: 'bg-amber', text: 'text-amber', glow: 'bg-amber/10' },
        crimson: { bar: 'bg-crimson', text: 'text-crimson', glow: 'bg-crimson/10' },
        emerald: { bar: 'bg-emerald-400', text: 'text-emerald-400', glow: 'bg-emerald-400/10' },
    };
    const c = colorMap[color];

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-xs text-smoke font-medium">{label}</span>
                <span className={`font-mono text-sm font-bold ${c.text}`}>
                    {value >= 0 ? '' : '-'}{Math.abs(value).toFixed(unit === '%' ? 1 : 2)}{unit}
                </span>
            </div>
            <div className="w-full h-2 bg-graphite/30 rounded-full overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
                    className={`h-full rounded-full ${c.bar}`}
                />
            </div>
            <p className="text-[10px] text-smoke/60">{description}</p>
        </div>
    );
};

// Quick stat pill
const StatPill = ({ label, value, accent }: { label: string; value: string; accent?: boolean }) => (
    <div className="flex flex-col items-center gap-0.5 px-3 py-2 bg-slate/30 rounded-xl border border-graphite/30 min-w-0 flex-1">
        <span className="text-[9px] text-smoke uppercase tracking-wider font-semibold whitespace-nowrap">{label}</span>
        <span className={`font-mono text-sm font-bold whitespace-nowrap ${accent ? 'text-amber' : 'text-chalk'}`}>{value}</span>
    </div>
);

export const DetailsModal = ({ isOpen, onClose, ticker, action, rational, confidence, history, quant }: ModalProps) => {
    const [showAllSteps, setShowAllSteps] = useState(false);

    const isActionable = action !== 'WAIT' && action !== 'WATCH_FOR_BREAKOUT';

    // Confidence arc
    const confidencePercent = Math.round(confidence * 100);
    const arcLength = 75.4;
    const arcOffset = arcLength - (arcLength * confidence);

    // Price calculations from history
    const latestPrice = history && history.length > 0 ? history[history.length - 1].Close : null;
    const firstPrice = history && history.length > 1 ? history[0].Close : null;
    const priceChange = latestPrice && firstPrice ? ((latestPrice - firstPrice) / firstPrice) * 100 : null;
    const isPositive = priceChange !== null ? priceChange >= 0 : true;

    // Chart color based on direction
    const chartColor = isPositive ? '#34D399' : '#F87171';
    const chartGradientId = `modalGrad-${ticker}`;

    // Determine visible steps
    const COLLAPSED_COUNT = 3;
    const shouldCollapse = rational.length > COLLAPSED_COUNT + 1;
    const visibleSteps = shouldCollapse && !showAllSteps
        ? rational.slice(0, COLLAPSED_COUNT)
        : rational;

    // Action badge styling
    const getActionBadge = () => {
        if (!isActionable) return { bg: 'bg-iron/10', text: 'text-iron', border: 'border-iron/20' };
        if (action.includes('CONDOR') || action.includes('SPREAD')) return { bg: 'bg-sage/10', text: 'text-sage', border: 'border-sage/20' };
        return { bg: 'bg-ember/10', text: 'text-ember', border: 'border-ember/20' };
    };
    const badge = getActionBadge();

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[100] flex items-end md:items-center justify-center"
                >
                    {/* Backdrop */}
                    <div
                        className="absolute inset-0 bg-void/85 backdrop-blur-2xl"
                        onClick={onClose}
                    />

                    {/* Modal — slide-up on mobile, centered on desktop */}
                    <motion.div
                        initial={{ opacity: 0, y: 40, scale: 0.97 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 40, scale: 0.97 }}
                        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                        onClick={(e) => e.stopPropagation()}
                        className="relative w-full md:max-w-2xl max-h-[92vh] md:max-h-[85vh] panel overflow-hidden flex flex-col rounded-t-2xl md:rounded-2xl"
                    >
                        {/* ===== SECTION 1: Hero Header ===== */}
                        <div className="p-5 md:p-6 border-b border-graphite/50 flex justify-between items-start gap-4">
                            <div className="flex items-center gap-4 min-w-0">
                                {/* Confidence Arc */}
                                <div className="relative flex items-center justify-center w-14 h-14 shrink-0">
                                    <svg className="absolute w-full h-full -rotate-90" viewBox="0 0 28 28">
                                        <circle cx="14" cy="14" r="12" fill="none" stroke="currentColor" strokeWidth="2" className="text-graphite" />
                                        <motion.circle
                                            cx="14" cy="14" r="12" fill="none" stroke="currentColor" strokeWidth="2.5"
                                            strokeDasharray={arcLength} strokeDashoffset={arcLength} strokeLinecap="round"
                                            animate={{ strokeDashoffset: arcOffset }}
                                            transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
                                            className={isActionable ? 'text-ember' : 'text-amber'}
                                        />
                                    </svg>
                                    <span className="font-mono text-sm font-bold text-chalk">{confidencePercent}</span>
                                </div>
                                {/* Ticker + Action */}
                                <div className="min-w-0">
                                    <h2 className="font-display text-2xl md:text-3xl font-bold text-chalk tracking-tight truncate">
                                        {ticker.replace('.NS', '')}
                                    </h2>
                                    <div className="flex items-center gap-2 mt-1.5">
                                        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-xs font-semibold border ${badge.bg} ${badge.text} ${badge.border}`}>
                                            {action.replace(/_/g, ' ')}
                                        </span>
                                        {quant && (
                                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[10px] font-bold border ${quant.WinRate > 0.6 ? 'bg-sage/10 text-sage border-sage/20' : quant.WinRate > 0.45 ? 'bg-amber/10 text-amber border-amber/20' : 'bg-crimson/10 text-crimson border-crimson/20'
                                                }`}>
                                                <Target size={10} />
                                                {(quant.WinRate * 100).toFixed(0)}% win
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 rounded-xl bg-obsidian hover:bg-slate text-smoke hover:text-chalk transition-colors border border-graphite/50 shrink-0 cursor-pointer"
                            >
                                <X size={16} />
                            </button>
                        </div>

                        {/* ===== SECTION 2: Quick Stats Strip ===== */}
                        <div className="px-5 md:px-6 py-3 border-b border-graphite/30 flex gap-2 overflow-x-auto">
                            {latestPrice !== null && (
                                <StatPill
                                    label="Price"
                                    value={`₹${latestPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                                />
                            )}
                            {priceChange !== null && (
                                <StatPill
                                    label="Change"
                                    value={`${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`}
                                    accent
                                />
                            )}
                            <StatPill
                                label="Confidence"
                                value={`${confidencePercent}%`}
                                accent
                            />
                            {quant && (
                                <StatPill
                                    label="Win Prob"
                                    value={`${(quant.WinRate * 100).toFixed(0)}%`}
                                />
                            )}
                        </div>

                        {/* Scrollable Content */}
                        <div className="overflow-y-auto flex-grow">

                            {/* ===== SECTION 3: Enhanced Chart ===== */}
                            <div className="w-full h-52 md:h-64 bg-void/30 border-b border-graphite/30 relative">
                                {history && history.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={history} margin={{ top: 16, right: 16, bottom: 0, left: 16 }}>
                                            <defs>
                                                <linearGradient id={chartGradientId} x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor={chartColor} stopOpacity={0.2} />
                                                    <stop offset="95%" stopColor={chartColor} stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <XAxis
                                                dataKey="Time"
                                                tick={false}
                                                axisLine={false}
                                                tickLine={false}
                                            />
                                            <YAxis
                                                yAxisId="price"
                                                domain={['auto', 'auto']}
                                                tick={false}
                                                axisLine={false}
                                                tickLine={false}
                                                width={0}
                                            />
                                            <YAxis
                                                yAxisId="volume"
                                                orientation="right"
                                                domain={[0, (max: number) => max * 4]}
                                                tick={false}
                                                axisLine={false}
                                                tickLine={false}
                                                width={0}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#0C0C10',
                                                    borderColor: '#2A2A32',
                                                    borderRadius: '12px',
                                                    color: '#F5F5F5',
                                                    fontFamily: 'JetBrains Mono, monospace',
                                                    fontSize: '11px',
                                                    padding: '8px 12px'
                                                }}
                                                itemStyle={{ color: chartColor }}
                                                labelFormatter={(label: string) => {
                                                    try {
                                                        return new Date(label).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
                                                    } catch { return label; }
                                                }}
                                                formatter={(value: number, name: string) => {
                                                    if (name === 'Close') return [`₹${value.toLocaleString()}`, 'Price'];
                                                    if (name === 'Volume') return [value.toLocaleString(), 'Vol'];
                                                    return [value, name];
                                                }}
                                            />
                                            <Bar
                                                yAxisId="volume"
                                                dataKey="Volume"
                                                fill="#2A2A32"
                                                opacity={0.4}
                                                radius={[2, 2, 0, 0]}
                                            />
                                            <Area
                                                yAxisId="price"
                                                type="monotone"
                                                dataKey="Close"
                                                stroke={chartColor}
                                                strokeWidth={2}
                                                fill={`url(#${chartGradientId})`}
                                            />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-full text-smoke text-sm gap-2">
                                        <Activity size={20} className="text-amber/30" />
                                        <span>No price history available</span>
                                        <span className="text-[10px] text-smoke/50">Chart data loads after a scan completes</span>
                                    </div>
                                )}
                                {/* Price label overlay */}
                                {latestPrice !== null && (
                                    <div className="absolute top-3 right-4 flex items-center gap-1.5">
                                        {isPositive ? (
                                            <TrendingUp size={12} className="text-emerald-400" />
                                        ) : (
                                            <TrendingDown size={12} className="text-red-400" />
                                        )}
                                        <span className={`font-mono text-xs font-bold ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                                            ₹{latestPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* ===== SECTION 4: Risk Assessment ===== */}
                            {quant && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.15 }}
                                    className="px-5 md:px-6 py-5 border-b border-graphite/30"
                                >
                                    <h3 className="text-[10px] font-bold text-smoke uppercase tracking-[0.25em] font-mono mb-4 flex items-center gap-2">
                                        <Shield size={12} className="text-amber" />
                                        Risk Assessment
                                    </h3>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        <RiskGauge
                                            label="Win Probability"
                                            value={quant.WinRate * 100}
                                            maxValue={100}
                                            unit="%"
                                            color={quant.WinRate > 0.6 ? 'sage' : quant.WinRate > 0.45 ? 'amber' : 'crimson'}
                                            description="Monte Carlo simulation win rate"
                                        />
                                        <RiskGauge
                                            label="Expected Value"
                                            value={quant.EV}
                                            maxValue={Math.max(Math.abs(quant.EV) * 2, 100)}
                                            unit=""
                                            color={quant.EV > 0 ? 'emerald' : 'crimson'}
                                            description="Expected portfolio impact"
                                        />
                                        {quant.VaR95 !== undefined && (
                                            <RiskGauge
                                                label="Value at Risk (95%)"
                                                value={quant.VaR95}
                                                maxValue={Math.max(quant.VaR95 * 2, 500)}
                                                unit=""
                                                color={quant.VaR95 < 200 ? 'amber' : 'crimson'}
                                                description="Max loss at 95% confidence"
                                            />
                                        )}
                                        {quant.MaxDrawdown !== undefined && (
                                            <RiskGauge
                                                label="Max Drawdown"
                                                value={quant.MaxDrawdown}
                                                maxValue={50}
                                                unit="%"
                                                color={quant.MaxDrawdown < 10 ? 'sage' : quant.MaxDrawdown < 25 ? 'amber' : 'crimson'}
                                                description="Largest historical decline"
                                            />
                                        )}
                                    </div>
                                </motion.div>
                            )}

                            {/* ===== SECTION 5: Analysis Chain ===== */}
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.25 }}
                                className="px-5 md:px-6 py-5"
                            >
                                <h3 className="text-[10px] font-bold text-smoke uppercase tracking-[0.25em] font-mono mb-4 flex items-center gap-2">
                                    <BarChart3 size={12} className="text-amber" />
                                    Analysis Chain
                                    <span className="text-smoke/40 font-normal normal-case tracking-normal ml-1">
                                        {rational.length} step{rational.length !== 1 ? 's' : ''}
                                    </span>
                                </h3>
                                <div className="space-y-3">
                                    {visibleSteps.map((step, idx) => {
                                        const isSolution = step.includes("Solution");
                                        return (
                                            <motion.div
                                                key={idx}
                                                initial={{ opacity: 0, x: -8 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: idx * 0.05 }}
                                                className="flex items-start gap-3"
                                            >
                                                {/* Step number */}
                                                <div className={`mt-0.5 w-5 h-5 rounded-md flex items-center justify-center shrink-0 text-[10px] font-bold ${isSolution
                                                    ? 'bg-sage/15 text-sage border border-sage/20'
                                                    : 'bg-slate/50 text-smoke border border-graphite/30'
                                                    }`}>
                                                    {isSolution ? <CheckCircle2 size={10} /> : idx + 1}
                                                </div>
                                                <p className={`text-sm leading-relaxed font-body ${isSolution ? 'text-sage font-semibold' : 'text-ash'
                                                    }`}>
                                                    {step}
                                                </p>
                                            </motion.div>
                                        );
                                    })}
                                </div>

                                {/* Show more / Show less toggle */}
                                {shouldCollapse && (
                                    <button
                                        onClick={() => setShowAllSteps(!showAllSteps)}
                                        className="mt-3 flex items-center gap-1.5 text-xs text-amber hover:text-chalk transition-colors cursor-pointer group"
                                    >
                                        {showAllSteps ? (
                                            <>
                                                <ChevronUp size={12} className="group-hover:-translate-y-0.5 transition-transform" />
                                                Show less
                                            </>
                                        ) : (
                                            <>
                                                <ChevronDown size={12} className="group-hover:translate-y-0.5 transition-transform" />
                                                Show all {rational.length} steps
                                            </>
                                        )}
                                    </button>
                                )}
                            </motion.div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};
