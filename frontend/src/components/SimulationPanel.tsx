import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, TrendingDown, RotateCcw } from 'lucide-react';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';

interface SimState {
    balance: number;
    cash: number;
    score: number;
    level: string;
    positions: Record<string, any>;
    history: string[];
    sharpe_ratio?: number;
    max_drawdown?: number;
    equity_curve?: number[];
}

interface SimulationPanelProps {
    simState: SimState | null;
    onReset: () => void;
}

export const SimulationPanel: React.FC<SimulationPanelProps> = ({ simState, onReset }) => {
    if (!simState) return null;

    const pnl = simState.balance - 10000;
    const isProfit = pnl >= 0;
    const pnlPercent = ((pnl / 10000) * 100).toFixed(2);

    // Prepare chart data
    const chartData = simState.equity_curve?.map((val, i) => ({ i, val })) || [];

    // Level calculation
    const level = simState.score > 0 ? Math.floor(simState.score / 20) + 1 : 1;
    const xpProgress = Math.min(100, (simState.score % 50) * 2);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="w-full max-w-[1400px] px-4 mb-6"
        >
            <div className="panel-elevated p-6 md:p-8 relative overflow-hidden">
                {/* Ambient glow */}
                <div className="absolute top-0 right-0 w-80 h-80 bg-amber/5 rounded-full blur-[100px] -mr-40 -mt-40 pointer-events-none" />

                <div className="flex flex-col lg:flex-row justify-between items-start gap-8 relative z-10">

                    {/* Left: Level & Status */}
                    <div className="flex flex-col gap-5">
                        <div className="flex items-center gap-5">
                            {/* Level Badge */}
                            <div className="relative">
                                <div className="h-16 w-16 rounded-2xl bg-void border border-graphite flex items-center justify-center">
                                    <Trophy className="text-amber" size={24} />
                                </div>
                                <div className="absolute -bottom-2 -right-2 px-2 py-0.5 bg-amber text-void text-[10px] font-bold rounded-full font-mono">
                                    LV.{level}
                                </div>
                            </div>

                            <div>
                                <h3 className="font-display text-xl font-bold text-chalk tracking-tight">
                                    {simState.level}
                                </h3>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className="text-xs font-mono text-smoke">XP</span>
                                    <span className={`font-mono font-bold ${simState.score >= 0 ? 'text-amber' : 'text-crimson'}`}>
                                        {simState.score} pts
                                    </span>
                                </div>

                                {/* XP Progress Bar */}
                                <div className="w-40 h-1 bg-graphite rounded-full mt-2.5 overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-amber to-ember"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${xpProgress}%` }}
                                        transition={{ duration: 0.8, ease: "easeOut" }}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Reset Button */}
                        <button
                            onClick={onReset}
                            className="btn-ghost flex items-center gap-2 w-fit text-xs cursor-pointer"
                        >
                            <RotateCcw size={12} />
                            Reset Simulation
                        </button>
                    </div>

                    {/* Right: Balance & Chart */}
                    <div className="flex flex-col items-end w-full lg:w-1/2">
                        {/* Balance */}
                        <div className="font-mono text-4xl md:text-5xl font-bold text-chalk tracking-tighter">
                            ₹{simState.balance.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </div>

                        {/* PnL Badge */}
                        <div className={`flex items-center gap-2 mt-2 px-3 py-1.5 rounded-lg ${isProfit ? 'bg-sage/10 text-sage' : 'bg-crimson/10 text-crimson'
                            }`}>
                            {isProfit ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                            <span className="font-mono font-semibold text-sm">
                                {isProfit ? '+' : ''}₹{pnl.toFixed(0)} ({pnlPercent}%)
                            </span>
                        </div>

                        {/* Advanced Metrics */}
                        <div className="flex gap-5 mt-3 text-[10px] font-mono text-smoke uppercase tracking-widest">
                            <div title="Sharpe Ratio">
                                SR: <span className="text-chalk">{simState.sharpe_ratio?.toFixed(2) || '0.00'}</span>
                            </div>
                            <div title="Max Drawdown">
                                DD: <span className="text-crimson">{simState.max_drawdown?.toFixed(1) || '0.0'}%</span>
                            </div>
                        </div>

                        {/* Mini Chart */}
                        <div className="w-full h-28 mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.25} />
                                            <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <Area
                                        type="monotone"
                                        dataKey="val"
                                        stroke="#F59E0B"
                                        strokeWidth={2}
                                        fill="url(#equityGradient)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};
