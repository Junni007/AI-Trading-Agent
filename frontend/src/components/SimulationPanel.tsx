import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, AlertTriangle, RefreshCcw } from 'lucide-react';

import { AreaChart, Area, Tooltip, ResponsiveContainer } from 'recharts';

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

    // Format Data for Chart
    const chartData = simState.equity_curve?.map((val, i) => ({ i, val })) || [];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-[1400px] px-4 mb-8"
        >
            <div className="bg-gradient-to-r from-gunmetal to-marine border border-white/5 rounded-3xl p-6 md:p-8 relative overflow-hidden shadow-2xl">
                {/* Background Glow */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-teal/10 rounded-full blur-[100px] -mr-20 -mt-20" />

                <div className="flex flex-col md:flex-row justify-between items-start gap-8 relative z-10">

                    {/* Left Side: Stats */}
                    <div className="flex flex-col justify-between h-full gap-6">
                        {/* Level & Score Section */}
                        <div className="flex items-center gap-6">
                            <div className="h-20 w-20 rounded-2xl bg-black/40 border border-teal/20 flex items-center justify-center relative">
                                <Trophy className="text-teal drop-shadow-[0_0_10px_rgba(0,173,181,0.5)]" size={32} />
                                <div className="absolute -bottom-2 px-3 py-1 bg-teal text-marine text-[10px] font-bold rounded-full uppercase tracking-wider">
                                    Lvl {simState.score > 0 ? Math.floor(simState.score / 20) + 1 : 1}
                                </div>
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold text-white tracking-tight">{simState.level}</h3>
                                <div className="flex items-center gap-2 mt-1">
                                    <span className="text-sm text-gray-400 font-mono">XP Score:</span>
                                    <span className={`text-lg font-bold ${simState.score >= 0 ? 'text-teal' : 'text-red-400'}`}>
                                        {simState.score} pts
                                    </span>
                                </div>
                                <div className="w-48 h-1.5 bg-gray-700/50 rounded-full mt-3 overflow-hidden">
                                    <motion.div
                                        className="h-full bg-teal shadow-[0_0_10px_#00ADB5]"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min(100, (simState.score % 50) * 2)}%` }}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-6 text-sm text-gray-400 mt-2">
                            <button
                                onClick={onReset}
                                className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-xs text-gray-400 hover:text-white transition-colors border border-white/5"
                            >
                                <RefreshCcw size={12} />
                                Reset Simulation
                            </button>
                        </div>
                    </div>

                    {/* Right Side: P&L + Graph */}
                    <div className="flex flex-col items-end w-full md:w-1/2">
                        <div className="text-4xl md:text-5xl font-mono font-bold text-white tracking-tighter shadow-black drop-shadow-lg">
                            ₹{simState.balance.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </div>
                        <div className={`flex items-center gap-2 mt-2 px-3 py-1 rounded-lg bg-black/20 ${isProfit ? 'text-teal' : 'text-red-400'}`}>
                            {isProfit ? <TrendingUp size={16} /> : <AlertTriangle size={16} />}
                            <span className="font-mono font-bold">
                                {isProfit ? '+' : ''}₹{pnl.toFixed(0)} ({((pnl / 10000) * 100).toFixed(2)}%)
                            </span>
                        </div>
                        {/* Advanced Stats */}
                        <div className="flex gap-4 mt-2 text-[10px] font-mono text-gray-500 uppercase tracking-widest">
                            <div title="Sharpe Ratio">
                                SR: <span className="text-white">{simState.sharpe_ratio ? simState.sharpe_ratio.toFixed(2) : '0.00'}</span>
                            </div>
                            <div title="Max Drawdown">
                                DD: <span className="text-red-400">{simState.max_drawdown ? simState.max_drawdown.toFixed(1) : '0.0'}%</span>
                            </div>
                        </div>

                        {/* Mini Graph */}
                        <div className="w-full h-32 mt-4 ml-8">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#00ADB5" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#00ADB5" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <Tooltip content={<></>} />
                                    <Area type="monotone" dataKey="val" stroke="#00ADB5" strokeWidth={2} fillOpacity={1} fill="url(#colorVal)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};
