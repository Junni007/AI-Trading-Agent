import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, Award, BarChart3, Activity, Info, ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface AnalyticsProps {
    simState?: any;
}

export const Analytics = ({ simState }: AnalyticsProps) => {
    const [metrics, setMetrics] = useState({
        totalTrades: 0,
        winRate: 0,
        avgReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        profitFactor: 0
    });

    useEffect(() => {
        if (simState?.history && simState.history.length > 0) {
            const history = simState.history;
            const trades = history.length;

            const wins = history.filter((h: any) => h.port_value > (h.prev_port_value || simState.initial_balance)).length;
            const winRate = trades > 0 ? (wins / trades) * 100 : 0;

            const returns = history.map((h: any) =>
                ((h.port_value - (h.prev_port_value || simState.initial_balance)) / (h.prev_port_value || simState.initial_balance)) * 100
            );
            const avgReturn = returns.reduce((a: number, b: number) => a + b, 0) / (returns.length || 1);

            let peak = simState.initial_balance;
            let maxDD = 0;
            history.forEach((h: any) => {
                if (h.port_value > peak) peak = h.port_value;
                const dd = ((peak - h.port_value) / peak) * 100;
                if (dd > maxDD) maxDD = dd;
            });

            setMetrics({
                totalTrades: trades,
                winRate,
                avgReturn,
                sharpeRatio: simState.sharpe_ratio || 0,
                maxDrawdown: maxDD,
                profitFactor: winRate > 0 ? (winRate / (100 - winRate)) : 0
            });
        }
    }, [simState]);

    const currentBalance = simState?.balance || simState?.initial_balance || 10000;
    const initialBalance = simState?.initial_balance || 10000;
    const totalReturn = ((currentBalance - initialBalance) / initialBalance) * 100;
    const level = simState?.level || 1;
    const status = simState?.status || 'IDLE';

    // Prepare equity curve data
    const equityCurve = simState?.equity_curve?.map((val: number, i: number) => ({
        i,
        value: val,
        label: `Point ${i + 1}`
    })) || [];

    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: { staggerChildren: 0.08 }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 16 },
        show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }
    };

    return (
        <motion.div
            className="w-full max-w-7xl mx-auto py-20 px-6"
            variants={containerVariants}
            initial="hidden"
            animate="show"
        >
            <motion.div variants={itemVariants}>
                <h1 className="font-display text-4xl font-bold text-chalk mb-2">
                    Performance Analytics
                </h1>
                <p className="text-smoke text-lg mb-8">
                    Real-time trading performance metrics
                </p>
            </motion.div>

            {/* Portfolio Overview — Responsive: 1 col mobile, 3 col desktop */}
            <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 mb-8">
                <div className="panel p-6 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-amber/5 rounded-full blur-[60px] -mr-16 -mt-16 pointer-events-none group-hover:bg-amber/10 transition-colors duration-500" />
                    <div className="flex items-center gap-2 text-smoke text-sm font-semibold mb-3">
                        <Activity size={14} className="text-amber" />
                        Current Balance
                    </div>
                    <div className="text-3xl md:text-4xl font-bold text-chalk font-mono">
                        ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className={`flex items-center gap-1.5 text-sm mt-3 ${totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {totalReturn >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                        <span className="font-mono font-semibold">{Math.abs(totalReturn).toFixed(2)}%</span>
                        <span className="text-smoke font-normal">from start</span>
                    </div>
                </div>

                <div className="panel p-6 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-amber/5 rounded-full blur-[60px] -mr-16 -mt-16 pointer-events-none group-hover:bg-amber/10 transition-colors duration-500" />
                    <div className="flex items-center gap-2 text-smoke text-sm font-semibold mb-3">
                        <Award size={14} className="text-amber" />
                        Agent Level
                    </div>
                    <div className="text-3xl md:text-4xl font-bold text-amber font-display">
                        Level {level}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-smoke mt-3">
                        <div className={`w-2 h-2 rounded-full ${status === 'ACTIVE' ? 'bg-sage animate-pulse' : status === 'DEAD' ? 'bg-crimson' : 'bg-iron'}`} />
                        {status}
                    </div>
                </div>

                <div className="panel p-6 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-amber/5 rounded-full blur-[60px] -mr-16 -mt-16 pointer-events-none group-hover:bg-amber/10 transition-colors duration-500" />
                    <div className="flex items-center gap-2 text-smoke text-sm font-semibold mb-3">
                        <BarChart3 size={14} className="text-amber" />
                        Total Trades
                    </div>
                    <div className="text-3xl md:text-4xl font-bold text-chalk font-mono">
                        {metrics.totalTrades}
                    </div>
                    <div className="text-sm text-smoke mt-3">
                        Since deployment
                    </div>
                </div>
            </motion.div>

            {/* Equity Curve Chart */}
            {equityCurve.length > 1 && (
                <motion.div variants={itemVariants} className="panel p-6 mb-8">
                    <h2 className="text-lg font-bold text-chalk font-display mb-4 flex items-center gap-2">
                        {totalReturn >= 0
                            ? <TrendingUp size={18} className="text-sage" />
                            : <TrendingDown size={18} className="text-crimson" />
                        }
                        Equity Curve
                        <span className={`text-xs font-mono ml-auto ${totalReturn >= 0 ? 'text-sage' : 'text-crimson'}`}>
                            {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
                        </span>
                    </h2>
                    <div className="w-full h-48 md:h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={equityCurve}>
                                <defs>
                                    <linearGradient id="analyticsEquityGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.25} />
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
                                    labelFormatter={() => ''}
                                    formatter={(value: number) => [`$${value.toLocaleString()}`, 'Balance']}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#F59E0B"
                                    strokeWidth={2}
                                    fill="url(#analyticsEquityGrad)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            )}

            {/* Performance Metrics — Responsive: 1 col mobile, 2 col desktop */}
            <motion.div variants={itemVariants}>
                <h2 className="text-2xl font-bold text-chalk mb-4 font-display">Key Metrics</h2>
            </motion.div>
            <motion.div variants={itemVariants} className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6 mb-8">
                <MetricCard
                    title="Win Rate"
                    value={`${metrics.winRate.toFixed(1)}%`}
                    description="Percentage of profitable trades"
                    color={metrics.winRate >= 60 ? 'emerald' : metrics.winRate >= 50 ? 'amber' : 'red'}
                    progress={metrics.winRate}
                />
                <MetricCard
                    title="Average Return"
                    value={`${metrics.avgReturn >= 0 ? '+' : ''}${metrics.avgReturn.toFixed(2)}%`}
                    description="Average return per trade"
                    color={metrics.avgReturn >= 0 ? 'emerald' : 'red'}
                    progress={Math.min(100, Math.abs(metrics.avgReturn) * 10)}
                />
                <MetricCard
                    title="Max Drawdown"
                    value={`${metrics.maxDrawdown.toFixed(2)}%`}
                    description="Largest peak-to-trough decline"
                    color={metrics.maxDrawdown <= 10 ? 'emerald' : metrics.maxDrawdown <= 20 ? 'amber' : 'red'}
                    progress={Math.min(100, metrics.maxDrawdown * 5)}
                />
                <MetricCard
                    title="Profit Factor"
                    value={metrics.profitFactor.toFixed(2)}
                    description="Ratio of gross profit to gross loss"
                    color={metrics.profitFactor >= 1.5 ? 'emerald' : metrics.profitFactor >= 1.0 ? 'amber' : 'red'}
                    progress={Math.min(100, metrics.profitFactor * 33)}
                />
            </motion.div>

            {/* Recent Activity */}
            <motion.div variants={itemVariants}>
                <h2 className="text-2xl font-bold text-chalk mb-4 font-display">Recent Activity</h2>
            </motion.div>
            <motion.div variants={itemVariants} className="panel p-6">
                {simState?.history && simState.history.length > 0 ? (
                    <div className="space-y-3">
                        {simState.history.slice(-5).reverse().map((trade: any, i: number) => {
                            const isProfit = trade.port_value > (trade.prev_port_value || initialBalance);
                            return (
                                <div key={i} className="flex items-center justify-between border-b border-graphite/30 pb-3 last:border-0">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isProfit ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                                            {isProfit
                                                ? <TrendingUp size={14} className="text-emerald-400" />
                                                : <TrendingDown size={14} className="text-red-400" />
                                            }
                                        </div>
                                        <div>
                                            <div className="text-chalk font-semibold">{trade.ticker || 'N/A'}</div>
                                            <div className="text-smoke text-sm">{trade.action || 'HOLD'}</div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className={`font-mono font-bold ${isProfit ? 'text-emerald-400' : 'text-red-400'}`}>
                                            ${trade.port_value?.toFixed(2) || '0.00'}
                                        </div>
                                        <div className="text-smoke text-sm">
                                            {new Date(trade.timestamp || Date.now()).toLocaleDateString()}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="text-smoke text-center py-8 flex flex-col items-center gap-3">
                        <Activity size={24} className="text-amber/30" />
                        <span>No trading activity yet. Start the simulation to see metrics.</span>
                    </div>
                )}
            </motion.div>

            {/* Info Banner */}
            <motion.div variants={itemVariants} className="mt-8 p-4 bg-slate/20 border border-graphite/50 rounded-lg">
                <div className="flex items-center gap-2 text-amber font-semibold mb-2">
                    <Info size={16} />
                    About Analytics
                </div>
                <div className="text-smoke text-sm">
                    This dashboard shows real-time performance metrics from your trading simulation.
                    Metrics update automatically as the agent trades. For deeper statistical analysis,
                    run <code className="bg-slate/30 px-2 py-1 rounded">python -m src.analysis.runner</code>
                    to generate expert performance reports and edge validation tests.
                </div>
            </motion.div>
        </motion.div>
    );
};

interface MetricCardProps {
    title: string;
    value: string;
    description: string;
    color: 'emerald' | 'amber' | 'red';
    progress?: number;
}

const MetricCard = ({ title, value, description, color, progress = 0 }: MetricCardProps) => {
    const colorConfig = {
        emerald: {
            text: 'text-emerald-400',
            border: 'border-emerald-500/30',
            bg: 'bg-emerald-500/5',
            bar: 'bg-emerald-400',
        },
        amber: {
            text: 'text-amber',
            border: 'border-amber/30',
            bg: 'bg-amber/5',
            bar: 'bg-amber',
        },
        red: {
            text: 'text-red-400',
            border: 'border-red-500/30',
            bg: 'bg-red-500/5',
            bar: 'bg-red-400',
        },
    };

    const c = colorConfig[color];

    return (
        <div className={`panel p-6 border-2 ${c.border} ${c.bg}`}>
            <div className="text-smoke text-sm font-semibold mb-2">{title}</div>
            <div className={`text-3xl font-bold ${c.text} font-mono`}>
                {value}
            </div>
            {/* Progress bar visualization */}
            <div className="w-full h-1.5 bg-graphite/50 rounded-full mt-3 overflow-hidden">
                <motion.div
                    className={`h-full rounded-full ${c.bar}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, progress)}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
                />
            </div>
            <div className="text-xs text-ash mt-2">{description}</div>
        </div>
    );
};
