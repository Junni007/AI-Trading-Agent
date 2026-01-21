import React from 'react';
import { motion } from 'framer-motion';
import {
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, Activity, AlertTriangle, Award } from 'lucide-react';

interface AnalyticsProps {
    simState: any;
}

export const Analytics: React.FC<AnalyticsProps> = ({ simState }) => {
    if (!simState) {
        return (
            <div className="w-full py-20 text-center text-smoke">
                Loading analytics...
            </div>
        );
    }

    const equityCurve = simState.equity_curve || [];
    const balance = simState.balance || 10000;
    const pnl = balance - 10000;
    const pnlPercent = ((pnl / 10000) * 100);
    const sharpe = simState.sharpe_ratio || 0;
    const maxDrawdown = simState.max_drawdown || 0;
    const score = simState.score || 0;
    const history = simState.history || [];

    // Prepare chart data
    const equityData = equityCurve.map((val: number, i: number) => ({
        day: i + 1,
        equity: val,
    }));

    // Calculate daily returns for histogram
    const dailyReturns: number[] = [];
    for (let i = 1; i < equityCurve.length; i++) {
        const ret = ((equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1]) * 100;
        dailyReturns.push(ret);
    }

    // Group returns into buckets for histogram
    const returnBuckets: { range: string; count: number; color: string }[] = [
        { range: '< -2%', count: 0, color: '#EF4444' },
        { range: '-2 to -1%', count: 0, color: '#F97316' },
        { range: '-1 to 0%', count: 0, color: '#FBBF24' },
        { range: '0 to 1%', count: 0, color: '#84CC16' },
        { range: '1 to 2%', count: 0, color: '#22C55E' },
        { range: '> 2%', count: 0, color: '#10B981' },
    ];

    dailyReturns.forEach(ret => {
        if (ret < -2) returnBuckets[0].count++;
        else if (ret < -1) returnBuckets[1].count++;
        else if (ret < 0) returnBuckets[2].count++;
        else if (ret < 1) returnBuckets[3].count++;
        else if (ret < 2) returnBuckets[4].count++;
        else returnBuckets[5].count++;
    });

    // Win/Loss pie data
    const wins = history.filter((h: string) => h.includes('SELL_TP')).length;
    const losses = history.filter((h: string) => h.includes('SELL_SL')).length;
    const pieData = [
        { name: 'Wins', value: wins, color: '#10B981' },
        { name: 'Losses', value: losses, color: '#EF4444' },
    ];
    const winRate = wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0;

    // Metric Card component
    const MetricCard = ({ icon: Icon, label, value, subvalue, color }: any) => (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel p-5"
        >
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-xs text-smoke uppercase tracking-wider mb-1">{label}</p>
                    <p className={`font-mono text-2xl font-bold ${color}`}>{value}</p>
                    {subvalue && <p className="text-xs text-smoke mt-1">{subvalue}</p>}
                </div>
                <div className={`p-2 rounded-lg bg-${color === 'text-sage' ? 'sage' : color === 'text-crimson' ? 'crimson' : 'amber'}/10`}>
                    <Icon size={18} className={color} />
                </div>
            </div>
        </motion.div>
    );

    return (
        <div className="w-full max-w-[1400px] px-4 py-8">
            <motion.h1
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="font-display text-3xl font-bold text-chalk mb-8"
            >
                Analytics
            </motion.h1>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <MetricCard
                    icon={pnl >= 0 ? TrendingUp : TrendingDown}
                    label="Total P&L"
                    value={`${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(0)}`}
                    subvalue={`${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`}
                    color={pnl >= 0 ? 'text-sage' : 'text-crimson'}
                />
                <MetricCard
                    icon={Activity}
                    label="Sharpe Ratio"
                    value={sharpe.toFixed(2)}
                    subvalue="Risk-adjusted return"
                    color={sharpe > 0 ? 'text-sage' : 'text-crimson'}
                />
                <MetricCard
                    icon={AlertTriangle}
                    label="Max Drawdown"
                    value={`${maxDrawdown.toFixed(1)}%`}
                    subvalue="Largest peak-to-trough"
                    color="text-amber"
                />
                <MetricCard
                    icon={Award}
                    label="Win Rate"
                    value={`${winRate.toFixed(0)}%`}
                    subvalue={`${wins}W / ${losses}L`}
                    color={winRate >= 50 ? 'text-sage' : 'text-crimson'}
                />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                {/* Equity Curve */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="panel p-6"
                >
                    <h3 className="font-display text-lg font-bold text-chalk mb-4">Equity Curve</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={equityData}>
                                <defs>
                                    <linearGradient id="equityGradientAnalytics" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A32" />
                                <XAxis dataKey="day" stroke="#71717A" fontSize={10} />
                                <YAxis stroke="#71717A" fontSize={10} tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`} />
                                <Tooltip
                                    contentStyle={{ background: '#141418', border: '1px solid #2A2A32', borderRadius: '8px' }}
                                    labelStyle={{ color: '#F5F5F5' }}
                                    formatter={(value: number) => [`₹${value.toFixed(0)}`, 'Equity']}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="equity"
                                    stroke="#F59E0B"
                                    strokeWidth={2}
                                    fill="url(#equityGradientAnalytics)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Return Distribution */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="panel p-6"
                >
                    <h3 className="font-display text-lg font-bold text-chalk mb-4">Return Distribution</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={returnBuckets}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2A2A32" />
                                <XAxis dataKey="range" stroke="#71717A" fontSize={9} />
                                <YAxis stroke="#71717A" fontSize={10} />
                                <Tooltip
                                    contentStyle={{ background: '#141418', border: '1px solid #2A2A32', borderRadius: '8px' }}
                                    labelStyle={{ color: '#F5F5F5' }}
                                />
                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                    {returnBuckets.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            </div>

            {/* Win/Loss Pie + Score */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="panel p-6"
                >
                    <h3 className="font-display text-lg font-bold text-chalk mb-4">Win/Loss Ratio</h3>
                    <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={40}
                                    outerRadius={70}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ background: '#141418', border: '1px solid #2A2A32', borderRadius: '8px' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="flex justify-center gap-6 mt-2">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-sage" />
                            <span className="text-xs text-smoke">Wins ({wins})</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-crimson" />
                            <span className="text-xs text-smoke">Losses ({losses})</span>
                        </div>
                    </div>
                </motion.div>

                {/* Score Progress */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="panel p-6 md:col-span-2"
                >
                    <h3 className="font-display text-lg font-bold text-chalk mb-4">Level Progress</h3>
                    <div className="space-y-4">
                        <div className="flex justify-between text-sm">
                            <span className="text-smoke">Current Level</span>
                            <span className="text-amber font-semibold">{simState.level}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-smoke">Score</span>
                            <span className="text-chalk font-mono">{score} pts</span>
                        </div>
                        <div className="space-y-2">
                            {[
                                { level: 'Novice', threshold: 0 },
                                { level: 'Apprentice', threshold: 50 },
                                { level: 'Pro', threshold: 100 },
                                { level: 'Grandmaster', threshold: 200 },
                                { level: 'Wolf', threshold: 500 },
                            ].map((tier, idx) => (
                                <div key={idx} className="flex items-center gap-3">
                                    <div className={`w-3 h-3 rounded-full ${score >= tier.threshold ? 'bg-amber' : 'bg-graphite'}`} />
                                    <span className={`text-xs ${score >= tier.threshold ? 'text-chalk' : 'text-smoke'}`}>
                                        {tier.level} ({tier.threshold}+)
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
};
