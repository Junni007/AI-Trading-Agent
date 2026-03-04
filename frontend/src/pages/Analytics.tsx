import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
import { TrendingUp, Activity, Info, ArrowUpRight, ArrowDownRight, Briefcase, Clock, ShieldAlert, Target } from 'lucide-react';

interface AnalyticsProps {
    simState?: any;
    marketData?: any[];
}

export const Analytics = ({ simState, marketData = [] }: AnalyticsProps) => {
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
    const level = simState?.level || "Novice";
    const status = simState?.status || 'IDLE';

    // Prepare equity curve data
    const equityCurve = simState?.equity_curve?.map((val: number, i: number) => ({
        name: `T+${i}`,
        value: val
    })) || [];

    // Parse Live Holdings
    const holdings = simState?.positions ? Object.entries(simState.positions).map(([ticker, pos]: [string, any]) => {
        const liveData = marketData.find(m => m.Ticker === ticker);
        const currentPrice = liveData?.Price || pos.avg_price;
        const value = pos.qty * currentPrice;
        const costBasis = pos.qty * pos.avg_price;
        const pnl = value - costBasis;
        const pnlPct = costBasis ? (pnl / costBasis) * 100 : 0;
        return { ticker, qty: pos.qty, avg_price: pos.avg_price, currentPrice, value, pnl, pnlPct };
    }) : [];

    const containerVariants = {
        hidden: { opacity: 0 },
        show: {
            opacity: 1,
            transition: { staggerChildren: 0.05 }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } }
    };

    return (
        <motion.div
            className="w-full max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8"
            variants={containerVariants}
            initial="hidden"
            animate="show"
        >
            {/* Header section */}
            <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-10">
                <div>
                    <h1 className="font-display text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-chalk to-ash mb-2 tracking-tight">
                        Portfolio Analytics
                    </h1>
                    <p className="text-smoke text-sm md:text-base flex items-center gap-2">
                        <span className="relative flex h-3 w-3">
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${status === 'ACTIVE' ? 'bg-sage' : 'bg-crimson'}`}></span>
                            <span className={`relative inline-flex rounded-full h-3 w-3 ${status === 'ACTIVE' ? 'bg-emerald-400' : 'bg-red-500'}`}></span>
                        </span>
                        Real-time AI Trading Engine • {level} Mode
                    </p>
                </div>

                <div className="flex bg-slate/40 backdrop-blur-xl border border-white/5 rounded-2xl p-4 shadow-2xl">
                    <div className="pr-6 border-r border-white/5">
                        <div className="text-xs text-smoke font-mono mb-1 uppercase tracking-wider">Total Balance</div>
                        <div className="text-3xl font-mono font-bold text-chalk tracking-tight">
                            ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                    </div>
                    <div className="pl-6 flex flex-col justify-center">
                        <div className="text-xs text-smoke font-mono mb-1 uppercase tracking-wider">Net Return</div>
                        <div className={`flex items-center gap-1.5 text-xl font-bold font-mono ${totalReturn >= 0 ? 'text-sage' : 'text-crimson'}`}>
                            {totalReturn >= 0 ? <ArrowUpRight size={20} /> : <ArrowDownRight size={20} />}
                            {Math.abs(totalReturn).toFixed(2)}%
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Main Visuals Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

                {/* Equity Curve (Spans 2 columns on large screens) */}
                <motion.div variants={itemVariants} className="lg:col-span-2 bg-slate/30 backdrop-blur-xl border border-white/5 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-96 h-96 bg-amber/5 rounded-full blur-[100px] -mr-48 -mt-48 pointer-events-none" />

                    <div className="flex items-center justify-between mb-6 relative z-10">
                        <h2 className="text-lg font-bold text-chalk font-display flex items-center gap-2">
                            <Activity size={18} className="text-amber" />
                            Equity Curve
                        </h2>
                        <div className="px-3 py-1 bg-white/5 rounded-full text-xs font-mono text-smoke border border-white/10">
                            Session Lifetime
                        </div>
                    </div>

                    <div className="w-full h-72 relative z-10">
                        {equityCurve.length > 1 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={equityCurve} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#F59E0B" stopOpacity={0.4} />
                                            <stop offset="100%" stopColor="#F59E0B" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2A2A32" vertical={false} />
                                    <XAxis dataKey="name" stroke="#52525B" fontSize={10} tickLine={false} axisLine={false} dy={10} />
                                    <YAxis stroke="#52525B" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v.toLocaleString()}`} dx={-10} domain={['auto', 'auto']} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#141418', borderColor: '#2A2A32', borderRadius: '12px', color: '#F5F5F5', fontFamily: 'JetBrains Mono', fontSize: '13px', boxShadow: '0 20px 25px -5px rgb(0 0 0 / 0.5)' }}
                                        itemStyle={{ color: '#F59E0B', fontWeight: 'bold' }}
                                        formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, 'Portfolio Value']}
                                    />
                                    <Area type="monotone" dataKey="value" stroke="#F59E0B" strokeWidth={3} fill="url(#equityGrad)" activeDot={{ r: 6, fill: '#F59E0B', stroke: '#1D1D24', strokeWidth: 3 }} />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="w-full h-full flex flex-col items-center justify-center text-smoke border border-dashed border-white/10 rounded-xl">
                                <Activity size={24} className="mb-2 opacity-50" />
                                <span>Awaiting trade completion to plot curve...</span>
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* KPI Cards Stack */}
                <motion.div variants={itemVariants} className="flex flex-col gap-4">
                    <KPICard
                        title="Win Rate"
                        value={`${metrics.winRate.toFixed(1)}%`}
                        sub={`${metrics.totalTrades} Total Trades`}
                        icon={<Target size={20} className={metrics.winRate >= 50 ? 'text-sage' : 'text-crimson'} />}
                        trend={metrics.winRate >= 50 ? 'positive' : 'negative'}
                    />
                    <KPICard
                        title="Avg Return / Trade"
                        value={`${metrics.avgReturn > 0 ? '+' : ''}${metrics.avgReturn.toFixed(2)}%`}
                        sub="Realized PnL Profile"
                        icon={<TrendingUp size={20} className={metrics.avgReturn >= 0 ? 'text-amber' : 'text-crimson'} />}
                        trend={metrics.avgReturn >= 0 ? 'neutral' : 'negative'}
                    />
                    <KPICard
                        title="Max Drawdown"
                        value={`-${metrics.maxDrawdown.toFixed(2)}%`}
                        sub="Peak-to-Trough Decline"
                        icon={<ShieldAlert size={20} className={metrics.maxDrawdown <= 10 ? 'text-sage' : 'text-amber'} />}
                        trend={metrics.maxDrawdown <= 15 ? 'positive' : 'negative'}
                    />
                </motion.div>

            </div>

            {/* Tables Row: Holdings + Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">

                {/* Live Holdings */}
                <motion.div variants={itemVariants} className="bg-slate/30 backdrop-blur-xl border border-white/5 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
                    <div className="p-6 border-b border-white/5 flex items-center justify-between bg-black/20">
                        <h2 className="text-lg font-bold text-chalk font-display flex items-center gap-2">
                            <Briefcase size={18} className="text-amber" />
                            Live Holdings
                        </h2>
                        <span className="text-xs bg-amber/10 text-amber px-2 py-1 rounded-md font-mono border border-amber/20">
                            {holdings.length} Positions
                        </span>
                    </div>

                    <div className="p-0 overflow-x-auto flex-grow">
                        {holdings.length > 0 ? (
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="bg-black/40 text-xs font-mono text-ash uppercase tracking-wider">
                                        <th className="p-4 rounded-tl-xl font-medium">Asset</th>
                                        <th className="p-4 font-medium">Qty</th>
                                        <th className="p-4 font-medium">Avg Price</th>
                                        <th className="p-4 text-right rounded-tr-xl font-medium">Float PnL</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {holdings.map((pos: any, idx: number) => (
                                        <tr key={idx} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors group">
                                            <td className="p-4">
                                                <div className="font-bold text-chalk text-sm group-hover:text-amber transition-colors">{pos.ticker}</div>
                                            </td>
                                            <td className="p-4 font-mono text-sm text-smoke">{pos.qty}</td>
                                            <td className="p-4 font-mono text-sm text-smoke">${pos.avg_price.toFixed(2)}</td>
                                            <td className="p-4 text-right">
                                                <div className={`font-mono text-sm font-bold ${pos.pnl >= 0 ? 'text-sage' : 'text-crimson'}`}>
                                                    {pos.pnl >= 0 ? '+' : ''}${pos.pnl.toFixed(2)}
                                                </div>
                                                <div className={`font-mono text-xs ${pos.pnlPct >= 0 ? 'text-sage/70' : 'text-crimson/70'}`}>
                                                    {pos.pnlPct >= 0 ? '+' : ''}{pos.pnlPct.toFixed(2)}%
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <div className="p-12 text-center flex flex-col items-center justify-center text-smoke/70">
                                <Briefcase size={32} className="mb-3 opacity-20" />
                                <p className="text-sm">No active positions.</p>
                                <p className="text-xs mt-1">Agent is scanning the market for entry signals.</p>
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Recent Activity Timeline */}
                <motion.div variants={itemVariants} className="bg-slate/30 backdrop-blur-xl border border-white/5 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
                    <div className="p-6 border-b border-white/5 flex items-center justify-between bg-black/20">
                        <h2 className="text-lg font-bold text-chalk font-display flex items-center gap-2">
                            <Clock size={18} className="text-amber" />
                            Recent Trades
                        </h2>
                    </div>

                    <div className="p-6 overflow-y-auto max-h-[400px]">
                        {simState?.history && simState.history.length > 0 ? (
                            <div className="relative border-l border-white/10 ml-3 space-y-6">
                                {simState.history.slice(0, 8).map((trade: any, i: number) => { // Render newest first (already insert(0))
                                    const isBuy = trade.action === 'BUY';
                                    const isProfit = trade.port_value > (trade.prev_port_value || initialBalance);

                                    // Determine dot color
                                    let dotColor = 'bg-slate border-white/20';
                                    if (isBuy) dotColor = 'bg-[#1D1D24] border-amber shadow-[0_0_10px_rgba(245,158,11,0.5)]';
                                    else if (trade.action === 'SELL_TP' || isProfit) dotColor = 'bg-[#1D1D24] border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]';
                                    else if (trade.action === 'SELL_SL' || !isProfit) dotColor = 'bg-[#1D1D24] border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';

                                    return (
                                        <div key={i} className="pl-6 relative">
                                            <div className={`absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full border-2 ${dotColor}`} />
                                            <div className="flex justify-between items-start mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className={`text-xs font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${isBuy ? 'bg-amber/10 text-amber border border-amber/20'
                                                        : trade.action === 'SELL_TP' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                                            : trade.action === 'SELL_SL' ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                                : 'bg-white/5 text-ash border border-white/10'
                                                        }`}>
                                                        {trade.action}
                                                    </span>
                                                    <span className="font-bold text-chalk">{trade.ticker}</span>
                                                </div>
                                                <span className="text-xs text-ash font-mono">
                                                    {new Date(trade.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                </span>
                                            </div>
                                            <div className="flex justify-between items-end text-sm text-smoke mt-2">
                                                <span className="font-mono">Executed @ ${trade.price?.toFixed(2)}</span>
                                                <div className="text-right">
                                                    <div className="text-xs text-ash mb-0.5">Port Value</div>
                                                    <div className="font-mono font-semibold text-chalk">
                                                        ${trade.port_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : (
                            <div className="py-12 text-center text-smoke/70">
                                <p className="text-sm">No transaction history.</p>
                            </div>
                        )}
                    </div>
                </motion.div>
            </div>

            {/* Info Footer */}
            <motion.div variants={itemVariants} className="p-4 bg-amber/5 border border-amber/10 rounded-xl flex gap-3 text-smoke/80 items-start">
                <Info size={18} className="text-amber shrink-0 mt-0.5" />
                <div className="text-sm leading-relaxed">
                    <strong className="text-chalk block mb-1">Algorithmic Insight Engine V1.0</strong>
                    This dashboard renders live telemetrics from the Python Simulation Engine.
                    The Agent scales trades dynamically (up to 80% available cash) based on RL-driven signal confidence.
                    For deeper backtest validation, execute <code className="bg-black/40 px-1.5 py-0.5 rounded text-amber">python -m src.analysis.runner</code>
                </div>
            </motion.div>
        </motion.div>
    );
};

// --- Helper Components ---

const KPICard = ({ title, value, sub, icon, trend }: { title: string, value: string, sub: string, icon: any, trend: 'positive' | 'negative' | 'neutral' }) => {
    let glow = '';
    if (trend === 'positive') glow = 'group-hover:shadow-[0_0_30px_rgba(52,211,153,0.1)]';
    if (trend === 'negative') glow = 'group-hover:shadow-[0_0_30px_rgba(248,113,113,0.1)]';
    if (trend === 'neutral') glow = 'group-hover:shadow-[0_0_30px_rgba(245,158,11,0.1)]';

    return (
        <div className={`bg-slate/30 backdrop-blur-md border border-white/5 rounded-2xl p-5 flex items-center justify-between transition-all duration-300 group ${glow}`}>
            <div>
                <div className="text-sm text-smoke font-medium mb-1">{title}</div>
                <div className="text-2xl font-bold text-chalk font-mono mb-1 tracking-tight">{value}</div>
                <div className="text-xs text-ash">{sub}</div>
            </div>
            <div className="w-12 h-12 rounded-full bg-black/40 flex items-center justify-center border border-white/5">
                {icon}
            </div>
        </div>
    );
};
