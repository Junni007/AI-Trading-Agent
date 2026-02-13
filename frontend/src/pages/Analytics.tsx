import { useState, useEffect } from 'react';

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
        // Calculate metrics from simulation state
        if (simState?.history && simState.history.length > 0) {
            const history = simState.history;
            const trades = history.length;

            // Calculate win rate
            const wins = history.filter((h: any) => h.port_value > (h.prev_port_value || simState.initial_balance)).length;
            const winRate = trades > 0 ? (wins / trades) * 100 : 0;

            // Calculate average return
            const returns = history.map((h: any) =>
                ((h.port_value - (h.prev_port_value || simState.initial_balance)) / (h.prev_port_value || simState.initial_balance)) * 100
            );
            const avgReturn = returns.reduce((a: number, b: number) => a + b, 0) / (returns.length || 1);

            // Calculate max drawdown
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

    return (
        <div className="w-full max-w-7xl mx-auto py-20 px-6">
            <h1 className="font-display text-4xl font-bold text-chalk mb-2">
                Performance Analytics
            </h1>
            <p className="text-smoke text-lg mb-8">
                Real-time trading performance metrics
            </p>

            {/* Portfolio Overview */}
            <div className="grid grid-cols-3 gap-6 mb-8">
                <div className="panel p-6">
                    <div className="text-smoke text-sm font-semibold mb-2">Current Balance</div>
                    <div className="text-4xl font-bold text-chalk">
                        ${currentBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className={`text-sm mt-2 ${totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {totalReturn >= 0 ? '↑' : '↓'} {Math.abs(totalReturn).toFixed(2)}% from start
                    </div>
                </div>

                <div className="panel p-6">
                    <div className="text-smoke text-sm font-semibold mb-2">Agent Level</div>
                    <div className="text-4xl font-bold text-amber">
                        Level {level}
                    </div>
                    <div className="text-sm text-smoke mt-2">
                        Status: {status}
                    </div>
                </div>

                <div className="panel p-6">
                    <div className="text-smoke text-sm font-semibold mb-2">Total Trades</div>
                    <div className="text-4xl font-bold text-chalk">
                        {metrics.totalTrades}
                    </div>
                    <div className="text-sm text-smoke mt-2">
                        Since deployment
                    </div>
                </div>
            </div>

            {/* Performance Metrics */}
            <h2 className="text-2xl font-bold text-chalk mb-4">Key Metrics</h2>
            <div className="grid grid-cols-2 gap-6 mb-8">
                <MetricCard
                    title="Win Rate"
                    value={`${metrics.winRate.toFixed(1)}%`}
                    description="Percentage of profitable trades"
                    color={metrics.winRate >= 60 ? 'emerald' : metrics.winRate >= 50 ? 'amber' : 'red'}
                />
                <MetricCard
                    title="Average Return"
                    value={`${metrics.avgReturn >= 0 ? '+' : ''}${metrics.avgReturn.toFixed(2)}%`}
                    description="Average return per trade"
                    color={metrics.avgReturn >= 0 ? 'emerald' : 'red'}
                />
                <MetricCard
                    title="Max Drawdown"
                    value={`${metrics.maxDrawdown.toFixed(2)}%`}
                    description="Largest peak-to-trough decline"
                    color={metrics.maxDrawdown <= 10 ? 'emerald' : metrics.maxDrawdown <= 20 ? 'amber' : 'red'}
                />
                <MetricCard
                    title="Profit Factor"
                    value={metrics.profitFactor.toFixed(2)}
                    description="Ratio of gross profit to gross loss"
                    color={metrics.profitFactor >= 1.5 ? 'emerald' : metrics.profitFactor >= 1.0 ? 'amber' : 'red'}
                />
            </div>

            {/* Recent Activity */}
            <h2 className="text-2xl font-bold text-chalk mb-4">Recent Activity</h2>
            <div className="panel p-6">
                {simState?.history && simState.history.length > 0 ? (
                    <div className="space-y-3">
                        {simState.history.slice(-5).reverse().map((trade: any, i: number) => (
                            <div key={i} className="flex items-center justify-between border-b border-graphite/30 pb-3 last:border-0">
                                <div>
                                    <div className="text-chalk font-semibold">{trade.ticker || 'N/A'}</div>
                                    <div className="text-smoke text-sm">{trade.action || 'HOLD'}</div>
                                </div>
                                <div className="text-right">
                                    <div className={`font-mono font-bold ${trade.port_value > (trade.prev_port_value || initialBalance)
                                            ? 'text-emerald-400'
                                            : 'text-red-400'
                                        }`}>
                                        ${trade.port_value?.toFixed(2) || '0.00'}
                                    </div>
                                    <div className="text-smoke text-sm">
                                        {new Date(trade.timestamp || Date.now()).toLocaleDateString()}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-smoke text-center py-8">
                        No trading activity yet. Start the simulation to see metrics.
                    </div>
                )}
            </div>

            {/* Info Banner */}
            <div className="mt-8 p-4 bg-slate/20 border border-graphite/50 rounded-lg">
                <div className="text-amber font-semibold mb-2">💡 About Analytics</div>
                <div className="text-smoke text-sm">
                    This dashboard shows real-time performance metrics from your trading simulation.
                    Metrics update automatically as the agent trades. For deeper statistical analysis,
                    run <code className="bg-slate/30 px-2 py-1 rounded">python -m src.analysis.runner</code>
                    to generate expert performance reports and edge validation tests.
                </div>
            </div>
        </div>
    );
};

interface MetricCardProps {
    title: string;
    value: string;
    description: string;
    color: 'emerald' | 'amber' | 'red';
}

const MetricCard = ({ title, value, description, color }: MetricCardProps) => {
    const colorClasses = {
        emerald: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/5',
        amber: 'text-amber border-amber/30 bg-amber/5',
        red: 'text-red-400 border-red-500/30 bg-red-500/5'
    };

    return (
        <div className={`panel p-6 border-2 ${colorClasses[color]}`}>
            <div className="text-smoke text-sm font-semibold mb-2">{title}</div>
            <div className={`text-3xl font-bold ${colorClasses[color].split(' ')[0]}`}>
                {value}
            </div>
            <div className="text-xs text-ash mt-2">{description}</div>
        </div>
    );
};
