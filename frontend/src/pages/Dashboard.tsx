import { useState } from 'react';
import { ThinkingNode } from '../components/ThinkingNode';
import { Hero } from '../components/Hero';
import { SimulationPanel } from '../components/SimulationPanel';
import { Terminal } from '../components/Terminal';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Radio, Radar } from 'lucide-react';

interface DashboardProps {
    data: any[];
    loading: boolean;
    marketMood: string;
    lastUpdated?: string;
    simState: any;
    onResetSim: () => void;
    onScan: () => void;
    isAuto: boolean;
    isLocked: boolean;
    onDetails: (ticker: string, action: string, steps: string[], confidence: number, history: any[], quant?: any) => void;
    logs: string[];
}

// Skeleton card for loading state
const SkeletonCard = ({ delay = 0 }: { delay?: number }) => (
    <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay, duration: 0.3 }}
        className="panel p-6 space-y-4 animate-pulse"
    >
        {/* Header */}
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-graphite/50" />
                <div className="space-y-1.5">
                    <div className="w-20 h-4 bg-graphite/50 rounded" />
                    <div className="w-14 h-3 bg-graphite/30 rounded" />
                </div>
            </div>
            <div className="w-12 h-12 rounded-full border-2 border-graphite/30" />
        </div>
        {/* Action badge */}
        <div className="w-28 h-6 bg-graphite/40 rounded-lg" />
        {/* Reasoning lines */}
        <div className="space-y-2 pt-2 border-t border-graphite/20">
            <div className="w-full h-3 bg-graphite/30 rounded" />
            <div className="w-4/5 h-3 bg-graphite/30 rounded" />
            <div className="w-3/5 h-3 bg-graphite/20 rounded" />
        </div>
        {/* Footer */}
        <div className="flex justify-between items-center pt-2">
            <div className="w-16 h-5 bg-graphite/20 rounded" />
            <div className="w-20 h-7 bg-graphite/30 rounded-lg" />
        </div>
    </motion.div>
);

export const Dashboard = ({ data, loading, marketMood, lastUpdated, simState, onResetSim, onScan, isAuto, isLocked, onDetails, logs }: DashboardProps) => {
    const [filter, setFilter] = useState<'all' | 'opportunities' | 'watch'>('all');
    const [searchQuery, setSearchQuery] = useState('');

    const filteredData = data.filter(item => {
        const isOpportunityAction = (item.Action !== 'WAIT' && item.Action !== 'WATCH_FOR_BREAKOUT');
        const matchesTab =
            filter === 'all' ? true :
                filter === 'opportunities' ? isOpportunityAction :
                    !isOpportunityAction;
        const matchesSearch = item.Ticker.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesTab && matchesSearch;
    });

    const categories = [
        { id: 'all', label: 'All Signals' },
        { id: 'opportunities', label: 'Active' },
        { id: 'watch', label: 'Watching' }
    ];

    const isInitialLoad = loading && data.length === 0;

    return (
        <div className="w-full flex flex-col items-center">
            <Hero
                onScan={onScan}
                loading={loading}
                isAuto={isAuto}
                isLocked={isLocked}
                marketMood={marketMood}
                lastUpdated={lastUpdated}
            />

            <SimulationPanel simState={simState} onReset={onResetSim} />
            <Terminal logs={logs} />

            {/* Controls Section */}
            {!isInitialLoad && data.length > 0 && (
                <div className="w-full max-w-[1400px] px-4 mb-8 flex flex-col md:flex-row justify-between items-center gap-4">

                    {/* Filter Tabs */}
                    <div className="flex p-1 rounded-xl panel">
                        {categories.map((cat) => (
                            <button
                                key={cat.id}
                                onClick={() => setFilter(cat.id as any)}
                                className={`relative px-5 py-2 text-sm font-medium rounded-lg cursor-pointer transition-all duration-200 ${filter === cat.id ? 'text-chalk' : 'text-smoke hover:text-ash'
                                    }`}
                            >
                                {filter === cat.id && (
                                    <motion.div
                                        layoutId="filter-pill"
                                        className="absolute inset-0 bg-slate rounded-lg border border-graphite"
                                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                    />
                                )}
                                <span className="relative z-10">{cat.label}</span>
                            </button>
                        ))}
                    </div>

                    {/* Search Bar */}
                    <div className="relative w-full md:w-64">
                        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-smoke" size={16} />
                        <input
                            type="text"
                            placeholder="Search ticker..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full panel py-2.5 pl-10 pr-4 text-sm text-chalk placeholder-smoke font-body focus:outline-none focus:ring-1 focus:ring-amber/30 transition-all"
                        />
                    </div>
                </div>
            )}

            {/* Skeleton Loading State */}
            {isInitialLoad ? (
                <div className="w-full max-w-[1400px] px-4">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-3 mb-6 text-smoke"
                    >
                        <Radar size={16} className="text-amber animate-spin" style={{ animationDuration: '3s' }} />
                        <span className="text-sm font-body">Scanning markets and analyzing signals...</span>
                    </motion.div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                        {[0, 1, 2, 3, 4, 5].map((i) => (
                            <SkeletonCard key={i} delay={i * 0.08} />
                        ))}
                    </div>
                </div>
            ) : (
                <motion.div
                    layout
                    className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 py-4 px-4 max-w-[1400px]"
                >
                    <AnimatePresence mode='popLayout'>
                        {filteredData.map((item) => (
                            <ThinkingNode
                                key={item.Ticker}
                                ticker={item.Ticker}
                                action={item.Action}
                                confidence={item.Confidence}
                                regime={item.Action.includes("CONDOR") || item.Action.includes("SPREAD") ? "Income Mode" : "Sniper Mode"}
                                steps={item.Rational}
                                history={item.History}
                                quant={item.QuantRisk}
                                onDetails={onDetails}
                            />
                        ))}
                    </AnimatePresence>
                </motion.div>
            )}

            {/* Empty State — No data at all (not scanning) */}
            {!loading && data.length === 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="py-20 flex flex-col items-center text-center gap-4"
                >
                    <div className="w-16 h-16 rounded-2xl bg-slate/50 flex items-center justify-center border border-graphite">
                        <Radio size={24} className="text-amber/40" />
                    </div>
                    <div>
                        <h3 className="text-lg font-display font-bold text-chalk mb-1">No Signals Yet</h3>
                        <p className="text-smoke text-sm max-w-md">
                            The engine hasn't generated any signals yet. Hit "Initialize Scan" above to begin analysis, or wait for the auto-scan to kick in.
                        </p>
                    </div>
                </motion.div>
            )}

            {/* Filtered Empty State */}
            {!loading && filteredData.length === 0 && data.length > 0 && (
                <div className="py-20 text-smoke text-lg font-body">
                    No signals match your filter.
                </div>
            )}
        </div>
    );
};
