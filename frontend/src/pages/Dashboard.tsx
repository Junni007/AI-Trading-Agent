import { useState } from 'react';
import { ThinkingNode } from '../components/ThinkingNode';
import { Loader } from '../components/Loader';
import { Hero } from '../components/Hero';
import { SimulationPanel } from '../components/SimulationPanel';
import { Terminal } from '../components/Terminal';
import { motion, AnimatePresence } from 'framer-motion';
import { Search } from 'lucide-react';

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
    onDetails: (ticker: string, action: string, steps: string[], confidence: number, history: any[]) => void;
    logs: string[];
}

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
            {!loading && data.length > 0 && (
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

            {/* Loading State */}
            {loading && data.length === 0 ? (
                <div className="py-24">
                    <Loader />
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

            {/* Empty State */}
            {!loading && filteredData.length === 0 && data.length > 0 && (
                <div className="py-20 text-smoke text-lg font-body">
                    No signals match your filter.
                </div>
            )}
        </div>
    );
};
