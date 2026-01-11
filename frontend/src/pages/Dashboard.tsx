import { useState } from 'react';
import { ThinkingNode } from '../components/ThinkingNode';
import { Loader } from '../components/Loader';
import { Hero } from '../components/Hero';
import { SimulationPanel } from '../components/SimulationPanel';
import { Terminal } from '../components/Terminal';
import { motion, AnimatePresence } from 'framer-motion';
import { Search } from 'lucide-react';

// Define Props for Dashboard
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
        // 1. Filter by Tab
        const isOpportunityAction = (item.Action !== 'WAIT' && item.Action !== 'WATCH_FOR_BREAKOUT');
        const matchesTab =
            filter === 'all' ? true :
                filter === 'opportunities' ? isOpportunityAction :
                    !isOpportunityAction;

        // 2. Filter by Search
        const matchesSearch = item.Ticker.toLowerCase().includes(searchQuery.toLowerCase());

        return matchesTab && matchesSearch;
    });

    const categories = [
        { id: 'all', label: 'All' },
        { id: 'opportunities', label: 'Active' },
        { id: 'watch', label: 'Watchlist' }
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

            {/* Simulation Module (Iteration 7) */}
            <SimulationPanel simState={simState} onReset={onResetSim} />

            {/* Live Terminal (Iteration 8) */}
            <Terminal logs={logs} />

            {/* Controls Section: Filter Tabs + Search */}
            {!loading && data.length > 0 && (
                <div className="w-full max-w-[1400px] px-4 mb-8 flex flex-col md:flex-row justify-between items-center gap-4">

                    {/* Segmented Control */}
                    <div className="flex p-1 bg-white/5 rounded-xl border border-white/5 backdrop-blur-md">
                        {categories.map((cat) => (
                            <button
                                key={cat.id}
                                onClick={() => setFilter(cat.id as any)}
                                className={`relative px-6 py-2 text-sm font-medium rounded-lg transition-all duration-300 ${filter === cat.id ? 'text-white' : 'text-gray-400 hover:text-gray-200'
                                    }`}
                            >
                                {filter === cat.id && (
                                    <motion.div
                                        layoutId="active-pill"
                                        className="absolute inset-0 bg-white/10 rounded-lg border border-white/10 shadow-sm"
                                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                                    />
                                )}
                                <span className="relative z-10">{cat.label}</span>
                            </button>
                        ))}
                    </div>

                    {/* Search Bar */}
                    <div className="relative w-full md:w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                        <input
                            type="text"
                            placeholder="Search Ticker..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-white/5 border border-white/5 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-teal/50 transition-colors backdrop-blur-sm"
                        />
                    </div>
                </div>
            )}

            {loading && data.length === 0 ? (
                <div className="py-20">
                    <Loader />
                </div>
            ) : (
                <motion.div layout className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 py-4 px-4 max-w-[1400px]">
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
                                onDetails={onDetails}
                            />
                        ))}
                    </AnimatePresence>
                </motion.div>
            )}

            {!loading && filteredData.length === 0 && data.length > 0 && (
                <div className="py-20 text-gray-500 text-lg">No assets found in this category.</div>
            )}
        </div>
    );
};
