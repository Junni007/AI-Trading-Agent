import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUpDown, Filter, Eye, Zap, Shield, Search, ArrowUpRight, Radio } from 'lucide-react';
import { Skeleton, CardSkeleton, EmptyState } from '../components/Skeleton';

interface SignalItem {
    Ticker: string;
    Action: string;
    Confidence: number;
    Rational: string[];
    History?: any[];
    QuantRisk?: any;
}

type SortKey = 'Ticker' | 'Action' | 'Confidence';
type SortDir = 'asc' | 'desc';
type FilterTab = 'all' | 'buy' | 'watch';

interface SignalsProps {
    data: SignalItem[];
    loading?: boolean;
    onDetails?: (ticker: string, action: string, steps: string[], confidence: number, history: any[], quant?: any) => void;
}

export const Signals = ({ data, loading = false, onDetails }: SignalsProps) => {
    const [sortKey, setSortKey] = useState<SortKey>('Confidence');
    const [sortDir, setSortDir] = useState<SortDir>('desc');
    const [filterTab, setFilterTab] = useState<FilterTab>('all');
    const [searchQuery, setSearchQuery] = useState('');

    // Only show skeletons on very first load — not during silent background scans
    const isInitialLoad = loading && data.length === 0;

    const handleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
        } else {
            setSortKey(key);
            setSortDir('desc');
        }
    };

    const isActionable = (action: string) => action !== 'WAIT' && action !== 'WATCH_FOR_BREAKOUT';

    const filteredAndSorted = useMemo(() => {
        let result = [...data];

        // Search filter
        if (searchQuery) {
            result = result.filter(item =>
                item.Ticker.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        // Tab filter
        if (filterTab === 'buy') {
            result = result.filter(item => isActionable(item.Action));
        } else if (filterTab === 'watch') {
            result = result.filter(item => !isActionable(item.Action));
        }

        // Sort
        result.sort((a, b) => {
            let cmp = 0;
            if (sortKey === 'Ticker') cmp = a.Ticker.localeCompare(b.Ticker);
            else if (sortKey === 'Action') cmp = a.Action.localeCompare(b.Action);
            else if (sortKey === 'Confidence') cmp = a.Confidence - b.Confidence;
            return sortDir === 'desc' ? -cmp : cmp;
        });

        return result;
    }, [data, sortKey, sortDir, filterTab, searchQuery]);

    const filterTabs = [
        { id: 'all' as FilterTab, label: 'All Signals', count: data.length },
        { id: 'buy' as FilterTab, label: 'Active', count: data.filter(d => isActionable(d.Action)).length },
        { id: 'watch' as FilterTab, label: 'Watching', count: data.filter(d => !isActionable(d.Action)).length },
    ];

    const getActionIcon = (action: string) => {
        if (action === 'WAIT' || action === 'WATCH_FOR_BREAKOUT') return <Eye size={12} />;
        if (action.includes('CONDOR') || action.includes('SPREAD')) return <Shield size={12} />;
        return <Zap size={12} />;
    };

    const getActionStyle = (action: string) => {
        if (action === 'WAIT' || action === 'WATCH_FOR_BREAKOUT') {
            return 'bg-iron/30 text-smoke border border-iron/50';
        }
        if (action.includes('CONDOR') || action.includes('SPREAD')) {
            return 'bg-sage/10 text-sage border border-sage/20';
        }
        return 'bg-amber/10 text-amber border border-amber/20';
    };

    return (
        <div className="w-full max-w-7xl mx-auto py-20 px-6">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <h1 className="font-display text-4xl font-bold text-chalk mb-2 tracking-tight">Active Signals</h1>
                <p className="text-smoke text-lg mb-6">Real-time trading signals from the analysis engine</p>
            </motion.div>

            {/* Controls Bar */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6"
            >
                {/* Filter Tabs */}
                <div className="flex items-center gap-2 p-1 rounded-xl panel">
                    <Filter size={14} className="text-smoke ml-2 shrink-0" />
                    {filterTabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setFilterTab(tab.id)}
                            className={`relative px-4 py-2 text-sm font-medium rounded-lg cursor-pointer transition-all duration-200 ${filterTab === tab.id ? 'text-chalk' : 'text-smoke hover:text-ash'
                                }`}
                        >
                            {filterTab === tab.id && (
                                <motion.div
                                    layoutId="signal-filter-pill"
                                    className="absolute inset-0 bg-slate rounded-lg border border-graphite"
                                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                />
                            )}
                            <span className="relative z-10 flex items-center gap-1.5">
                                {tab.label}
                                <span className="text-[10px] font-mono text-smoke">{tab.count}</span>
                            </span>
                        </button>
                    ))}
                </div>

                {/* Search */}
                <div className="relative w-full sm:w-56">
                    <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-smoke" size={14} />
                    <input
                        type="text"
                        placeholder="Search ticker..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full panel py-2 pl-9 pr-4 text-sm text-chalk placeholder-smoke font-body focus:outline-none focus:ring-1 focus:ring-amber/30 transition-all"
                    />
                </div>
            </motion.div>

            {/* Desktop Table View */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="hidden md:block overflow-x-auto panel"
            >
                <table className="w-full text-left">
                    <thead className="bg-obsidian text-smoke uppercase text-xs tracking-wider border-b border-graphite">
                        <tr>
                            <SortHeader label="Ticker" sortKey="Ticker" currentSort={sortKey} sortDir={sortDir} onSort={handleSort} />
                            <SortHeader label="Action" sortKey="Action" currentSort={sortKey} sortDir={sortDir} onSort={handleSort} />
                            <SortHeader label="Confidence" sortKey="Confidence" currentSort={sortKey} sortDir={sortDir} onSort={handleSort} />
                            <th className="px-6 py-4 font-semibold">Reasoning</th>
                            <th className="px-4 py-4 font-semibold w-10"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite/50 text-sm font-body">
                        {isInitialLoad ? (
                            Array.from({ length: 5 }).map((_, i) => (
                                <tr key={i} className="animate-pulse">
                                    <td className="px-6 py-4"><Skeleton className="w-24 h-5" /></td>
                                    <td className="px-6 py-4"><Skeleton className="w-20 h-6 rounded-lg" /></td>
                                    <td className="px-6 py-4"><div className="flex items-center gap-3"><Skeleton className="w-20 h-2 bg-graphite/50" /><Skeleton className="w-8 h-4" /></div></td>
                                    <td className="px-6 py-4"><Skeleton className="w-full h-4" /></td>
                                    <td className="px-4 py-4"><Skeleton className="w-6 h-6 rounded-lg" /></td>
                                </tr>
                            ))
                        ) : filteredAndSorted.length === 0 ? (
                            <tr>
                                <td colSpan={5} className="px-6 py-10">
                                    <EmptyState
                                        icon={Radio}
                                        title={data.length === 0 ? 'No Active Signals' : 'No Matches'}
                                        description={data.length === 0 ? "The engine hasn't generated any signals yet." : "No signals match your current filters."}
                                    />
                                </td>
                            </tr>
                        ) : (
                            <AnimatePresence mode="popLayout">
                                {filteredAndSorted.map((item) => (
                                    <motion.tr
                                        key={item.Ticker}
                                        layout
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        className="hover:bg-slate/30 transition-colors"
                                    >
                                        <td className="px-6 py-4 font-display font-bold text-chalk">{item.Ticker.replace('.NS', '')}</td>
                                        <td className="px-6 py-4">
                                            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold ${getActionStyle(item.Action)}`}>
                                                {getActionIcon(item.Action)}
                                                {item.Action.replace(/_/g, ' ')}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-20 h-1.5 bg-graphite/50 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full rounded-full transition-all duration-500 ${item.Confidence >= 0.8 ? 'bg-sage' : item.Confidence >= 0.6 ? 'bg-amber' : 'bg-iron'
                                                            }`}
                                                        style={{ width: `${item.Confidence * 100}%` }}
                                                    />
                                                </div>
                                                <span className="font-mono text-amber text-sm w-10">
                                                    {(item.Confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-ash truncate max-w-md">
                                            {item.Rational[item.Rational.length - 1]}
                                        </td>
                                        <td className="px-4 py-4">
                                            <button
                                                onClick={() => onDetails?.(item.Ticker, item.Action, item.Rational, item.Confidence, item.History || [], item.QuantRisk)}
                                                className="p-1.5 rounded-lg hover:bg-slate/50 text-smoke hover:text-amber transition-all cursor-pointer group"
                                                title={`View details for ${item.Ticker}`}
                                                aria-label={`View details for ${item.Ticker}`}
                                            >
                                                <ArrowUpRight size={14} className="group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                                            </button>
                                        </td>
                                    </motion.tr>
                                ))}
                            </AnimatePresence>
                        )}
                    </tbody>
                </table>
            </motion.div>

            {/* Mobile Card View */}
            <div className="md:hidden space-y-3">
                <AnimatePresence mode="popLayout">
                    {isInitialLoad ? (
                        Array.from({ length: 3 }).map((_, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: i * 0.1 }}
                            >
                                <CardSkeleton />
                            </motion.div>
                        ))
                    ) : filteredAndSorted.length === 0 ? (
                        <div className="py-8">
                            <EmptyState
                                icon={Radio}
                                title={data.length === 0 ? 'No Active Signals' : 'No Matches'}
                                description={data.length === 0 ? "The engine hasn't generated any signals yet." : "No signals match your current filters."}
                            />
                        </div>
                    ) : (
                        filteredAndSorted.map((item, i) => (
                            <motion.div
                                key={item.Ticker}
                                layout
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ delay: i * 0.03 }}
                                className="panel p-4"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div>
                                        <div className="font-display font-bold text-lg text-chalk">{item.Ticker.replace('.NS', '')}</div>
                                        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-semibold mt-1 ${getActionStyle(item.Action)}`}>
                                            {getActionIcon(item.Action)}
                                            {item.Action.replace(/_/g, ' ')}
                                        </span>
                                    </div>
                                    <div className="text-right">
                                        <div className="font-mono text-amber font-bold">{(item.Confidence * 100).toFixed(0)}%</div>
                                        <div className="w-16 h-1.5 bg-graphite/50 rounded-full overflow-hidden mt-1">
                                            <div
                                                className={`h-full rounded-full ${item.Confidence >= 0.8 ? 'bg-sage' : item.Confidence >= 0.6 ? 'bg-amber' : 'bg-iron'}`}
                                                style={{ width: `${item.Confidence * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between mt-3 pt-2 border-t border-graphite/30">
                                    <p className="text-xs text-ash line-clamp-1 flex-1 mr-3">{item.Rational[item.Rational.length - 1]}</p>
                                    <button
                                        onClick={() => onDetails?.(item.Ticker, item.Action, item.Rational, item.Confidence, item.History || [], item.QuantRisk)}
                                        className="flex items-center gap-1 text-[11px] text-amber hover:text-chalk transition-colors cursor-pointer group shrink-0"
                                        title={`View details for ${item.Ticker}`}
                                    >
                                        Details
                                        <ArrowUpRight size={12} className="group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                                    </button>
                                </div>
                            </motion.div>
                        ))
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

// Sortable column header component
const SortHeader = ({
    label,
    sortKey,
    currentSort,
    sortDir,
    onSort,
}: {
    label: string;
    sortKey: SortKey;
    currentSort: SortKey;
    sortDir: SortDir;
    onSort: (key: SortKey) => void;
}) => {
    const isActive = currentSort === sortKey;
    return (
        <th
            className="px-6 py-4 font-semibold cursor-pointer select-none hover:text-chalk transition-colors"
            onClick={() => onSort(sortKey)}
        >
            <span className="flex items-center gap-1.5">
                {label}
                <ArrowUpDown
                    size={12}
                    className={`transition-colors ${isActive ? 'text-amber' : 'text-graphite'}`}
                />
                {isActive && (
                    <span className="text-[9px] text-amber font-mono">{sortDir === 'asc' ? '↑' : '↓'}</span>
                )}
            </span>
        </th>
    );
};
