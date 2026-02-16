import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertSettings } from '../components/AlertSettings';
import {
    AlertTriangle, Save, RotateCcw, CheckCircle2,
    Wifi, WifiOff, Server, SlidersHorizontal, Bell,
    Trash2, Info, Globe, Minus, Plus, Cpu, Zap
} from 'lucide-react';
import api from '../services/api';
import { useToast } from '../components/Toast';

interface SettingsState {
    universe: 'nifty50' | 'nasdaq';
    confidenceThreshold: number;
    maxPositions: number;
}

const DEFAULT_SETTINGS: SettingsState = {
    universe: 'nifty50',
    confidenceThreshold: 70,
    maxPositions: 5,
};

const STORAGE_KEY = 'signal_engine_settings';


// Section Header
const SectionHeader = ({ icon: Icon, title, description }: {
    icon: React.ElementType;
    title: string;
    description: string;
}) => (
    <div className="flex items-start gap-3 mb-5">
        <div className="p-2 rounded-xl bg-amber/10 text-amber mt-0.5">
            <Icon size={16} />
        </div>
        <div>
            <h3 className="font-display text-lg font-bold text-chalk">{title}</h3>
            <p className="text-sm text-smoke mt-0.5">{description}</p>
        </div>
    </div>
);

export const Settings = () => {
    const { addToast } = useToast();
    const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
    const [hasChanges, setHasChanges] = useState(false);
    const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
    const [apiLatency, setApiLatency] = useState<number | null>(null);
    const [confirmReset, setConfirmReset] = useState(false);

    // Load settings from localStorage
    useEffect(() => {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            try {
                setSettings(JSON.parse(saved));
            } catch { /* use defaults */ }
        }
    }, []);

    // Check API connectivity
    useEffect(() => {
        const checkApi = async () => {
            try {
                const start = performance.now();
                const res = await api.get('/api/health');
                const latency = Math.round(performance.now() - start);
                setApiLatency(latency);
                setApiStatus(res.status === 200 ? 'online' : 'offline');
            } catch {
                setApiStatus('offline');
                setApiLatency(null);
            }
        };
        checkApi();
        const interval = setInterval(checkApi, 15000);
        return () => clearInterval(interval);
    }, []);

    const updateSetting = <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => {
        setSettings(prev => ({ ...prev, [key]: value }));
        setHasChanges(true);
    };

    const handleSave = () => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
        setHasChanges(false);
        addToast('success', 'Settings saved successfully');
    };

    const handleReset = () => {
        if (!confirmReset) {
            setConfirmReset(true);
            setTimeout(() => setConfirmReset(false), 3000);
            return;
        }
        setSettings(DEFAULT_SETTINGS);
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem('alert_thresholds');
        setHasChanges(false);
        setConfirmReset(false);
        addToast('info', 'All settings reset to defaults');
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        show: { opacity: 1, transition: { staggerChildren: 0.08 } }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 16 },
        show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }
    };

    const confidenceProgress = ((settings.confidenceThreshold - 50) / 40) * 100;

    return (
        <motion.div
            className="w-full max-w-3xl mx-auto py-16 md:py-20 px-4 md:px-6"
            variants={containerVariants}
            initial="hidden"
            animate="show"
        >
            {/* Page Header */}
            <motion.div variants={itemVariants} className="mb-8">
                <h1 className="font-display text-3xl md:text-4xl font-bold text-chalk tracking-tight">Configuration</h1>
                <p className="text-smoke text-base md:text-lg mt-1">Manage engine parameters, alerts, and system settings</p>
            </motion.div>

            {/* Sticky Save Bar - appears when there are changes */}
            {hasChanges && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="sticky top-20 z-30 mb-6"
                >
                    <div className="panel p-3 flex items-center justify-between gap-3 border-amber/30 bg-obsidian/95 backdrop-blur-xl">
                        <div className="flex items-center gap-2 text-sm text-amber">
                            <AlertTriangle size={14} />
                            <span className="font-medium">Unsaved changes</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => {
                                    const saved = localStorage.getItem(STORAGE_KEY);
                                    if (saved) setSettings(JSON.parse(saved));
                                    else setSettings(DEFAULT_SETTINGS);
                                    setHasChanges(false);
                                }}
                                className="btn-ghost text-xs py-1.5 px-3 cursor-pointer"
                            >
                                Discard
                            </button>
                            <button
                                onClick={handleSave}
                                className="flex items-center gap-1.5 px-4 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-b from-amber to-amber/90 text-void shadow-lg shadow-amber/20 hover:shadow-xl cursor-pointer transition-all"
                            >
                                <Save size={12} />
                                Save
                            </button>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* === SECTION 1: System Status === */}
            <motion.div variants={itemVariants} className="panel p-5 md:p-6 mb-4">
                <SectionHeader
                    icon={Server}
                    title="System Status"
                    description="Backend API health and engine information"
                />
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {/* API Status */}
                    <div className="flex items-center gap-3 p-3 bg-void/40 rounded-xl border border-graphite/30">
                        {apiStatus === 'online' ? (
                            <div className="relative">
                                <Wifi size={16} className="text-sage" />
                                <div className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-sage rounded-full animate-pulse" />
                            </div>
                        ) : apiStatus === 'offline' ? (
                            <WifiOff size={16} className="text-crimson" />
                        ) : (
                            <div className="w-4 h-4 border-2 border-smoke border-t-amber rounded-full animate-spin" />
                        )}
                        <div>
                            <div className="text-xs text-smoke">API</div>
                            <div className={`text-sm font-semibold ${apiStatus === 'online' ? 'text-sage' : apiStatus === 'offline' ? 'text-crimson' : 'text-smoke'}`}>
                                {apiStatus === 'online' ? 'Connected' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
                            </div>
                        </div>
                    </div>
                    {/* Latency */}
                    <div className="flex items-center gap-3 p-3 bg-void/40 rounded-xl border border-graphite/30">
                        <Zap size={16} className={apiLatency !== null && apiLatency < 200 ? 'text-sage' : 'text-amber'} />
                        <div>
                            <div className="text-xs text-smoke">Latency</div>
                            <div className="text-sm font-mono font-semibold text-chalk">
                                {apiLatency !== null ? `${apiLatency}ms` : '—'}
                            </div>
                        </div>
                    </div>
                    {/* Engine */}
                    <div className="flex items-center gap-3 p-3 bg-void/40 rounded-xl border border-graphite/30">
                        <Cpu size={16} className="text-amber" />
                        <div>
                            <div className="text-xs text-smoke">Engine</div>
                            <div className="text-sm font-semibold text-chalk">Signal.Engine</div>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* === SECTION 2: Trading Parameters === */}
            <motion.div variants={itemVariants} className="panel p-5 md:p-6 mb-4">
                <SectionHeader
                    icon={SlidersHorizontal}
                    title="Trading Parameters"
                    description="Configure the engine's scanning behavior and risk limits"
                />

                {/* Universe Selection */}
                <div className="mb-6">
                    <label className="text-xs text-smoke uppercase tracking-wider font-semibold mb-3 block">Market Universe</label>
                    <div className="grid grid-cols-2 gap-3">
                        <button
                            onClick={() => updateSetting('universe', 'nifty50')}
                            className={`cursor-pointer relative p-4 rounded-xl border-2 transition-all duration-200 text-left ${settings.universe === 'nifty50'
                                ? 'border-amber bg-amber/5 shadow-lg shadow-amber/10'
                                : 'border-graphite/50 bg-void/30 hover:border-graphite hover:bg-void/50'
                                }`}
                        >
                            <div className="flex items-center gap-2 mb-1">
                                <Globe size={14} className={settings.universe === 'nifty50' ? 'text-amber' : 'text-smoke'} />
                                <span className={`font-semibold text-sm ${settings.universe === 'nifty50' ? 'text-amber' : 'text-chalk'}`}>
                                    Nifty 500
                                </span>
                            </div>
                            <p className="text-[11px] text-smoke">Indian market — 500 large-cap stocks</p>
                            {settings.universe === 'nifty50' && (
                                <motion.div
                                    layoutId="universe-check"
                                    className="absolute top-3 right-3"
                                >
                                    <CheckCircle2 size={14} className="text-amber" />
                                </motion.div>
                            )}
                        </button>
                        <button
                            onClick={() => updateSetting('universe', 'nasdaq')}
                            className={`cursor-pointer relative p-4 rounded-xl border-2 transition-all duration-200 text-left ${settings.universe === 'nasdaq'
                                ? 'border-amber bg-amber/5 shadow-lg shadow-amber/10'
                                : 'border-graphite/50 bg-void/30 hover:border-graphite hover:bg-void/50'
                                }`}
                        >
                            <div className="flex items-center gap-2 mb-1">
                                <Globe size={14} className={settings.universe === 'nasdaq' ? 'text-amber' : 'text-smoke'} />
                                <span className={`font-semibold text-sm ${settings.universe === 'nasdaq' ? 'text-amber' : 'text-chalk'}`}>
                                    US Tech
                                </span>
                            </div>
                            <p className="text-[11px] text-smoke">Nasdaq — Top tech equities</p>
                            {settings.universe === 'nasdaq' && (
                                <motion.div
                                    layoutId="universe-check"
                                    className="absolute top-3 right-3"
                                >
                                    <CheckCircle2 size={14} className="text-amber" />
                                </motion.div>
                            )}
                        </button>
                    </div>
                </div>

                <div className="h-px bg-graphite/40 mb-6" />

                {/* Confidence Threshold Slider */}
                <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-xs text-smoke uppercase tracking-wider font-semibold" htmlFor="confidence-threshold">
                            Min Confidence Threshold
                        </label>
                        <span className="font-mono text-lg font-bold text-amber">{settings.confidenceThreshold}%</span>
                    </div>
                    <input
                        id="confidence-threshold"
                        type="range"
                        min={50}
                        max={90}
                        value={settings.confidenceThreshold}
                        onChange={(e) => updateSetting('confidenceThreshold', parseInt(e.target.value))}
                        className="range-amber w-full"
                        style={{ '--range-progress': `${confidenceProgress}%` } as React.CSSProperties}
                    />
                    <div className="flex justify-between text-[10px] text-smoke/50 mt-1.5 font-mono">
                        <span>50% Aggressive</span>
                        <span>70% Balanced</span>
                        <span>90% Conservative</span>
                    </div>
                </div>

                <div className="h-px bg-graphite/40 mb-6" />

                {/* Max Positions Counter */}
                <div>
                    <div className="flex items-center justify-between">
                        <div>
                            <label className="text-xs text-smoke uppercase tracking-wider font-semibold block" htmlFor="max-positions">
                                Max Open Positions
                            </label>
                            <p className="text-[11px] text-smoke/60 mt-0.5">
                                Limits simultaneous portfolio holdings
                            </p>
                        </div>
                        <div className="flex items-center gap-1">
                            <button
                                onClick={() => updateSetting('maxPositions', Math.max(1, settings.maxPositions - 1))}
                                className="w-8 h-8 rounded-lg bg-void border border-graphite/50 flex items-center justify-center text-smoke hover:text-chalk hover:border-graphite transition-colors cursor-pointer"
                            >
                                <Minus size={14} />
                            </button>
                            <div className="w-12 h-8 rounded-lg bg-void border border-graphite/50 flex items-center justify-center">
                                <span className="font-mono text-base font-bold text-chalk">{settings.maxPositions}</span>
                            </div>
                            <button
                                onClick={() => updateSetting('maxPositions', Math.min(20, settings.maxPositions + 1))}
                                className="w-8 h-8 rounded-lg bg-void border border-graphite/50 flex items-center justify-center text-smoke hover:text-chalk hover:border-graphite transition-colors cursor-pointer"
                            >
                                <Plus size={14} />
                            </button>
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* === SECTION 3: Notifications === */}
            <motion.div variants={itemVariants} className="panel p-5 md:p-6 mb-4">
                <SectionHeader
                    icon={Bell}
                    title="Notifications"
                    description="Browser push alerts for signals and portfolio events"
                />
                <AlertSettings />
            </motion.div>

            {/* === SECTION 4: Danger Zone === */}
            <motion.div variants={itemVariants} className="panel p-5 md:p-6 mb-4 border-crimson/10">
                <SectionHeader
                    icon={Trash2}
                    title="Danger Zone"
                    description="Destructive actions that cannot be undone"
                />
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-4 bg-crimson/5 border border-crimson/15 rounded-xl">
                    <div>
                        <div className="text-sm font-semibold text-chalk">Reset All Settings</div>
                        <p className="text-xs text-smoke mt-0.5">Restore all parameters and alert thresholds to factory defaults</p>
                    </div>
                    <button
                        onClick={handleReset}
                        className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer shrink-0 ${confirmReset
                            ? 'bg-crimson text-white shadow-lg shadow-crimson/20'
                            : 'bg-crimson/10 text-crimson border border-crimson/20 hover:bg-crimson/20'
                            }`}
                    >
                        {confirmReset ? (
                            <>
                                <AlertTriangle size={14} />
                                Confirm Reset
                            </>
                        ) : (
                            <>
                                <RotateCcw size={14} />
                                Reset Defaults
                            </>
                        )}
                    </button>
                </div>
                {/* Warning */}
                <div className="mt-4 p-3 bg-amber/5 border border-amber/15 rounded-xl text-amber text-xs flex items-start gap-2.5">
                    <AlertTriangle size={13} className="mt-0.5 shrink-0" />
                    <span>Configuration changes require a backend restart in this version. After saving, restart the server for changes to take effect.</span>
                </div>
            </motion.div>

            {/* === SECTION 5: About Footer === */}
            <motion.div variants={itemVariants} className="mt-2">
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-2 py-4 text-smoke/50">
                    <div className="flex items-center gap-2 text-xs">
                        <Info size={12} />
                        <span>Signal.Engine</span>
                        <span className="px-1.5 py-0.5 rounded bg-graphite/30 text-[10px] font-mono font-semibold text-smoke">v2.5.0</span>
                    </div>
                    <div className="flex items-center gap-2 text-[10px] font-mono">
                        <span className="px-1.5 py-0.5 rounded bg-graphite/20 text-smoke/40">React</span>
                        <span className="px-1.5 py-0.5 rounded bg-graphite/20 text-smoke/40">FastAPI</span>
                        <span className="px-1.5 py-0.5 rounded bg-graphite/20 text-smoke/40">PyTorch</span>
                        <span className="px-1.5 py-0.5 rounded bg-graphite/20 text-smoke/40">Monte Carlo</span>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    );
};
