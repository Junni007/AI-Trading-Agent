import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { AlertSettings } from '../components/AlertSettings';
import { AlertTriangle, Save, RotateCcw, CheckCircle2, Wifi, WifiOff } from 'lucide-react';
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

export const Settings = () => {
    const { addToast } = useToast();
    const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS);
    const [hasChanges, setHasChanges] = useState(false);
    const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

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
                const res = await fetch('http://localhost:8000/api/simulation/state');
                setApiStatus(res.ok ? 'online' : 'offline');
            } catch {
                setApiStatus('offline');
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
        setSettings(DEFAULT_SETTINGS);
        localStorage.removeItem(STORAGE_KEY);
        setHasChanges(false);
        addToast('info', 'Settings reset to defaults');
    };

    const containerVariants = {
        hidden: { opacity: 0 },
        show: { opacity: 1, transition: { staggerChildren: 0.08 } }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 16 },
        show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }
    };

    return (
        <motion.div
            className="w-full max-w-3xl mx-auto py-20 px-6"
            variants={containerVariants}
            initial="hidden"
            animate="show"
        >
            <motion.div variants={itemVariants}>
                <h1 className="font-display text-4xl font-bold text-chalk mb-2 tracking-tight">Configuration</h1>
                <p className="text-smoke text-lg mb-8">Manage agent parameters and notifications</p>
            </motion.div>

            {/* API Status */}
            <motion.div variants={itemVariants} className="panel p-4 mb-6 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    {apiStatus === 'online' ? (
                        <Wifi size={16} className="text-sage" />
                    ) : apiStatus === 'offline' ? (
                        <WifiOff size={16} className="text-crimson" />
                    ) : (
                        <div className="w-4 h-4 border-2 border-smoke border-t-amber rounded-full animate-spin" />
                    )}
                    <div>
                        <div className="text-sm font-medium text-chalk">Backend API</div>
                        <div className="text-xs text-smoke font-mono">
                            {apiStatus === 'online' ? 'Connected — localhost:8000' : apiStatus === 'offline' ? 'Unreachable' : 'Checking...'}
                        </div>
                    </div>
                </div>
                <div className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold uppercase tracking-wider ${apiStatus === 'online'
                    ? 'bg-sage/10 text-sage border border-sage/20'
                    : apiStatus === 'offline'
                        ? 'bg-crimson/10 text-crimson border border-crimson/20'
                        : 'bg-iron/30 text-smoke border border-iron/50'
                    }`}>
                    {apiStatus}
                </div>
            </motion.div>

            <motion.div variants={itemVariants} className="panel p-6 md:p-8 space-y-8">
                {/* Universe Selection */}
                <div>
                    <h3 className="font-display text-lg font-bold text-amber mb-4">Universe Selection</h3>
                    <div className="flex gap-3">
                        <button
                            onClick={() => updateSetting('universe', 'nifty50')}
                            className={`cursor-pointer ${settings.universe === 'nifty50' ? 'btn-primary' : 'btn-ghost'}`}
                        >
                            Nifty 50 (India)
                        </button>
                        <button
                            onClick={() => updateSetting('universe', 'nasdaq')}
                            className={`cursor-pointer ${settings.universe === 'nasdaq' ? 'btn-primary' : 'btn-ghost'}`}
                        >
                            US Tech (Nasdaq)
                        </button>
                    </div>
                </div>

                <div className="h-px bg-graphite" />

                {/* Risk Parameters */}
                <div>
                    <h3 className="font-display text-lg font-bold text-amber mb-4">Risk Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="flex flex-col gap-2">
                            <label className="text-sm text-smoke" htmlFor="confidence-threshold">
                                Min Confidence Threshold
                            </label>
                            <input
                                id="confidence-threshold"
                                type="range"
                                min={50}
                                max={90}
                                value={settings.confidenceThreshold}
                                onChange={(e) => updateSetting('confidenceThreshold', parseInt(e.target.value))}
                                className="accent-amber h-2 bg-graphite rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-smoke/60">
                                <span>50%</span>
                                <span className="text-amber font-mono font-semibold">{settings.confidenceThreshold}%</span>
                                <span>90%</span>
                            </div>
                        </div>
                        <div className="flex flex-col gap-2">
                            <label className="text-sm text-smoke" htmlFor="max-positions">Max Open Positions</label>
                            <input
                                id="max-positions"
                                type="number"
                                min={1}
                                max={20}
                                value={settings.maxPositions}
                                onChange={(e) => updateSetting('maxPositions', parseInt(e.target.value) || 1)}
                                className="bg-void border border-graphite rounded-xl py-2.5 px-4 text-chalk font-mono focus:border-amber/50 focus:outline-none transition-colors"
                                placeholder="5"
                            />
                        </div>
                    </div>
                </div>

                <div className="h-px bg-graphite" />

                {/* Save / Reset Actions */}
                <div className="flex items-center gap-3">
                    <button
                        onClick={handleSave}
                        disabled={!hasChanges}
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${hasChanges
                            ? 'bg-gradient-to-b from-amber to-amber/90 text-void shadow-lg shadow-amber/20 hover:shadow-xl hover:shadow-amber/30'
                            : 'bg-iron/30 text-smoke cursor-not-allowed'
                            }`}
                    >
                        {hasChanges ? <Save size={14} /> : <CheckCircle2 size={14} />}
                        {hasChanges ? 'Save Changes' : 'Saved'}
                    </button>
                    <button
                        onClick={handleReset}
                        className="btn-ghost flex items-center gap-2 text-sm cursor-pointer"
                    >
                        <RotateCcw size={14} />
                        Reset Defaults
                    </button>
                </div>

                {/* Warning */}
                <div className="p-4 bg-amber/5 border border-amber/20 rounded-xl text-amber text-sm flex items-start gap-3">
                    <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                    <span>Configuration changes require a backend restart in this version (v1.0).</span>
                </div>
            </motion.div>

            {/* Alert Settings Section */}
            <motion.div variants={itemVariants} className="mt-8">
                <h2 className="font-display text-2xl font-bold text-chalk mb-4">Notifications</h2>
                <AlertSettings />
            </motion.div>
        </motion.div>
    );
};
