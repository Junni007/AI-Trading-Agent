import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Bell, BellOff, Check, X } from 'lucide-react';
import { notificationService, NotificationPermission } from '../services/notificationService';

interface AlertSettingsProps {
    onClose?: () => void;
}

export const AlertSettings: React.FC<AlertSettingsProps> = ({ onClose }) => {
    const [permission, setPermission] = useState<NotificationPermission>('default');
    const [enabled, setEnabled] = useState(false);
    const [thresholds, setThresholds] = useState({
        highConfidence: 0.8,
        profitAlert: 5,
        lossAlert: -3,
    });

    useEffect(() => {
        setPermission(notificationService.getPermission());
        setEnabled(notificationService.isEnabled());

        // Load saved thresholds
        const saved = localStorage.getItem('alert_thresholds');
        if (saved) {
            try {
                setThresholds(JSON.parse(saved));
            } catch { /* ignore */ }
        }
    }, []);

    const handleRequestPermission = async () => {
        const result = await notificationService.requestPermission();
        setPermission(result);
        setEnabled(result === 'granted');
    };

    const handleToggle = () => {
        const newEnabled = !enabled;
        setEnabled(newEnabled);
        notificationService.setEnabled(newEnabled);
    };

    const handleThresholdChange = (key: keyof typeof thresholds, value: number) => {
        const updated = { ...thresholds, [key]: value };
        setThresholds(updated);
        localStorage.setItem('alert_thresholds', JSON.stringify(updated));
    };

    const handleTestNotification = () => {
        notificationService.send({
            title: '🔔 Test Notification',
            body: 'Alerts are working correctly!',
            tag: 'test',
        });
    };

    const isSupported = notificationService.isSupported();

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="panel p-6 w-full max-w-md"
        >
            <div className="flex justify-between items-center mb-6">
                <h3 className="font-display text-lg font-bold text-chalk">Alert Settings</h3>
                {onClose && (
                    <button onClick={onClose} className="text-smoke hover:text-chalk transition-colors">
                        <X size={18} />
                    </button>
                )}
            </div>

            {!isSupported ? (
                <div className="text-sm text-smoke bg-crimson/10 p-3 rounded-lg">
                    ⚠️ Browser notifications are not supported in your browser.
                </div>
            ) : (
                <>
                    {/* Permission Status */}
                    <div className="flex items-center justify-between mb-6 p-4 bg-obsidian rounded-xl">
                        <div className="flex items-center gap-3">
                            {enabled ? (
                                <Bell className="text-sage" size={20} />
                            ) : (
                                <BellOff className="text-smoke" size={20} />
                            )}
                            <div>
                                <div className="font-medium text-chalk text-sm">Notifications</div>
                                <div className="text-xs text-smoke">
                                    {permission === 'granted' ? 'Allowed' : permission === 'denied' ? 'Blocked' : 'Not set'}
                                </div>
                            </div>
                        </div>

                        {permission === 'granted' ? (
                            <button
                                onClick={handleToggle}
                                className={`relative w-12 h-6 rounded-full transition-colors ${enabled ? 'bg-sage' : 'bg-graphite'
                                    }`}
                            >
                                <motion.div
                                    className="absolute top-1 w-4 h-4 bg-white rounded-full"
                                    animate={{ left: enabled ? '26px' : '4px' }}
                                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                                />
                            </button>
                        ) : permission === 'denied' ? (
                            <span className="text-xs text-crimson">Blocked in browser</span>
                        ) : (
                            <button
                                onClick={handleRequestPermission}
                                className="btn-primary text-xs py-2 px-4"
                            >
                                Enable
                            </button>
                        )}
                    </div>

                    {/* Thresholds */}
                    {enabled && (
                        <div className="space-y-4">
                            <div>
                                <label className="text-xs text-smoke uppercase tracking-wider">
                                    Signal Confidence Threshold
                                </label>
                                <div className="flex items-center gap-3 mt-2">
                                    <input
                                        type="range"
                                        min="0.5"
                                        max="1"
                                        step="0.05"
                                        value={thresholds.highConfidence}
                                        onChange={(e) => handleThresholdChange('highConfidence', parseFloat(e.target.value))}
                                        className="flex-1 accent-amber"
                                    />
                                    <span className="font-mono text-sm text-chalk w-12">
                                        {Math.round(thresholds.highConfidence * 100)}%
                                    </span>
                                </div>
                            </div>

                            <div>
                                <label className="text-xs text-smoke uppercase tracking-wider">
                                    Profit Alert (%)
                                </label>
                                <input
                                    type="number"
                                    value={thresholds.profitAlert}
                                    onChange={(e) => handleThresholdChange('profitAlert', parseFloat(e.target.value))}
                                    className="w-full mt-2 panel py-2 px-3 text-sm text-chalk font-mono"
                                />
                            </div>

                            <div>
                                <label className="text-xs text-smoke uppercase tracking-wider">
                                    Loss Alert (%)
                                </label>
                                <input
                                    type="number"
                                    value={thresholds.lossAlert}
                                    onChange={(e) => handleThresholdChange('lossAlert', parseFloat(e.target.value))}
                                    className="w-full mt-2 panel py-2 px-3 text-sm text-chalk font-mono"
                                />
                            </div>

                            {/* Test Button */}
                            <button
                                onClick={handleTestNotification}
                                className="btn-ghost w-full mt-4 flex items-center justify-center gap-2"
                            >
                                <Check size={14} />
                                Test Notification
                            </button>
                        </div>
                    )}
                </>
            )}
        </motion.div>
    );
};
