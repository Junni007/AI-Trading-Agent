import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

interface HeroProps {
    onScan: () => void;
    loading: boolean;
    isAuto?: boolean;
    isLocked?: boolean;
    marketMood: string;
    lastUpdated?: string;
}

export const Hero = ({ onScan, loading, isAuto, isLocked, marketMood, lastUpdated }: HeroProps) => {

    // Track active time with persistence
    const [activeTime, setActiveTime] = useState(0);

    useEffect(() => {
        const STORAGE_KEY = 'signal_engine_start_time';

        // Get or set the start timestamp
        let startTime = localStorage.getItem(STORAGE_KEY);
        if (!startTime) {
            startTime = Date.now().toString();
            localStorage.setItem(STORAGE_KEY, startTime);
        }

        // Update every second
        const updateTime = () => {
            const elapsed = Math.floor((Date.now() - parseInt(startTime!)) / 1000);
            setActiveTime(elapsed);
        };

        updateTime();
        const interval = setInterval(updateTime, 1000);

        return () => clearInterval(interval);
    }, []);

    // Format active time as Xd HH:MM:SS
    const formatTime = (seconds: number) => {
        const days = Math.floor(seconds / 86400);
        const hrs = Math.floor((seconds % 86400) / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        const timeStr = `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        return days > 0 ? `${days}d ${timeStr}` : timeStr;
    };

    // Mood configuration
    const getMoodConfig = () => {
        if (marketMood.includes("FEAR")) return { color: "text-crimson", glow: "shadow-crimson/20" };
        if (marketMood.includes("GREED")) return { color: "text-sage", glow: "shadow-sage/20" };
        return { color: "text-amber", glow: "shadow-amber/20" };
    };

    const mood = getMoodConfig();

    return (
        <section className="relative w-full py-20 md:py-28 px-6 flex flex-col items-center text-center overflow-hidden">

            {/* Atmospheric Glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] pointer-events-none">
                <div className="absolute inset-0 bg-gradient-radial from-amber/10 via-amber/5 to-transparent blur-3xl" />
            </div>

            {/* Grid Pattern Overlay */}
            <div
                className="absolute inset-0 opacity-[0.015] pointer-events-none"
                style={{
                    backgroundImage: `linear-gradient(rgba(245,158,11,0.3) 1px, transparent 1px), 
                                      linear-gradient(90deg, rgba(245,158,11,0.3) 1px, transparent 1px)`,
                    backgroundSize: '60px 60px'
                }}
            />

            {/* Headline */}
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                className="relative z-10"
            >
                <h1 className="font-display text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter text-chalk leading-none mb-2">
                    <span className="text-gradient-amber">Signal</span>
                    <span className="text-chalk/40">.</span>
                    <span>Engine</span>
                </h1>
            </motion.div>

            <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.6 }}
                className="font-body text-ash text-lg md:text-xl font-light tracking-wide max-w-md mt-4 mb-12"
            >
                Hybrid intelligence for calculated opportunities
            </motion.p>

            {/* Control Unit */}
            <div className="flex flex-col items-center gap-6 relative z-20">

                {/* Primary CTA */}
                <motion.button
                    onClick={onScan}
                    disabled={(loading && !isAuto) || isLocked}
                    whileHover={isLocked ? {} : { scale: 1.03 }}
                    whileTap={isLocked ? {} : { scale: 0.97 }}
                    className={`
                        group relative flex items-center justify-center gap-3 px-8 py-4 rounded-2xl font-body font-semibold text-sm tracking-wide transition-all duration-300
                        ${isLocked
                            ? 'bg-iron/30 border border-iron/50 cursor-not-allowed text-smoke'
                            : isAuto
                                ? 'bg-gradient-to-r from-crimson/20 to-ember/20 border border-crimson/40 text-crimson shadow-lg shadow-crimson/10'
                                : 'bg-gradient-to-b from-amber to-amber/90 text-void shadow-lg shadow-amber/25 hover:shadow-xl hover:shadow-amber/35'
                        }
                    `}
                >
                    {/* Pulse indicator */}
                    <div className={`w-2.5 h-2.5 rounded-full transition-colors duration-300 ${isLocked ? 'bg-iron' : isAuto ? 'bg-crimson animate-pulse' : 'bg-void'
                        }`} />

                    <span className="uppercase tracking-widest">
                        {isLocked ? "🔒 Locked" : isAuto ? "Live — Scanning" : "Initialize Scan"}
                    </span>

                    {/* Glow effect on active */}
                    {isAuto && !isLocked && (
                        <motion.div
                            className="absolute inset-0 rounded-2xl bg-crimson/10 blur-xl -z-10"
                            animate={{ opacity: [0.3, 0.6, 0.3] }}
                            transition={{ repeat: Infinity, duration: 2 }}
                        />
                    )}
                </motion.button>

                {/* Status Panel */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="flex items-center gap-4 text-xs font-mono tracking-wider text-smoke panel px-5 py-2.5"
                >
                    <div className="flex items-center gap-2">
                        <span className="text-ash">REGIME</span>
                        <span className={`font-semibold ${mood.color}`}>{marketMood.split(' ')[0]}</span>
                    </div>

                    <div className="w-px h-4 bg-graphite" />

                    <div className="flex items-center gap-2">
                        <span className="text-ash">SYNC</span>
                        <span className="text-chalk">{lastUpdated || "--:--"}</span>
                    </div>

                    <div className="w-px h-4 bg-graphite" />

                    <div className="flex items-center gap-2">
                        <span className="text-ash">UPTIME</span>
                        <span className={`${isAuto ? 'text-sage' : 'text-smoke'}`}>{formatTime(activeTime)}</span>
                    </div>
                </motion.div>
            </div>
        </section>
    );
};
