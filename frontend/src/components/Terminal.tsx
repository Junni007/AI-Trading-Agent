import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface TerminalProps {
    logs: string[];
}

export const Terminal: React.FC<TerminalProps> = ({ logs }) => {
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollContainerRef.current) {
            scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="w-full max-w-[1400px] px-4 mb-6">
            <div className="panel overflow-hidden">
                {/* Terminal Header */}
                <div className="flex items-center justify-between px-4 py-2.5 bg-obsidian border-b border-graphite/50">
                    <div className="flex items-center gap-2">
                        <div className="flex gap-1.5">
                            <div className="w-2.5 h-2.5 rounded-full bg-crimson/60" />
                            <div className="w-2.5 h-2.5 rounded-full bg-amber/60" />
                            <div className="w-2.5 h-2.5 rounded-full bg-sage/60" />
                        </div>
                        <span className="text-[10px] font-mono font-semibold text-smoke uppercase tracking-widest ml-2">
                            sys::log
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-sage animate-pulse" />
                        <span className="text-[10px] font-mono text-smoke">STREAMING</span>
                    </div>
                </div>

                {/* Log Stream */}
                <div
                    ref={scrollContainerRef}
                    className="h-36 overflow-y-auto px-4 py-3 font-mono text-xs bg-void/50"
                >
                    {logs.length === 0 ? (
                        <div className="text-smoke/50 italic flex items-center gap-2">
                            <span className="text-amber">$</span> awaiting signal...
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {logs.map((log, i) => (
                                <motion.div
                                    key={`${log}-${i}`}
                                    initial={{ opacity: 0, x: -5 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="flex items-start gap-3 py-0.5 hover:bg-white/[0.02] -mx-2 px-2 rounded transition-colors"
                                >
                                    <span className="text-amber/50 shrink-0">{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                                    <span className="text-ash/90">{log}</span>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Scanline Effect (CSS-based, respects reduced-motion) */}
                <div
                    className="absolute inset-0 pointer-events-none opacity-[0.015]"
                    style={{
                        background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
                    }}
                />
            </div>
        </div>
    );
};
