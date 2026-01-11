import React, { useEffect, useRef } from 'react';
import { Terminal as TerminalIcon } from 'lucide-react';
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
        <div className="w-full max-w-[1400px] px-4 mb-4">
            <div className="bg-gunmetal/80 border border-white/10 rounded-xl overflow-hidden shadow-2xl backdrop-blur-md">
                {/* Header */}
                <div className="flex items-center gap-2 px-4 py-2 bg-black/40 border-b border-white/5">
                    <TerminalIcon size={14} className="text-teal" />
                    <span className="text-xs font-mono font-bold text-gray-400 uppercase tracking-widest">System Logs</span>
                </div>

                {/* Log Stream */}
                <div
                    ref={scrollContainerRef}
                    className="h-32 overflow-y-auto p-4 font-mono text-xs space-y-1 custom-scrollbar"
                >
                    {logs.length === 0 ? (
                        <div className="text-gray-600 italic">System ready. Waiting for scan...</div>
                    ) : (
                        logs.map((log, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="text-mist/80 border-l-2 border-transparent hover:border-teal/30 pl-2 transition-colors"
                            >
                                <span className="text-teal/50 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                {log}
                            </motion.div>
                        ))
                    )}
                    <div />
                </div>
            </div>
        </div>
    );
};
