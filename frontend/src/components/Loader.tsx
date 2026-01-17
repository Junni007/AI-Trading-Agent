import { motion } from "framer-motion";

export const Loader = () => {
    return (
        <div className="flex flex-col items-center justify-center h-64 space-y-5">
            {/* Pulsing rings */}
            <div className="relative w-16 h-16">
                <motion.div
                    className="absolute inset-0 border-2 border-amber/30 rounded-full"
                    animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                />
                <motion.div
                    className="absolute inset-0 border-2 border-amber/30 rounded-full"
                    animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut", delay: 0.5 }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-3 h-3 bg-amber rounded-full" />
                </div>
            </div>
            <p className="text-smoke font-mono text-sm tracking-wider">
                <span className="text-amber">▣</span> Scanning signals...
            </p>
        </div>
    );
};
