import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

// Custom Signal Logo
const SignalLogo = () => (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="1.5" opacity="0.3" />
        <circle cx="16" cy="16" r="8" stroke="currentColor" strokeWidth="1.5" opacity="0.6" />
        <circle cx="16" cy="16" r="3" fill="currentColor" />
        <line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="16" y1="26" x2="16" y2="30" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="2" y1="16" x2="6" y2="16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <line x1="26" y1="16" x2="30" y2="16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
);

export const Header = () => {
    const location = useLocation();

    const navItems = [
        { path: '/', label: 'Dashboard' },
        { path: '/signals', label: 'Signals' },
        { path: '/analytics', label: 'Analytics' },
        { path: '/settings', label: 'Config' },
    ];

    return (
        <nav className="w-full py-5 px-6 md:px-10 flex justify-between items-center relative z-50">
            {/* Logo & Brand */}
            <Link to="/" className="flex items-center gap-3 group">
                <motion.div
                    className="text-amber"
                    whileHover={{ rotate: 45 }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                >
                    <SignalLogo />
                </motion.div>
                <div className="flex flex-col">
                    <span className="font-display text-lg font-bold tracking-tight text-chalk">
                        Signal.Engine
                    </span>
                    <span className="text-[9px] uppercase tracking-[0.25em] text-smoke font-medium">
                        Trading Intelligence
                    </span>
                </div>
            </Link>

            {/* Navigation */}
            <div className="hidden md:flex items-center gap-1 p-1 rounded-xl bg-obsidian/60 border border-graphite/50 backdrop-blur-md">
                {navItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={`relative px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-200 ${isActive
                                ? 'text-chalk'
                                : 'text-smoke hover:text-ash'
                                }`}
                        >
                            {isActive && (
                                <motion.div
                                    layoutId="nav-indicator"
                                    className="absolute inset-0 bg-slate rounded-lg border border-graphite"
                                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                />
                            )}
                            <span className="relative z-10">{item.label}</span>
                        </Link>
                    );
                })}
            </div>

            {/* Status Indicator */}
            <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-sage animate-pulse" />
                <span className="text-xs font-mono text-smoke hidden sm:inline">ONLINE</span>
            </div>
        </nav>
    );
};
