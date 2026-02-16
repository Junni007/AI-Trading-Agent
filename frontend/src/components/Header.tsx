import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';

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
    const [mobileOpen, setMobileOpen] = useState(false);

    // Close mobile menu on route change
    useEffect(() => {
        setMobileOpen(false);
    }, [location.pathname]);

    // Close on Escape key
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setMobileOpen(false);
        };
        if (mobileOpen) {
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [mobileOpen]);

    const navItems = [
        { path: '/', label: 'Dashboard' },
        { path: '/signals', label: 'Signals' },
        { path: '/analytics', label: 'Analytics' },
        { path: '/settings', label: 'Config' },
    ];

    return (
        <>
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
                        <span className="text-[9px] uppercase tracking-[0.25em] text-smoke font-medium hidden sm:inline">
                            Trading Intelligence
                        </span>
                    </div>
                </Link>

                {/* Desktop Navigation */}
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

                {/* Right side: Status + Mobile Burger */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-sage animate-pulse" />
                        <span className="text-xs font-mono text-smoke hidden sm:inline">ONLINE</span>
                    </div>

                    {/* Mobile Hamburger */}
                    <button
                        onClick={() => setMobileOpen(!mobileOpen)}
                        className="md:hidden p-2 rounded-lg bg-obsidian/60 border border-graphite/50 text-smoke hover:text-chalk transition-colors cursor-pointer"
                        aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
                    >
                        {mobileOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>
                </div>
            </nav>

            {/* Mobile Drawer */}
            <AnimatePresence>
                {mobileOpen && (
                    <>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="fixed inset-0 bg-void/80 backdrop-blur-sm z-40 md:hidden"
                            onClick={() => setMobileOpen(false)}
                        />

                        {/* Drawer Panel */}
                        <motion.div
                            initial={{ x: '100%' }}
                            animate={{ x: 0 }}
                            exit={{ x: '100%' }}
                            transition={{ type: 'spring', stiffness: 400, damping: 40 }}
                            className="fixed top-0 right-0 h-full w-72 bg-obsidian border-l border-graphite/50 z-50 md:hidden flex flex-col"
                        >
                            {/* Drawer Header */}
                            <div className="flex items-center justify-between p-6 border-b border-graphite/50">
                                <span className="font-display text-lg font-bold text-chalk">Navigate</span>
                                <button
                                    onClick={() => setMobileOpen(false)}
                                    className="p-2 rounded-lg hover:bg-slate text-smoke hover:text-chalk transition-colors cursor-pointer"
                                    aria-label="Close menu"
                                >
                                    <X size={18} />
                                </button>
                            </div>

                            {/* Nav Links */}
                            <div className="flex flex-col p-4 gap-1">
                                {navItems.map((item, i) => {
                                    const isActive = location.pathname === item.path;
                                    return (
                                        <motion.div
                                            key={item.path}
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.05 }}
                                        >
                                            <Link
                                                to={item.path}
                                                className={`block px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer ${isActive
                                                    ? 'bg-slate text-chalk border border-graphite'
                                                    : 'text-smoke hover:text-chalk hover:bg-slate/50'
                                                    }`}
                                            >
                                                {item.label}
                                            </Link>
                                        </motion.div>
                                    );
                                })}
                            </div>

                            {/* Drawer Footer */}
                            <div className="mt-auto p-6 border-t border-graphite/50">
                                <div className="flex items-center gap-2 text-xs text-smoke">
                                    <div className="w-2 h-2 rounded-full bg-sage animate-pulse" />
                                    <span className="font-mono">System Online</span>
                                </div>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </>
    );
};
