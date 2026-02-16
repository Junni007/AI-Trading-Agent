import { useState, useCallback, createContext, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertTriangle, Info, X } from 'lucide-react';

type ToastType = 'success' | 'error' | 'info';

interface Toast {
    id: string;
    type: ToastType;
    message: string;
}

interface ToastContextType {
    addToast: (type: ToastType, message: string) => void;
}

const ToastContext = createContext<ToastContextType>({ addToast: () => { } });

export const useToast = () => useContext(ToastContext);

const TOAST_DURATION = 4000;

const toastConfig = {
    success: {
        icon: CheckCircle2,
        className: 'border-sage/30 bg-sage/5',
        iconClass: 'text-sage',
    },
    error: {
        icon: AlertTriangle,
        className: 'border-crimson/30 bg-crimson/5',
        iconClass: 'text-crimson',
    },
    info: {
        icon: Info,
        className: 'border-amber/30 bg-amber/5',
        iconClass: 'text-amber',
    },
};

export const ToastProvider = ({ children }: { children: React.ReactNode }) => {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const addToast = useCallback((type: ToastType, message: string) => {
        const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        setToasts(prev => [...prev, { id, type, message }]);

        setTimeout(() => {
            setToasts(prev => prev.filter(t => t.id !== id));
        }, TOAST_DURATION);
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    return (
        <ToastContext.Provider value={{ addToast }}>
            {children}

            {/* Toast Container */}
            <div className="fixed bottom-6 right-6 z-[200] flex flex-col gap-2 max-w-sm">
                <AnimatePresence>
                    {toasts.map((toast) => {
                        const config = toastConfig[toast.type];
                        const Icon = config.icon;
                        return (
                            <motion.div
                                key={toast.id}
                                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, x: 80, scale: 0.95 }}
                                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                                className={`panel px-4 py-3 flex items-start gap-3 border ${config.className}`}
                            >
                                <Icon size={16} className={`${config.iconClass} mt-0.5 shrink-0`} />
                                <p className="text-sm text-chalk font-body flex-1">{toast.message}</p>
                                <button
                                    onClick={() => removeToast(toast.id)}
                                    className="text-smoke hover:text-chalk transition-colors cursor-pointer shrink-0"
                                    aria-label="Dismiss notification"
                                >
                                    <X size={14} />
                                </button>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>
        </ToastContext.Provider>
    );
};
