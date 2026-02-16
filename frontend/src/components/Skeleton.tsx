import { motion } from 'framer-motion';

/**
 * Reusable skeleton loader with amber shimmer animation.
 * Matches the Observatory design system.
 */
export const Skeleton = ({
    className = '',
    variant = 'rect',
}: {
    className?: string;
    variant?: 'rect' | 'circle' | 'text';
}) => {
    const baseClasses = `relative overflow-hidden bg-graphite/30 ${variant === 'circle' ? 'rounded-full' : variant === 'text' ? 'rounded-md h-3' : 'rounded-xl'
        } ${className}`;

    return (
        <div className={baseClasses}>
            <motion.div
                className="absolute inset-0"
                style={{
                    background: 'linear-gradient(90deg, transparent 0%, rgba(245,158,11,0.06) 50%, transparent 100%)',
                }}
                animate={{ x: ['-100%', '100%'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
            />
        </div>
    );
};

/**
 * Skeleton for a signal/thinking card.
 */
export const CardSkeleton = () => (
    <div className="panel p-5 space-y-3">
        <div className="flex items-center gap-3">
            <Skeleton variant="circle" className="w-10 h-10" />
            <div className="flex-1 space-y-2">
                <Skeleton variant="text" className="w-24" />
                <Skeleton variant="text" className="w-16" />
            </div>
            <Skeleton className="w-16 h-6 rounded-lg" />
        </div>
        <Skeleton className="w-full h-20" />
        <div className="flex gap-2">
            <Skeleton variant="text" className="w-20" />
            <Skeleton variant="text" className="w-20" />
        </div>
    </div>
);

/**
 * Empty state component when no data exists.
 */
export const EmptyState = ({
    icon: Icon,
    title,
    description,
    action,
    onAction,
}: {
    icon: React.ElementType;
    title: string;
    description: string;
    action?: string;
    onAction?: () => void;
}) => (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
        <div className="w-14 h-14 rounded-2xl bg-amber/5 border border-amber/10 flex items-center justify-center mb-4">
            <Icon size={24} className="text-amber/40" />
        </div>
        <h3 className="font-display text-lg font-bold text-chalk/80 mb-1">{title}</h3>
        <p className="text-sm text-smoke max-w-sm">{description}</p>
        {action && onAction && (
            <button
                onClick={onAction}
                className="mt-4 flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-b from-amber to-amber/90 text-void shadow-lg shadow-amber/20 cursor-pointer transition-all hover:shadow-xl"
            >
                {action}
            </button>
        )}
    </div>
);
