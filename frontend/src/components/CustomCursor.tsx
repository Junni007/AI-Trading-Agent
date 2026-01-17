import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export const CustomCursor = () => {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        // Only show cursor on non-touch devices
        const isTouchDevice = 'ontouchstart' in window;
        if (isTouchDevice) return;

        setIsVisible(true);

        const mouseMove = (e: MouseEvent) => {
            setMousePosition({ x: e.clientX, y: e.clientY });
        };

        window.addEventListener("mousemove", mouseMove);
        return () => window.removeEventListener("mousemove", mouseMove);
    }, []);

    if (!isVisible) return null;

    return (
        <>
            {/* Core dot */}
            <motion.div
                className="hidden md:block fixed top-0 left-0 w-2 h-2 bg-amber rounded-full pointer-events-none z-[9999]"
                style={{ mixBlendMode: 'difference' }}
                animate={{ x: mousePosition.x - 4, y: mousePosition.y - 4 }}
                transition={{ type: "tween", ease: "backOut", duration: 0 }}
            />
            {/* Outer ring */}
            <motion.div
                className="hidden md:block fixed top-0 left-0 w-8 h-8 border border-amber/40 rounded-full pointer-events-none z-[9999]"
                animate={{ x: mousePosition.x - 16, y: mousePosition.y - 16 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
            />
        </>
    );
};
