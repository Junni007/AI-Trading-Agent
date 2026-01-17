/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // === Observatory Palette ===
                // Deep space backgrounds
                void: "#0D0D0F",
                obsidian: "#141418",
                slate: "#1E1E24",
                graphite: "#2A2A32",
                iron: "#3D3D47",

                // Signal Colors (intentional, not decorative)
                amber: "#F59E0B",
                ember: "#F97316",
                sage: "#10B981",
                crimson: "#EF4444",

                // Typography Colors
                chalk: "#F5F5F5",
                ash: "#A1A1AA",
                smoke: "#71717A",

                // Legacy aliases (for gradual migration)
                marine: "#0D0D0F",
                gunmetal: "#141418",
                teal: "#F59E0B",
                mist: "#F5F5F5",
            },
            fontFamily: {
                display: ['Syne', 'sans-serif'],
                body: ['Figtree', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
                sans: ['Figtree', 'sans-serif'],
            },
            backgroundImage: {
                'noise': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E\")",
                'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
            },
            boxShadow: {
                'inner-glow': 'inset 0 1px 0 0 rgba(255,255,255,0.05)',
                'signal': '0 0 30px -5px var(--tw-shadow-color)',
                'elevated': '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow': 'glow 2s ease-in-out infinite alternate',
            },
            keyframes: {
                glow: {
                    '0%': { opacity: '0.5' },
                    '100%': { opacity: '1' },
                },
            },
        },
    },
    plugins: [],
}
