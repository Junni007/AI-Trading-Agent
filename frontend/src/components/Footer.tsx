export const Footer = () => {
    return (
        <footer className="w-full py-8 mt-20 border-t border-graphite/50 bg-obsidian/50 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center text-sm text-smoke">
                <div className="flex items-center gap-2 mb-4 md:mb-0">
                    <span className="font-mono text-amber">◆</span>
                    <span className="font-body">Observatory &copy; 2026</span>
                </div>

                <div className="flex gap-6 text-smoke/70">
                    <a href="https://ai-trading-agent.superdocs.cloud/" target="_blank" className="hover:text-amber transition-colors">Docs</a>
                    <a href="#" className="hover:text-amber transition-colors">Status</a>
                    <a href="#" className="hover:text-amber transition-colors">Legal</a>
                </div>
            </div>
        </footer>
    );
};
