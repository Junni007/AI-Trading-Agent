import { Link } from 'react-router-dom';

export const Footer = () => {
    return (
        <footer className="w-full py-8 mt-20 border-t border-graphite/50 bg-obsidian/50 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center text-sm text-smoke">
                <div className="flex items-center gap-2 mb-4 md:mb-0">
                    <span className="font-mono text-amber">◆</span>
                    <span className="font-body">Signal.Engine &copy; {new Date().getFullYear()}</span>
                </div>

                <div className="flex gap-6 text-smoke/70">
                    <a href="https://ai-trading-agent.superdocs.cloud/" target="_blank" rel="noopener noreferrer" className="hover:text-amber transition-colors cursor-pointer">Docs</a>
                    <Link to="/analytics" className="hover:text-amber transition-colors cursor-pointer">Analytics</Link>
                    <Link to="/settings" className="hover:text-amber transition-colors cursor-pointer">Settings</Link>
                </div>
            </div>
        </footer>
    );
};
