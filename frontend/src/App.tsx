import { useEffect, useState } from 'react';
import api from './services/api';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { CustomCursor } from './components/CustomCursor';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { DetailsModal } from './components/DetailsModal';
import { ToastProvider, useToast } from './components/Toast';
import { notificationService } from './services/notificationService';
import { ErrorBoundary } from './components/ErrorBoundary';
import { WifiOff } from 'lucide-react';

// Pages
import { Dashboard } from './pages/Dashboard';
import { Signals } from './pages/Signals';
import { Settings } from './pages/Settings';
import { Analytics } from './pages/Analytics';

interface Decision {
    Ticker: string;
    Action: string;
    Confidence: number;
    Rational: string[];
    History?: any[];
    QuantRisk?: {
        WinRate: number;
        EV: number;
        VaR95: number;
        MaxDrawdown?: number;
    };
}

// Scroll to top on route changes
const ScrollToTop = () => {
    const { pathname } = useLocation();
    useEffect(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, [pathname]);
    return null;
};

function AppContent() {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<Decision[]>([]);
    const [marketMood, setMarketMood] = useState('NEUTRAL');
    const [lastUpdated, setLastUpdated] = useState<string>('');
    const [simState, setSimState] = useState<any>(null);
    const [isAuto, setIsAuto] = useState(true);
    const [logs, setLogs] = useState<string[]>([]);
    const [isConnected, setIsConnected] = useState(true);
    const [connectionDismissed, setConnectionDismissed] = useState(false);

    // Modal State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalData, setModalData] = useState({
        ticker: '',
        action: '',
        rational: [] as string[],
        confidence: 0,
        history: [] as any[],
        quant: undefined as { WinRate: number; EV: number; VaR95?: number; MaxDrawdown?: number } | undefined
    });

    const { addToast } = useToast();

    // Fetch initial Sim State
    useEffect(() => {
        api.get('/api/simulation/state')
            .then(res => {
                setSimState(res.data);
                setIsConnected(true);
            })
            .catch(err => {
                console.error(err);
                setIsConnected(false);
            });
    }, []);

    // Real-Time Loop
    useEffect(() => {
        let interval: any;
        if (isAuto) {
            interval = setInterval(() => {
                if (!loading) runScan(true);
            }, 2000);
        }
        return () => clearInterval(interval);
    }, [isAuto, loading]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Escape closes modal
            if (e.key === 'Escape' && isModalOpen) {
                setIsModalOpen(false);
            }
        };
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, [isModalOpen]);

    const runScan = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            await api.get('/api/scan');
            const res = await api.get('/api/results');

            if (!isConnected) {
                setIsConnected(true);
                setConnectionDismissed(false);
                addToast('success', 'Connection restored');
            }

            if (res.data.status === 'success') {
                if (res.data.data && res.data.data.length > 0) {
                    setData(res.data.data);
                    deriveMarketMood(res.data.data);

                    // Send notifications for high-confidence signals
                    res.data.data.forEach((signal: Decision) => {
                        if (signal.Confidence >= 0.85) {
                            notificationService.sendSignalAlert(
                                signal.Ticker,
                                signal.Action,
                                signal.Confidence
                            );
                        }
                    });
                }
                if (res.data.simulation) setSimState(res.data.simulation);
                if (res.data.logs) setLogs(prev => [...res.data.logs.slice(0, 5), ...prev].slice(0, 50));
                setLastUpdated(new Date().toLocaleTimeString());
            }
        } catch (err) {
            console.error("Failed to fetch brain data", err);
            if (isConnected) {
                setIsConnected(false);
                setConnectionDismissed(false);
            }
        }
        if (!silent) setLoading(false);
    };

    const resetSim = async () => {
        try {
            const res = await api.post('/api/simulation/reset');
            setSimState(res.data.state);
            addToast('success', 'Simulation reset successfully');
        } catch (err) {
            console.error(err);
            addToast('error', 'Failed to reset simulation');
        }
    };

    const deriveMarketMood = (decisions: Decision[]) => {
        const volCount = decisions.filter(d => d.Rational.some(r => r.includes("High Volatility"))).length;
        if (volCount > decisions.length / 2) setMarketMood("VOLATILE");
        else setMarketMood("CALM");
    };

    const openDetails = (ticker: string, action: string, rational: string[], confidence: number, history: any[], quant?: any) => {
        setModalData({ ticker, action, rational, confidence, history, quant });
        setIsModalOpen(true);
    };

    useEffect(() => {
        runScan();
    }, []);

    const isLocked = isAuto && simState?.status !== 'DEAD';

    return (
        <div className="min-h-screen bg-void text-chalk font-body selection:bg-amber/20 cursor-fancy flex flex-col relative">

            <CustomCursor />
            <Header />

            {/* Connection Error Banner */}
            {!isConnected && !connectionDismissed && (
                <div className="w-full bg-crimson/10 border-b border-crimson/20 px-6 py-3 flex items-center justify-center gap-3 relative z-40">
                    <WifiOff size={14} className="text-crimson shrink-0" />
                    <span className="text-sm text-crimson font-medium">
                        Unable to reach backend API. Data may be stale.
                    </span>
                    <button
                        onClick={() => setConnectionDismissed(true)}
                        className="text-crimson/60 hover:text-crimson text-xs ml-4 cursor-pointer"
                        aria-label="Dismiss"
                    >
                        Dismiss
                    </button>
                </div>
            )}

            <DetailsModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                {...modalData}
            />

            <main className="flex-grow flex flex-col items-center max-w-7xl mx-auto w-full px-4">
                <ScrollToTop />
                <Routes>
                    <Route path="/" element={
                        <Dashboard
                            data={data}
                            loading={loading}
                            marketMood={marketMood}
                            lastUpdated={lastUpdated}
                            onScan={() => {
                                if (isLocked) return;
                                setIsAuto(!isAuto);
                            }}
                            isAuto={isAuto}
                            isLocked={isLocked}
                            onDetails={openDetails}
                            simState={simState}
                            onResetSim={resetSim}
                            logs={logs}
                        />
                    } />
                    <Route path="/signals" element={<Signals data={data} loading={loading} onDetails={openDetails} />} />
                    <Route path="/analytics" element={<Analytics simState={simState} />} />
                    <Route path="/settings" element={<Settings />} />
                </Routes>
            </main>

            <Footer />
        </div>
    )
}

function App() {
    return (
        <Router>
            <ToastProvider>
                <ErrorBoundary>
                    <AppContent />
                </ErrorBoundary>
            </ToastProvider>
        </Router>
    );
}

export default App
