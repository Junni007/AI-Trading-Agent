import { useEffect, useState } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CustomCursor } from './components/CustomCursor';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { DetailsModal } from './components/DetailsModal';
import { notificationService } from './services/notificationService';

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

function App() {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<Decision[]>([]);
    const [marketMood, setMarketMood] = useState('NEUTRAL');
    const [lastUpdated, setLastUpdated] = useState<string>('');
    const [simState, setSimState] = useState<any>(null);
    const [isAuto, setIsAuto] = useState(true);
    const [logs, setLogs] = useState<string[]>([]);

    // Modal State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalData, setModalData] = useState({
        ticker: '',
        action: '',
        rational: [] as string[],
        confidence: 0,
        history: [] as any[]
    });

    // Fetch initial Sim State
    useEffect(() => {
        axios.get('http://localhost:8000/api/simulation/state')
            .then(res => setSimState(res.data))
            .catch(err => console.error(err));
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

    const runScan = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            await axios.get('http://localhost:8000/api/scan');
            const res = await axios.get('http://localhost:8000/api/results');

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
        }
        if (!silent) setLoading(false);
    };

    const resetSim = async () => {
        try {
            const res = await axios.post('http://localhost:8000/api/simulation/reset');
            setSimState(res.data.state);
        } catch (err) { console.error(err); }
    };

    const deriveMarketMood = (decisions: Decision[]) => {
        const volCount = decisions.filter(d => d.Rational.some(r => r.includes("High Volatility"))).length;
        if (volCount > decisions.length / 2) setMarketMood("VOLATILE");
        else setMarketMood("CALM");
    };

    const openDetails = (ticker: string, action: string, rational: string[], confidence: number, history: any[]) => {
        setModalData({ ticker, action, rational, confidence, history });
        setIsModalOpen(true);
    };

    useEffect(() => {
        runScan();
    }, []);

    const isLocked = isAuto && simState?.status !== 'DEAD';

    return (
        <Router>
            <div className="min-h-screen bg-void text-chalk font-body selection:bg-amber/20 cursor-fancy flex flex-col relative">

                <CustomCursor />
                <Header />

                <DetailsModal
                    isOpen={isModalOpen}
                    onClose={() => setIsModalOpen(false)}
                    {...modalData}
                />

                <main className="flex-grow flex flex-col items-center max-w-7xl mx-auto w-full px-4">
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
                        <Route path="/signals" element={<Signals data={data} />} />
                        <Route path="/analytics" element={<Analytics simState={simState} />} />
                        <Route path="/settings" element={<Settings />} />
                    </Routes>
                </main>

                <Footer />
            </div>
        </Router>
    )
}

export default App
