import { useEffect, useState } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CustomCursor } from './components/CustomCursor';
import { ParticleBackground } from './components/ParticleBackground';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { DetailsModal } from './components/DetailsModal';

// Pages
import { Dashboard } from './pages/Dashboard';
import { Signals } from './pages/Signals';
import { Settings } from './pages/Settings';

interface Decision {
    Ticker: string;
    Action: string;
    Confidence: number;
    Rational: string[];
    History?: any[];
}

function App() {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<Decision[]>([]);
    const [marketMood, setMarketMood] = useState('NEUTRAL');
    const [lastUpdated, setLastUpdated] = useState<string>('');
    const [simState, setSimState] = useState<any>(null);
    const [isAuto, setIsAuto] = useState(true); // Default to Auto-Scan

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

    // Real-Time Loop (The "Pulse")
    useEffect(() => {
        let interval: any;
        if (isAuto) {
            interval = setInterval(() => {
                if (!loading) runScan(true); // Silent scan
            }, 2000); // 2 Second Loop
        }
        return () => clearInterval(interval);
    }, [isAuto, loading]);

    const runScan = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            // In production, point to your FastAPI URL
            const res = await axios.get('http://localhost:8000/api/scan');
            if (res.data.status === 'success') {
                setData(res.data.data);
                if (res.data.simulation) setSimState(res.data.simulation); // Update Sim State
                deriveMarketMood(res.data.data);
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
        if (volCount > decisions.length / 2) setMarketMood("HIGH VOLATILITY (FEAR)");
        else setMarketMood("LOW VOLATILITY (COILED)");
    };

    const openDetails = (ticker: string, action: string, rational: string[], confidence: number, history: any[]) => {
        setModalData({ ticker, action, rational, confidence, history });
        setIsModalOpen(true);
    };

    useEffect(() => {
        runScan();
    }, []);

    // Lock Logic: Can only turn off if DEAD (or if not started)
    const isLocked = isAuto && simState?.status !== 'DEAD';

    return (
        <Router>
            <div className="min-h-screen bg-marine text-mist font-sans selection:bg-teal/30 cursor-fancy flex flex-col relative">

                <ParticleBackground />
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
                                    // Prevent stopping if locked
                                    if (isLocked) return;
                                    setIsAuto(!isAuto);
                                }}
                                isAuto={isAuto}
                                isLocked={isLocked} // Pass Lock State
                                onDetails={openDetails}
                                simState={simState}
                                onResetSim={resetSim}
                            />
                        } />
                        <Route path="/signals" element={<Signals data={data} />} />
                        <Route path="/settings" element={<Settings />} />
                    </Routes>
                </main>

                <Footer />
            </div>
        </Router>
    )
}

export default App
