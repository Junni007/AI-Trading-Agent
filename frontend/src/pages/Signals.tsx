export const Signals = ({ data }: { data: any[] }) => {
    return (
        <div className="w-full max-w-7xl mx-auto py-20 px-6">
            <h1 className="font-display text-4xl font-bold text-chalk mb-8 tracking-tight">Active Signals</h1>

            <div className="overflow-x-auto panel">
                <table className="w-full text-left">
                    <thead className="bg-obsidian text-smoke uppercase text-xs tracking-wider border-b border-graphite">
                        <tr>
                            <th className="px-6 py-4 font-semibold">Ticker</th>
                            <th className="px-6 py-4 font-semibold">Action</th>
                            <th className="px-6 py-4 font-semibold">Confidence</th>
                            <th className="px-6 py-4 font-semibold">Reasoning</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-graphite/50 text-sm font-body">
                        {data.length === 0 ? (
                            <tr>
                                <td colSpan={4} className="px-6 py-10 text-center text-smoke">
                                    No active signals. Run a scan first.
                                </td>
                            </tr>
                        ) : (
                            data.map((item) => (
                                <tr key={item.Ticker} className="hover:bg-slate/30 transition-colors">
                                    <td className="px-6 py-4 font-display font-bold text-chalk">{item.Ticker.replace('.NS', '')}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2.5 py-1 rounded-lg text-xs font-semibold ${item.Action === 'WAIT'
                                                ? 'bg-iron/30 text-smoke'
                                                : 'bg-amber/10 text-amber border border-amber/20'
                                            }`}>
                                            {item.Action.replace(/_/g, ' ')}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 font-mono text-amber">
                                        {(item.Confidence * 100).toFixed(0)}%
                                    </td>
                                    <td className="px-6 py-4 text-ash truncate max-w-md">
                                        {item.Rational[item.Rational.length - 1]}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
