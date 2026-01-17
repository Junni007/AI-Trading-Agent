export const Settings = () => {
    return (
        <div className="w-full max-w-3xl mx-auto py-20 px-6">
            <h1 className="font-display text-4xl font-bold text-chalk mb-8 tracking-tight">Configuration</h1>

            <div className="panel p-8 space-y-8">

                {/* Universe Selection */}
                <div>
                    <h3 className="font-display text-lg font-bold text-amber mb-4">Universe Selection</h3>
                    <div className="flex gap-3">
                        <button className="btn-primary">Nifty 50 (India)</button>
                        <button className="btn-ghost">US Tech (Nasdaq)</button>
                    </div>
                </div>

                <div className="h-px bg-graphite" />

                {/* Risk Parameters */}
                <div>
                    <h3 className="font-display text-lg font-bold text-amber mb-4">Risk Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="flex flex-col gap-2">
                            <label className="text-sm text-smoke">Min Confidence Threshold</label>
                            <input
                                type="range"
                                className="accent-amber h-2 bg-graphite rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between text-xs text-smoke/60">
                                <span>50%</span>
                                <span>90%</span>
                            </div>
                        </div>
                        <div className="flex flex-col gap-2">
                            <label className="text-sm text-smoke">Max Open Positions</label>
                            <input
                                type="number"
                                className="bg-void border border-graphite rounded-xl py-2.5 px-4 text-chalk font-mono focus:border-amber/50 focus:outline-none transition-colors"
                                placeholder="5"
                            />
                        </div>
                    </div>
                </div>

                <div className="h-px bg-graphite" />

                {/* Warning */}
                <div className="p-4 bg-amber/5 border border-amber/20 rounded-xl text-amber text-sm flex items-start gap-3">
                    <span className="text-lg">⚠</span>
                    <span>Configuration changes require a backend restart in this version (v1.0).</span>
                </div>

            </div>
        </div>
    );
};
