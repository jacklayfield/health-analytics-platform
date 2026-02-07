import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const API_BASE = 'http://localhost:8001/api';

interface Experiment {
    experiment_id: string;
    name: string;
    artifact_location: string;
}

interface Run {
    info: {
        run_id: string;
        experiment_id: string;
        status: string;
        start_time: number;
        end_time: number;
    };
    data: {
        metrics: { [key: string]: number };
        params: { [key: string]: string };
    };
}

interface ModelVersion {
    name: string;
    version: string;
    creation_timestamp: number;
    last_updated_timestamp: number;
    current_stage: string;
}

interface Model {
    name: string;
    creation_timestamp: number;
    last_updated_timestamp: number;
    versions: ModelVersion[];
}

const MLDashboard: React.FC = () => {
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [latestRun, setLatestRun] = useState<Run | null>(null);
    const [models, setModels] = useState<Model[]>([]);
    const [selectedExperiment, setSelectedExperiment] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchMLData();
    }, []);

    useEffect(() => {
        if (selectedExperiment) {
            fetchLatestRun(selectedExperiment);
        }
    }, [selectedExperiment]);

    const fetchMLData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [experimentsRes, modelsRes] = await Promise.all([
                axios.get(`${API_BASE}/ml/experiments`),
                axios.get(`${API_BASE}/ml/models`)
            ]);

            setExperiments(experimentsRes.data || []);
            setModels(modelsRes.data || []);
        } catch (err) {
            console.error('Error fetching ML data:', err);
            setError('Failed to load ML data');
        } finally {
            setLoading(false);
        }
    };

    const fetchLatestRun = async (experimentName: string) => {
        try {
            const response = await axios.get(`${API_BASE}/ml/experiments/latest?name=${experimentName}`);
            setLatestRun(response.data);
        } catch (error) {
            console.error('Error fetching latest run:', error);
            setLatestRun(null);
        }
    };

    const renderMetricsChart = () => {
        if (!latestRun?.data?.metrics) return null;

        const metrics = latestRun.data.metrics;
        const metricNames = Object.keys(metrics);
        const metricValues = Object.values(metrics);

        return (
            <Plot
                data={[{
                    x: metricNames,
                    y: metricValues,
                    type: 'bar',
                    marker: { color: '#06b6d4' }
                }]}
                layout={{
                    paper_bgcolor: 'rgba(15, 23, 42, 0)',
                    plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
                    font: { color: '#cbd5e1', family: 'system-ui' },
                    margin: { l: 50, r: 50, t: 40, b: 40 },
                    xaxis: {
                        showgrid: true,
                        gridcolor: 'rgba(100, 116, 139, 0.2)',
                        zeroline: false
                    },
                    yaxis: {
                        showgrid: true,
                        gridcolor: 'rgba(100, 116, 139, 0.2)',
                        zeroline: false
                    }
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%', height: '300px' }}
            />
        );
    };

    const StatCard = ({ label, value, icon: Icon }: { label: string; value: number | string; icon: React.ReactNode }) => (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-lg p-4 border border-slate-700 hover:border-cyan-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/10">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-slate-400 text-xs font-medium mb-1">{label}</p>
                    <p className="text-2xl font-bold text-white">{value}</p>
                </div>
                <div className="w-6 h-6 flex-shrink-0 flex items-center justify-center">
                    {Icon}
                </div>
            </div>
        </div>
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-cyan-500/20 mb-3">
                        <svg className="w-5 h-5 text-cyan-400 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                    </div>
                    <p className="text-slate-400">Loading ML data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {error && (
                <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 flex items-start space-x-3">
                    <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                    <p className="text-red-300">{error}</p>
                </div>
            )}

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <StatCard
                    label="Total Experiments"
                    value={experiments.length}
                    icon={<svg className="w-5 h-5 text-cyan-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
                />
                <StatCard
                    label="Registered Models"
                    value={models.length}
                    icon={<svg className="w-5 h-5 text-cyan-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" /></svg>}
                />
                <StatCard
                    label="Latest Run Status"
                    value={latestRun?.info?.status || 'No Runs'}
                    icon={<svg className="w-5 h-5 text-cyan-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>}
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Experiments Section */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
                    <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 border-b border-slate-600">
                        <h3 className="text-lg font-semibold text-white">Experiments</h3>
                        <p className="text-slate-400 text-sm mt-1">Available ML experiments</p>
                    </div>
                    <div className="p-6 space-y-3 max-h-96 overflow-y-auto">
                        {experiments.length === 0 ? (
                            <p className="text-slate-400 text-center py-8">No experiments found</p>
                        ) : (
                            experiments.map((exp) => (
                                <div
                                    key={exp.experiment_id}
                                    className={`bg-slate-900/50 border rounded-lg p-4 cursor-pointer transition-all duration-200 ${selectedExperiment === exp.name
                                        ? 'border-cyan-500 bg-cyan-500/10'
                                        : 'border-slate-700 hover:border-cyan-500/50'
                                        }`}
                                    onClick={() => setSelectedExperiment(exp.name)}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <h4 className="font-semibold text-white">{exp.name}</h4>
                                            <p className="text-xs text-slate-500 mt-1 font-mono">ID: {exp.experiment_id}</p>
                                        </div>
                                        <svg className={`w-5 h-5 transition-transform ${selectedExperiment === exp.name ? 'text-cyan-400' : 'text-slate-500'}`} fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 10 10.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Registered Models Section */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
                    <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 border-b border-slate-600">
                        <h3 className="text-lg font-semibold text-white">Registered Models</h3>
                        <p className="text-slate-400 text-sm mt-1">Production-ready models</p>
                    </div>
                    <div className="p-6 space-y-3 max-h-96 overflow-y-auto">
                        {models.length === 0 ? (
                            <p className="text-slate-400 text-center py-8">No models found</p>
                        ) : (
                            models.map((model) => (
                                <div
                                    key={model.name}
                                    className="bg-slate-900/50 border border-slate-700 rounded-lg p-4 hover:border-cyan-500/50 transition-all duration-200"
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <h4 className="font-semibold text-white">{model.name}</h4>
                                        <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-300">
                                            {model.versions?.length || 0} versions
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-500">
                                        Created: {new Date(model.creation_timestamp).toLocaleDateString()}
                                    </p>
                                    <p className="text-xs text-slate-500">
                                        Updated: {new Date(model.last_updated_timestamp).toLocaleDateString()}
                                    </p>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Latest Run Details */}
            {latestRun && (
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
                    <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 border-b border-slate-600">
                        <h3 className="text-lg font-semibold text-white">Latest Run Details</h3>
                        <p className="text-slate-400 text-sm mt-1">{selectedExperiment || 'Most recent'} experiment</p>
                    </div>
                    <div className="p-6 space-y-6">
                        {/* Run Info Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                                <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Run Information</h4>
                                <div className="space-y-2">
                                    <div>
                                        <p className="text-xs text-slate-500">Run ID</p>
                                        <p className="text-sm text-white font-mono">{latestRun.info.run_id.substring(0, 12)}...</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Status</p>
                                        <div className="flex items-center space-x-2 mt-1">
                                            <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${latestRun.info.status === 'FINISHED'
                                                ? 'bg-green-500/20 text-green-300'
                                                : 'bg-yellow-500/20 text-yellow-300'
                                                }`}>
                                                {latestRun.info.status}
                                            </span>
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Start Time</p>
                                        <p className="text-sm text-white">{new Date(latestRun.info.start_time).toLocaleString()}</p>
                                    </div>
                                    {latestRun.info.end_time && (
                                        <div>
                                            <p className="text-xs text-slate-500">End Time</p>
                                            <p className="text-sm text-white">{new Date(latestRun.info.end_time).toLocaleString()}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                                <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Parameters</h4>
                                <div className="space-y-2 max-h-48 overflow-y-auto">
                                    {latestRun.data.params && Object.entries(latestRun.data.params).length > 0 ? (
                                        Object.entries(latestRun.data.params).map(([key, value]) => (
                                            <div key={key}>
                                                <p className="text-xs text-slate-500">{key}</p>
                                                <p className="text-sm text-white font-mono break-all">{value}</p>
                                            </div>
                                        ))
                                    ) : (
                                        <p className="text-slate-400 text-sm">No parameters recorded</p>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Metrics Chart */}
                        {latestRun.data.metrics && Object.keys(latestRun.data.metrics).length > 0 && (
                            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                                <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">Performance Metrics</h4>
                                {renderMetricsChart()}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default MLDashboard;