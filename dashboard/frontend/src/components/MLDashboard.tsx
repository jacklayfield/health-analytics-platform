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
        try {
            const [experimentsRes, modelsRes] = await Promise.all([
                axios.get(`${API_BASE}/ml/experiments`),
                axios.get(`${API_BASE}/ml/models`)
            ]);

            setExperiments(experimentsRes.data || []);
            setModels(modelsRes.data || []);
        } catch (error) {
            console.error('Error fetching ML data:', error);
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
                    marker: { color: 'rgba(75, 192, 192, 0.6)' }
                }]}
                layout={{
                    title: { text: 'Latest Run Metrics' },
                    xaxis: { title: { text: 'Metric' } },
                    yaxis: { title: { text: 'Value' } }
                }}
                style={{ width: '100%', height: '300px' }}
            />
        );
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div className="p-6">
            <h2 className="text-2xl font-bold mb-6">ML Dashboard</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Experiments</h3>
                    <p className="text-2xl">{experiments.length}</p>
                </div>
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Models</h3>
                    <p className="text-2xl">{models.length}</p>
                </div>
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Latest Run Status</h3>
                    <p className={`text-lg ${latestRun?.info?.status === 'FINISHED' ? 'text-green-600' : 'text-yellow-600'}`}>
                        {latestRun?.info?.status || 'No runs'}
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold mb-4">Experiments</h3>
                    <div className="space-y-2">
                        {experiments.map((exp) => (
                            <div key={exp.experiment_id} className="border p-3 rounded">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <h4 className="font-medium">{exp.name}</h4>
                                        <p className="text-sm text-gray-600">ID: {exp.experiment_id}</p>
                                    </div>
                                    <button
                                        onClick={() => setSelectedExperiment(exp.name)}
                                        className="bg-blue-500 text-white px-3 py-1 rounded text-sm"
                                    >
                                        View Latest
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold mb-4">Registered Models</h3>
                    <div className="space-y-2">
                        {models.map((model) => (
                            <div key={model.name} className="border p-3 rounded">
                                <h4 className="font-medium">{model.name}</h4>
                                <p className="text-sm text-gray-600">
                                    Created: {new Date(model.creation_timestamp).toLocaleDateString()}
                                </p>
                                <p className="text-sm text-gray-600">
                                    Versions: {model.versions?.length || 0}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {latestRun && (
                <div className="bg-white p-4 rounded shadow mb-6">
                    <h3 className="text-lg font-semibold mb-4">Latest Run Details</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div>
                            <h4 className="font-medium">Run Info</h4>
                            <p>Run ID: {latestRun.info.run_id}</p>
                            <p>Status: <span className={latestRun.info.status === 'FINISHED' ? 'text-green-600' : 'text-yellow-600'}>{latestRun.info.status}</span></p>
                            <p>Start Time: {new Date(latestRun.info.start_time).toLocaleString()}</p>
                            {latestRun.info.end_time && (
                                <p>End Time: {new Date(latestRun.info.end_time).toLocaleString()}</p>
                            )}
                        </div>
                        <div>
                            <h4 className="font-medium">Parameters</h4>
                            {latestRun.data.params && Object.entries(latestRun.data.params).map(([key, value]) => (
                                <p key={key}>{key}: {value}</p>
                            ))}
                        </div>
                    </div>
                    {renderMetricsChart()}
                </div>
            )}
        </div>
    );
};

export default MLDashboard;