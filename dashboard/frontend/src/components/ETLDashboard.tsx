import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:8001/api';

interface Dag {
    dag_id: string;
    description: string;
    is_paused: boolean;
    fileloc: string;
}

interface DagRun {
    dag_id: string;
    run_id: string;
    state: string;
    execution_date: string;
}

const ETLDashboard: React.FC = () => {
    const [dags, setDags] = useState<Dag[]>([]);
    const [selectedDag, setSelectedDag] = useState<string>('');
    const [dagRuns, setDagRuns] = useState<DagRun[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchDags();
    }, []);

    useEffect(() => {
        if (selectedDag) {
            fetchDagRuns(selectedDag);
        }
    }, [selectedDag]);

    const fetchDags = async () => {
        setLoading(true);
        try {
            const response = await axios.get(`${API_BASE}/etl/dags`);
            setDags(response.data.dags || []);
        } catch (error) {
            console.error('Error fetching DAGs:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchDagRuns = async (dagId: string) => {
        try {
            const response = await axios.get(`${API_BASE}/etl/dags/${dagId}/runs`);
            setDagRuns(response.data.dag_runs || []);
        } catch (error) {
            console.error('Error fetching DAG runs:', error);
        }
    };

    const triggerDag = async (dagId: string) => {
        try {
            await axios.post(`${API_BASE}/etl/dags/${dagId}/trigger`);
            alert(`DAG ${dagId} triggered successfully`);
            fetchDagRuns(dagId);
        } catch (error) {
            console.error('Error triggering DAG:', error);
            alert('Failed to trigger DAG');
        }
    };

    const getStateColor = (state: string) => {
        switch (state?.toLowerCase()) {
            case 'success': return 'text-green-600';
            case 'failed': return 'text-red-600';
            case 'running': return 'text-blue-600';
            default: return 'text-gray-600';
        }
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div className="p-6">
            <h2 className="text-2xl font-bold mb-6">ETL Dashboard</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold mb-4">DAGs</h3>
                    <div className="space-y-2">
                        {dags.map((dag) => (
                            <div key={dag.dag_id} className="border p-3 rounded">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <h4 className="font-medium">{dag.dag_id}</h4>
                                        <p className="text-sm text-gray-600">{dag.description}</p>
                                        <span className={`text-sm ${dag.is_paused ? 'text-red-500' : 'text-green-500'}`}>
                                            {dag.is_paused ? 'Paused' : 'Active'}
                                        </span>
                                    </div>
                                    <div className="space-x-2">
                                        <button
                                            onClick={() => setSelectedDag(dag.dag_id)}
                                            className="bg-blue-500 text-white px-3 py-1 rounded text-sm"
                                        >
                                            View Runs
                                        </button>
                                        <button
                                            onClick={() => triggerDag(dag.dag_id)}
                                            className="bg-green-500 text-white px-3 py-1 rounded text-sm"
                                        >
                                            Trigger
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {selectedDag && (
                    <div className="bg-white p-4 rounded shadow">
                        <h3 className="text-lg font-semibold mb-4">Runs for {selectedDag}</h3>
                        <div className="space-y-2 max-h-96 overflow-y-auto">
                            {dagRuns.map((run) => (
                                <div key={run.run_id} className="border p-3 rounded">
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <p className="font-medium">{run.run_id}</p>
                                            <p className="text-sm text-gray-600">{run.execution_date}</p>
                                        </div>
                                        <span className={`font-medium ${getStateColor(run.state)}`}>
                                            {run.state}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ETLDashboard;