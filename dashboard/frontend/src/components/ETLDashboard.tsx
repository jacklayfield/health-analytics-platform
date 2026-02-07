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
    const [error, setError] = useState<string | null>(null);
    const [triggering, setTriggering] = useState<string | null>(null);

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
        setError(null);
        try {
            const response = await axios.get(`${API_BASE}/etl/dags`);
            setDags(response.data.dags || []);
        } catch (err) {
            console.error('Error fetching DAGs:', err);
            setError('Failed to load DAGs');
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
            setError('Failed to load DAG runs');
        }
    };

    const triggerDag = async (dagId: string) => {
        setTriggering(dagId);
        try {
            await axios.post(`${API_BASE}/etl/dags/${dagId}/trigger`);
            await new Promise(r => setTimeout(r, 500));
            await fetchDagRuns(dagId);
        } catch (error) {
            console.error('Error triggering DAG:', error);
            setError(`Failed to trigger ${dagId}`);
        } finally {
            setTriggering(null);
        }
    };

    const getStateIcon = (state: string) => {
        switch (state?.toLowerCase()) {
            case 'success':
                return <svg className="w-5 h-5 text-green-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>;
            case 'failed':
                return <svg className="w-5 h-5 text-red-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>;
            case 'running':
                return <svg className="w-5 h-5 text-cyan-400 animate-spin flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="currentColor" viewBox="0 0 20 20"><path d="M4.555 5.659c0 1.456.91 2.734 2.177 3.322a6.046 6.046 0 002.269-.798 1 1 0 00-1.194-1.595A4.048 4.048 0 005.555 5.66a1 1 0 00-.999.001zm10.333 0a1 1 0 10.999-.001 1 1 0 00-.999.001zM10 15a1 1 0 01-1-1v-2a1 1 0 112 0v2a1 1 0 01-1 1z" /></svg>;
            default:
                return <svg className="w-5 h-5 text-slate-400 flex-shrink-0" style={{ width: '20px', height: '20px' }} fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 5v8a2 2 0 01-2 2h-5l-5 4v-4H4a2 2 0 01-2-2V5a2 2 0 012-2h12a2 2 0 012 2zM7 8H5v2h2V8zm2 0h2v2H9V8zm6 0h-2v2h2V8z" clipRule="evenodd" /></svg>;
        }
    };

    const getStateBgColor = (state: string) => {
        switch (state?.toLowerCase()) {
            case 'success': return 'bg-green-500/20 text-green-300';
            case 'failed': return 'bg-red-500/20 text-red-300';
            case 'running': return 'bg-cyan-500/20 text-cyan-300';
            default: return 'bg-slate-500/20 text-slate-300';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-cyan-500/20 mb-3">
                        <svg className="w-5 h-5 text-cyan-400 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                    </div>
                    <p className="text-slate-400">Loading ETL data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {error && (
                <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 flex items-start space-x-3">
                    <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                    <div className="flex-1">
                        <p className="text-red-300">{error}</p>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* DAGs Section */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
                    <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 border-b border-slate-600">
                        <h3 className="text-lg font-semibold text-white">Airflow DAGs</h3>
                        <p className="text-slate-400 text-sm mt-1">Configured data pipelines</p>
                    </div>
                    <div className="p-6 space-y-3 max-h-96 overflow-y-auto">
                        {dags.length === 0 ? (
                            <p className="text-slate-400 text-center py-8">No DAGs found</p>
                        ) : (
                            dags.map((dag) => (
                                <div
                                    key={dag.dag_id}
                                    className="bg-slate-900/50 border border-slate-700 rounded-lg p-4 hover:border-cyan-500/50 transition-all duration-200"
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex-1">
                                            <h4 className="font-semibold text-white">{dag.dag_id}</h4>
                                            <p className="text-sm text-slate-400 mt-1">{dag.description}</p>
                                        </div>
                                        <span
                                            className={`ml-2 inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-medium ${dag.is_paused
                                                ? 'bg-orange-500/20 text-orange-300'
                                                : 'bg-green-500/20 text-green-300'
                                                }`}
                                        >
                                            <span className={`w-2 h-2 rounded-full ${dag.is_paused ? 'bg-orange-400' : 'bg-green-400'}`}></span>
                                            {dag.is_paused ? 'Paused' : 'Active'}
                                        </span>
                                    </div>
                                    <div className="flex space-x-2">
                                        <button
                                            onClick={() => setSelectedDag(dag.dag_id)}
                                            className="flex-1 px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 text-white text-xs font-medium rounded transition-all duration-200 flex items-center justify-center space-x-1"
                                        >
                                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                                            <span>Runs</span>
                                        </button>
                                        <button
                                            onClick={() => triggerDag(dag.dag_id)}
                                            disabled={triggering === dag.dag_id}
                                            className="px-3 py-1.5 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-500 hover:to-emerald-600 disabled:from-slate-600 disabled:to-slate-700 text-white text-xs font-medium rounded transition-all duration-200 flex items-center justify-center space-x-1"
                                        >
                                            {triggering === dag.dag_id ? (
                                                <>
                                                    <svg className="w-3 h-3 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                                                    <span>Running</span>
                                                </>
                                            ) : (
                                                <>
                                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                                    <span>Trigger</span>
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* DAG Runs Section */}
                {selectedDag && (
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-slate-700 overflow-hidden shadow-lg">
                        <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 border-b border-slate-600 flex items-center justify-between">
                            <div>
                                <h3 className="text-lg font-semibold text-white">Runs: {selectedDag}</h3>
                                <p className="text-slate-400 text-sm mt-1">Execution history</p>
                            </div>
                            <button
                                onClick={() => setSelectedDag('')}
                                className="text-slate-400 hover:text-white transition-colors"
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                            </button>
                        </div>
                        <div className="p-6 space-y-3 max-h-96 overflow-y-auto">
                            {dagRuns.length === 0 ? (
                                <p className="text-slate-400 text-center py-8">No runs found</p>
                            ) : (
                                dagRuns.map((run) => (
                                    <div
                                        key={run.run_id}
                                        className="bg-slate-900/50 border border-slate-700 rounded-lg p-4 hover:border-cyan-500/50 transition-all duration-200"
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1 min-w-0">
                                                <p className="font-mono text-sm text-slate-300 truncate">{run.run_id}</p>
                                                <p className="text-xs text-slate-500 mt-1">{new Date(run.execution_date).toLocaleString()}</p>
                                            </div>
                                            <div className={`ml-3 inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${getStateBgColor(run.state)}`}>
                                                {getStateIcon(run.state)}
                                                <span>{run.state}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ETLDashboard;