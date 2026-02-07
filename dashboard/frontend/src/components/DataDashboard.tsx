import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const API_BASE = 'http://localhost:8001/api';

interface Patient {
    id: string;
    first: string;
    last: string;
    birthdate: string;
    gender: string;
}

interface Condition {
    patient: string;
    code: string;
    description: string;
}

interface Medication {
    patient: string;
    code: string;
    description: string;
}

interface TrendData {
    group_key: number;
    date: string;
    avg_value: number;
}

const DataDashboard: React.FC = () => {
    const [patients, setPatients] = useState<Patient[]>([]);
    const [conditions, setConditions] = useState<Condition[]>([]);
    const [medications, setMedications] = useState<Medication[]>([]);
    const [trends, setTrends] = useState<TrendData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [patientsRes, conditionsRes, medicationsRes, trendsRes] = await Promise.all([
                axios.get(`${API_BASE}/data/synthea/patients`),
                axios.get(`${API_BASE}/data/synthea/conditions`),
                axios.get(`${API_BASE}/data/synthea/medications`),
                axios.get(`${API_BASE}/data/trends?table=openfda_events&metric_col=serious&group_by=patientsex&limit=100`),
            ]);

            setPatients(patientsRes.data || []);
            setConditions(conditionsRes.data || []);
            setMedications(medicationsRes.data || []);
            setTrends(Array.isArray(trendsRes.data) ? trendsRes.data : []);
        } catch (err) {
            console.error('Error fetching data:', err);
            setError('Failed to load data. Please try again later.');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="w-full flex items-center justify-center py-16">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400 mb-4"></div>
                    <p className="text-slate-400">Loading data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
                    <p className="text-red-300 mb-4">{error}</p>
                    <button
                        onClick={fetchData}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors font-semibold"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    const patientCount = Array.isArray(patients) ? patients.length : 0;
    const conditionCount = Array.isArray(conditions) ? conditions.length : 0;
    const medicationCount = Array.isArray(medications) ? medications.length : 0;

    return (
        <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                <StatCard label="Total Patients" value={patientCount} />
                <StatCard label="Total Conditions" value={conditionCount} />
                <StatCard label="Total Medications" value={medicationCount} />
            </div>

            {/* Trends Chart */}
            {trends.length > 0 && (
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 mb-8">
                    <h3 className="text-lg font-semibold text-white mb-4">Trend Analysis</h3>
                    <TrendsChart data={trends} />
                </div>
            )}

            {/* Data Tables */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <PatientsTable patients={patients.slice(0, 10)} />
                <ConditionsTable conditions={conditions.slice(0, 10)} />
            </div>
        </div>
    );
};

// Stat Card Component
interface StatCardProps {
    label: string;
    value: number;
}

const StatCard: React.FC<StatCardProps> = ({ label, value }) => (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
        <p className="text-sm text-slate-400 mb-2">{label}</p>
        <p className="text-3xl font-bold text-cyan-400">{value.toLocaleString()}</p>
    </div>
);

// Trends Chart Component
interface TrendsChartProps {
    data: TrendData[];
}

const TrendsChart: React.FC<TrendsChartProps> = ({ data }) => {
    const x = data.map(d => d.date);
    const y = data.map(d => d.avg_value);

    return (
        <Plot
            data={[
                {
                    x,
                    y,
                    mode: 'lines+markers' as const,
                    type: 'scatter' as const,
                    line: { color: '#06b6d4', width: 2 },
                    marker: { size: 6, color: '#06b6d4' },
                },
            ]}
            layout={{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'rgba(15, 23, 42, 0.5)',
                font: { color: '#cbd5e1', family: 'sans-serif' },
                margin: { l: 50, r: 50, t: 0, b: 40 },
                xaxis: { showgrid: false, zeroline: false },
                yaxis: { showgrid: true, gridcolor: 'rgba(100, 116, 139, 0.1)', zeroline: false },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: '300px' }}
        />
    );
};

// Patients Table Component
interface PatientsTableProps {
    patients: Patient[];
}

const PatientsTable: React.FC<PatientsTableProps> = ({ patients }) => (
    <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        <div className="bg-slate-900 px-6 py-4 border-b border-slate-700">
            <h3 className="font-semibold text-white">Recent Patients</h3>
        </div>
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-slate-700 bg-slate-900/50">
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Name</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Gender</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Birth Date</th>
                    </tr>
                </thead>
                <tbody>
                    {patients.map((patient) => (
                        <tr key={patient.id} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                            <td className="px-6 py-4 text-slate-200">{patient.first} {patient.last}</td>
                            <td className="px-6 py-4 text-slate-400">{patient.gender}</td>
                            <td className="px-6 py-4 text-slate-400">{patient.birthdate}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);

// Conditions Table Component
interface ConditionsTableProps {
    conditions: Condition[];
}

const ConditionsTable: React.FC<ConditionsTableProps> = ({ conditions }) => (
    <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        <div className="bg-slate-900 px-6 py-4 border-b border-slate-700">
            <h3 className="font-semibold text-white">Recent Conditions</h3>
        </div>
        <div className="overflow-x-auto">
            <table className="w-full text-sm">
                <thead>
                    <tr className="border-b border-slate-700 bg-slate-900/50">
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Code</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Description</th>
                    </tr>
                </thead>
                <tbody>
                    {conditions.map((condition) => (
                        <tr key={condition.code} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                            <td className="px-6 py-4 text-cyan-300 font-mono text-xs">{condition.code}</td>
                            <td className="px-6 py-4 text-slate-200">{condition.description}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);

export default DataDashboard;