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
    encounter: string;
    code: string;
    description: string;
}

interface Medication {
    patient: string;
    encounter: string;
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

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [patientsRes, conditionsRes, medicationsRes, trendsRes] = await Promise.all([
                axios.get(`${API_BASE}/data/synthea/patients`),
                axios.get(`${API_BASE}/data/synthea/conditions`),
                axios.get(`${API_BASE}/data/synthea/medications`),
                axios.get(`${API_BASE}/data/trends?table=openfda_events&metric_col=serious&group_by=patientsex&limit=100`)
            ]);

            setPatients(patientsRes.data || []);
            setConditions(conditionsRes.data || []);
            setMedications(medicationsRes.data || []);
            setTrends(Array.isArray(trendsRes.data) ? trendsRes.data : []);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    const renderTrendsChart = () => {
        if (!Array.isArray(trends) || trends.length === 0) return null;

        const x = trends.map(d => d.date);
        const y = trends.map(d => d.avg_value);

        const data = [{
            x,
            y,
            mode: 'markers' as const,
            type: 'scatter' as const,
            marker: {
                color: trends.map(d => d.group_key === 1 ? 'blue' : d.group_key === 0 ? 'red' : 'green')
            }
        }];

        return (
            <Plot
                data={data}
                layout={{}}
                style={{ width: '100%', height: '400px' }}
            />
        );
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div className="p-6">
            <h2 className="text-2xl font-bold mb-6">Data Dashboard</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Patients</h3>
                    <p className="text-2xl">{Array.isArray(patients) ? patients.length : 0}</p>
                </div>
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Conditions</h3>
                    <p className="text-2xl">{Array.isArray(conditions) ? conditions.length : 0}</p>
                </div>
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold">Medications</h3>
                    <p className="text-2xl">{Array.isArray(medications) ? medications.length : 0}</p>
                </div>
            </div>

            <div className="bg-white p-4 rounded shadow mb-6">
                <h3 className="text-lg font-semibold mb-4">Trends Visualization</h3>
                {renderTrendsChart()}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold mb-4">Recent Patients</h3>
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead>
                                <tr>
                                    <th className="text-left">ID</th>
                                    <th className="text-left">Name</th>
                                    <th className="text-left">Birthdate</th>
                                    <th className="text-left">Gender</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Array.isArray(patients) && patients.slice(0, 10).map((patient) => (
                                    <tr key={patient.id}>
                                        <td>{patient.id}</td>
                                        <td>{patient.first} {patient.last}</td>
                                        <td>{patient.birthdate}</td>
                                        <td>{patient.gender}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div className="bg-white p-4 rounded shadow">
                    <h3 className="text-lg font-semibold mb-4">Recent Conditions</h3>
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead>
                                <tr>
                                    <th className="text-left">Patient</th>
                                    <th className="text-left">Code</th>
                                    <th className="text-left">Description</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Array.isArray(conditions) && conditions.slice(0, 10).map((condition, idx) => (
                                    <tr key={idx}>
                                        <td>{condition.patient}</td>
                                        <td>{condition.code}</td>
                                        <td>{condition.description}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DataDashboard;