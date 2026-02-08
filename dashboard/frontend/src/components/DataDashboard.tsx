import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const API_BASE = 'http://localhost:8001/api';

interface Patient {
    id: string;
    first?: string;
    last?: string;
    birthdate?: string;
    gender?: string;
    birthDate?: string;
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

interface OpenFDAEvent {
    safetyreportid: string;
    receivedate: string;
    serious: number;
    patientonsetage: number | string;
    patientsex: string;
    reaction: string;
    medicinalproduct: string;
    drugauthorizationnumb: string;
    brand_name: string;
    manufacturer_name: string;
    product_ndc: string;
}

interface UniquePatient {
    patientonsetage: number | string;
    patientsex: number | string;
    event_count: number;
    serious_count: number;
    first_event: string | number;
    last_event: string | number;
}

const DataDashboard: React.FC = () => {
    const [patients, setPatients] = useState<Patient[]>([]);
    const [uniquePatients, setUniquePatients] = useState<UniquePatient[]>([]);
    const [conditions, setConditions] = useState<Condition[]>([]);
    const [medications, setMedications] = useState<Medication[]>([]);
    const [openfdaEvents, setOpenfdaEvents] = useState<OpenFDAEvent[]>([]);
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
            const [patientsRes, conditionsRes, medicationsRes, trendsRes, openfdaRes, postgresRes] = await Promise.all([
                axios.get(`${API_BASE}/data/synthea/patients`),
                axios.get(`${API_BASE}/data/synthea/conditions`),
                axios.get(`${API_BASE}/data/synthea/medications`),
                axios.get(`${API_BASE}/data/trends?table=openfda_events&metric_col=serious&group_by=patientsex&limit=100`),
                axios.get(`${API_BASE}/data/openfda/events?size=50`),
                axios.get(`${API_BASE}/data/openfda/patients?limit=50`),
            ]);

            console.log('Synthea Patients:', patientsRes.data?.length);
            console.log('Conditions:', conditionsRes.data?.length);
            console.log('Medications:', medicationsRes.data?.length);
            console.log('Trends:', trendsRes.data?.length);
            console.log('OpenFDA Events:', openfdaRes.data?.length);
            console.log('Postgres Patients:', postgresRes.data?.length);

            setPatients(patientsRes.data || []);
            setConditions(conditionsRes.data || []);
            setMedications(medicationsRes.data || []);
            setTrends(Array.isArray(trendsRes.data) ? trendsRes.data : []);

            // Handle OpenFDA events
            if (Array.isArray(openfdaRes.data)) {
                setOpenfdaEvents(openfdaRes.data);
            } else if (openfdaRes.data?.error) {
                console.error('OpenFDA API Error:', openfdaRes.data.error);
                setOpenfdaEvents([]);
            } else {
                console.warn('OpenFDA: Unexpected response format');
                setOpenfdaEvents([]);
            }

            if (Array.isArray(postgresRes.data)) {
                setUniquePatients(postgresRes.data);
            } else if (postgresRes.data?.error) {
                console.error('OpenFDA Patients API Error:', postgresRes.data.error);
                setUniquePatients([]);
            } else {
                setUniquePatients([]);
            }
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
    const postgresPatientCount = Array.isArray(uniquePatients) ? uniquePatients.length : 0;
    const openfdaCount = Array.isArray(openfdaEvents) ? openfdaEvents.length : 0;

    return (
        <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="mb-8">
                <h2 className="text-xl font-bold text-white mb-4">Data Overview</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    <StatCard label="Synthea Patients" value={patientCount} color="cyan" />
                    <StatCard label="Unique Patients" value={postgresPatientCount} color="orange" />
                    <StatCard label="Conditions" value={conditionCount} color="purple" />
                    <StatCard label="Medications" value={medicationCount} color="purple" />
                    <StatCard label="OpenFDA Events" value={openfdaCount} color="orange" />
                </div>
            </div>

            {trends.length > 0 && (
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 mb-8">
                    <h3 className="text-lg font-semibold text-white mb-4">Trend Analysis</h3>
                    <TrendsChart data={trends} />
                </div>
            )}

            <div className="space-y-8">
                <div>
                    <div className="flex items-center gap-3 mb-4">
                        <h2 className="text-xl font-bold text-white">OpenFDA Safety Data</h2>
                        <div className="h-3 w-3 bg-orange-500 rounded-full"></div>
                    </div>
                    {openfdaEvents && openfdaEvents.length > 0 ? (
                        <OpenFDAEventsTable events={openfdaEvents.slice(0, 20)} />
                    ) : (
                        <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 text-slate-400">
                            No OpenFDA events data available
                        </div>
                    )}
                </div>

                <div>
                    <div className="flex items-center gap-3 mb-4">
                        <h2 className="text-xl font-bold text-white">Synthea Patient Data</h2>
                        <div className="h-3 w-3 bg-cyan-500 rounded-full"></div>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <PatientsTable patients={patients.slice(0, 10)} />
                        <ConditionsTable conditions={conditions.slice(0, 10)} />
                    </div>
                </div>

                <div>
                    <div className="flex items-center gap-3 mb-4">
                        <h2 className="text-xl font-bold text-white">Unique Patients from Events</h2>
                        <div className="h-3 w-3 bg-orange-500 rounded-full"></div>
                    </div>
                    {uniquePatients.length > 0 && (
                        <UniquePatientTable patients={uniquePatients.slice(0, 20)} />
                    )}
                </div>
            </div>
        </div>
    );
};

interface StatCardProps {
    label: string;
    value: number;
    color?: 'cyan' | 'orange' | 'purple';
}

const colorMap = {
    cyan: { text: 'text-cyan-400', border: 'border-cyan-500/30' },
    orange: { text: 'text-orange-400', border: 'border-orange-500/30' },
    purple: { text: 'text-purple-400', border: 'border-purple-500/30' },
};

const StatCard: React.FC<StatCardProps> = ({ label, value, color = 'cyan' }) => {
    const colors = colorMap[color];
    return (
        <div className={`bg-slate-800 rounded-lg p-4 border border-slate-700 ${colors.border}`}>
            <p className="text-sm text-slate-400 mb-2">{label}</p>
            <p className={`text-3xl font-bold ${colors.text}`}>{value.toLocaleString()}</p>
        </div>
    );
};

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

interface PatientsTableProps {
    patients: Patient[];
}

const PatientsTable: React.FC<PatientsTableProps> = ({ patients }) => (
    <div className="bg-slate-800 rounded-lg border-2 border-cyan-500/30 overflow-hidden">
        <div className="bg-slate-900 px-6 py-4 border-b border-cyan-500/30">
            <h3 className="font-semibold text-white flex items-center gap-2">
                <span className="w-2 h-2 bg-cyan-500 rounded-full"></span>
                Synthea Patients
            </h3>
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
                            <td className="px-6 py-4 text-slate-200">{patient.first || patient.last ? `${patient.first || ''} ${patient.last || ''}`.trim() : 'N/A'}</td>
                            <td className="px-6 py-4 text-slate-400 capitalize">{patient.gender || 'N/A'}</td>
                            <td className="px-6 py-4 text-slate-400">{patient.birthdate || 'N/A'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);

interface ConditionsTableProps {
    conditions: Condition[];
}

const ConditionsTable: React.FC<ConditionsTableProps> = ({ conditions }) => (
    <div className="bg-slate-800 rounded-lg border-2 border-purple-500/30 overflow-hidden">
        <div className="bg-slate-900 px-6 py-4 border-b border-purple-500/30">
            <h3 className="font-semibold text-white flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                Recent Conditions
            </h3>
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
                    {conditions.map((condition, idx) => (
                        <tr key={`${condition.code}-${idx}`} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                            <td className="px-6 py-4 text-cyan-300 font-mono text-xs">{condition.code}</td>
                            <td className="px-6 py-4 text-slate-200">{condition.description}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);

interface UniquePatientTableProps {
    patients: UniquePatient[];
}

const UniquePatientTable: React.FC<UniquePatientTableProps> = ({ patients }) => {
    const formatDate = (dateStr: string | number) => {
        if (!dateStr) return 'N/A';
        const strDate = String(dateStr);
        if (strDate.length === 8) {
            return `${strDate.substring(0, 4)}-${strDate.substring(4, 6)}-${strDate.substring(6, 8)}`;
        }
        return strDate;
    };

    const formatSex = (sex: number | string) => {
        const s = String(sex);
        return s === '1' ? 'M' : s === '2' ? 'F' : s;
    };

    return (
        <div className="bg-slate-800 rounded-lg border-2 border-orange-500/30 overflow-hidden">
            <div className="bg-slate-900 px-6 py-4 border-b border-orange-500/30">
                <h3 className="font-semibold text-white flex items-center gap-2">
                    <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                    Patient Demographics from Events
                </h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-slate-700 bg-slate-900/50">
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Age</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Sex</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Total Events</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Serious Events</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">First Report</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Last Report</th>
                        </tr>
                    </thead>
                    <tbody>
                        {patients.map((patient, idx) => (
                            <tr key={`${patient.patientonsetage}-${patient.patientsex}-${idx}`} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                                <td className="px-6 py-4 text-slate-200">{patient.patientonsetage || 'N/A'}</td>
                                <td className="px-6 py-4 text-slate-400 font-semibold">{formatSex(patient.patientsex)}</td>
                                <td className="px-6 py-4 text-slate-200 font-semibold">{patient.event_count}</td>
                                <td className="px-6 py-4">
                                    <span className="px-2 py-1 rounded text-xs font-semibold bg-red-900/40 text-red-300">
                                        {patient.serious_count || 0}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-400 text-xs">{formatDate(patient.first_event)}</td>
                                <td className="px-6 py-4 text-slate-400 text-xs">{formatDate(patient.last_event)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

interface OpenFDAEventsTableProps {
    events: OpenFDAEvent[];
}

const OpenFDAEventsTable: React.FC<OpenFDAEventsTableProps> = ({ events }) => {
    const formatDate = (dateStr: string | number) => {
        if (!dateStr) return 'N/A';
        const strDate = String(dateStr);
        if (strDate.length === 8) {
            // Format YYYYMMDD
            return `${strDate.substring(0, 4)}-${strDate.substring(4, 6)}-${strDate.substring(6, 8)}`;
        }
        return strDate;
    };

    const sortedEvents = [...events].sort((a, b) => {
        const dateA = String(a.receivedate);
        const dateB = String(b.receivedate);
        return dateB.localeCompare(dateA);
    });

    return (
        <div className="bg-slate-800 rounded-lg border-2 border-orange-500/30 overflow-hidden">
            <div className="bg-slate-900 px-6 py-4 border-b border-orange-500/30">
                <h3 className="font-semibold text-white flex items-center gap-2">
                    <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                    Most Recent Safety Events
                </h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-slate-700 bg-slate-900/50">
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Report ID</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Date</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Serious</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Age</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Sex</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Reaction</th>
                            <th className="px-6 py-3 text-left text-xs font-semibold text-slate-300">Product</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sortedEvents.map((event, idx) => (
                            <tr key={`${event.safetyreportid}-${idx}`} className="border-b border-slate-700 hover:bg-slate-700/30 transition-colors">
                                <td className="px-6 py-4 text-cyan-300 font-mono text-xs">{event.safetyreportid}</td>
                                <td className="px-6 py-4 text-slate-400 text-xs">{formatDate(event.receivedate)}</td>
                                <td className="px-6 py-4">
                                    <span className={`px-2 py-1 rounded text-xs font-semibold ${event.serious === 1 ? 'bg-red-900/40 text-red-300' : 'bg-green-900/40 text-green-300'}`}>
                                        {event.serious === 1 ? 'Yes' : 'No'}
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-slate-400">{event.patientonsetage || 'N/A'}</td>
                                <td className="px-6 py-4 text-slate-400 capitalize">{event.patientsex === '1' ? 'M' : event.patientsex === '2' ? 'F' : event.patientsex}</td>
                                <td className="px-6 py-4 text-slate-200 max-w-xs truncate">{event.reaction || 'N/A'}</td>
                                <td className="px-6 py-4 text-slate-300 max-w-xs truncate">{event.medicinalproduct || event.brand_name || 'N/A'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default DataDashboard;