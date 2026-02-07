import { useState } from 'react';
import DataDashboard from './components/DataDashboard';
import './App.css';

type Tab = 'data' | 'etl' | 'ml';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('data');

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-cyan-400">Health Analytics Dashboard</h1>
          <p className="text-sm text-slate-300 mt-2">Real-time healthcare data platform</p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-slate-800 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-2 py-3">
            <button
              onClick={() => setActiveTab('data')}
              className="px-4 py-2 font-medium text-sm transition-all border-b-2 text-cyan-400 border-b-cyan-400 hover:text-cyan-300"
            >
              DATA
            </button>
            <button
              disabled
              className="px-4 py-2 font-medium text-sm transition-all border-b-2 text-slate-500 border-b-transparent opacity-50 cursor-not-allowed"
            >
              ETL <span className="text-xs ml-2">(coming soon)</span>
            </button>
            <button
              disabled
              className="px-4 py-2 font-medium text-sm transition-all border-b-2 text-slate-500 border-b-transparent opacity-50 cursor-not-allowed"
            >
              ML <span className="text-xs ml-2">(coming soon)</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main>
        {activeTab === 'data' && <DataDashboard />}
      </main>
    </div>
  );
}

export default App;
