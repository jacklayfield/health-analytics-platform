import { useState } from 'react';
import DataDashboard from './components/DataDashboard';
import ETLDashboard from './components/ETLDashboard';
import MLDashboard from './components/MLDashboard';
import './App.css';

type Tab = 'data' | 'etl' | 'ml';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('data');

  const renderTabContent = () => {
    switch (activeTab) {
      case 'data':
        return <DataDashboard />;
      case 'etl':
        return <ETLDashboard />;
      case 'ml':
        return <MLDashboard />;
      default:
        return <DataDashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">Health Analytics Dashboard</h1>
          </div>
        </div>
      </header>

      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab('data')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'data'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              Data
            </button>
            <button
              onClick={() => setActiveTab('etl')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'etl'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              ETL
            </button>
            <button
              onClick={() => setActiveTab('ml')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'ml'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
            >
              ML
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {renderTabContent()}
      </main>
    </div>
  );
}

export default App;
