import { useState } from 'react';
import Layout from './components/Layout';
import type { TabType } from './components/Layout';
import RecommendationsView from './components/RecommendationsView';
import ListsView from './components/ListsView';
import AboutView from './components/AboutView';
import MethodologyView from './components/MethodologyView';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('recommend');

  const renderContent = () => {
    switch (activeTab) {
      case 'recommend': return <RecommendationsView />;
      case 'lists': return <ListsView />;
      case 'about': return <AboutView />;
      case 'methodology': return <MethodologyView />;
      default: return <RecommendationsView />;
    }
  };

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      {renderContent()}
    </Layout>
  );
}

export default App;
