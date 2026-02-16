import { useState, useEffect, useCallback } from 'react';
import Layout from './components/Layout';
import type { TabType } from './components/Layout';
import RecommendationsView from './components/RecommendationsView';
import ListsView from './components/ListsView';
import PersonalizationView from './components/PersonalizationView';
import AboutView from './components/AboutView';
import MethodologyView from './components/MethodologyView';
import ChangelogView from './components/ChangelogView';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>(() => {
    try {
      return (sessionStorage.getItem('activeTab') as TabType) || 'recommend';
    } catch (e) {
      return 'recommend';
    }
  });
  const [appliedProfile, setAppliedProfile] = useState<any>(() => {
    try {
      const saved = localStorage.getItem('appliedProfile');
      return saved ? JSON.parse(saved) : null;
    } catch (e) {
      console.error("Failed to parse appliedProfile from localStorage", e);
      return null;
    }
  });

  useEffect(() => {
    sessionStorage.setItem('activeTab', activeTab);
  }, [activeTab]);

  useEffect(() => {
    if (appliedProfile) {
      localStorage.setItem('appliedProfile', JSON.stringify(appliedProfile));
    } else {
      localStorage.removeItem('appliedProfile');
    }
  }, [appliedProfile]);

  const handleProfileClear = useCallback(() => {
    setAppliedProfile(null);
  }, []);

  const handleApplyProfile = (profile: any) => {
    console.log("Applying Profile:", profile);
    if (!profile) return;

    try {
      // 1. Pre-calculate filter positions from profile weights
      const savedFiltersStr = sessionStorage.getItem('recommendations_filters');
      let filters = savedFiltersStr ? JSON.parse(savedFiltersStr) : {};
      
      // 2. Safely extract metadata and vibe
      const metadata = profile.metadata || {};
      const vibeVector = (profile.vibe_vector || []).map((v: any) => v || 0);
      const squaredSum = vibeVector.reduce((acc: number, val: number) => acc + (val * val || 0), 0);
      let vibeNorm = Math.sqrt(squaredSum);
      
      if (isNaN(vibeNorm)) vibeNorm = 0;

      console.log("Calculated Beta Weight (Vibe Norm):", vibeNorm);

      const newFilters = {
        ...filters,
        // In Linear Mode, sliders act as multipliers on the solved weights.
        // 0.5 = 100% of solved weight, 0.0 = 0%, 1.0 = 200%.
        quality_pref: 0.5,
        age_pref: 0.5,
        pop_pref: 0.5,
        length_pref: 0.5,
        difficulty_pref: 0.5,
        
        vibe_vector: vibeVector,
        intercept: profile.intercept || 0,
        metadata_weights: metadata,
        
        alpha: 1.0,
        beta: 1.0 
      };

      console.log("New Filters Prepared:", newFilters);

      // 4. Persist to session storage so RecommendationsView picks it up on mount
      sessionStorage.setItem('recommendations_filters', JSON.stringify(newFilters));
      
      // 5. Update global state and switch tab
      setAppliedProfile(profile);
      setActiveTab('recommend');
    } catch (err) {
      console.error("Failed to apply profile:", err);
      alert("Failed to apply taste profile. See console for details.");
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'recommend': return (
        <RecommendationsView 
          onProfileClear={handleProfileClear} 
        />
      );
      case 'lists': return <ListsView />;
      case 'personalize': return (
        <PersonalizationView 
          onApply={handleApplyProfile} 
        />
      );
      case 'about': return <AboutView />;
      case 'methodology': return <MethodologyView />;
      case 'changelog': return <ChangelogView />;
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
