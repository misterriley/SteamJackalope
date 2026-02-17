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
    // Also clear profile data from persisted filters
    const saved = sessionStorage.getItem('recommendations_filters');
    if (saved) {
      try {
        const filters = JSON.parse(saved);
        const { vibe_vector, intercept, metadata_weights, library_appids, rated_appids, ...rest } = filters;
        sessionStorage.setItem('recommendations_filters', JSON.stringify({
          ...rest,
          profile_filter: 'none',
          library_appids: [],
          rated_appids: []
        }));
      } catch (e) {}
    }
  }, []);

  const handleApplyProfile = (profile: any) => {
    console.log("Applying Profile:", profile);
    if (!profile) return;

    try {
      // 1. Get current filters
      const savedFiltersStr = sessionStorage.getItem('recommendations_filters');
      let filters = savedFiltersStr ? JSON.parse(savedFiltersStr) : {};
      
      // 2. Extract metadata and vibe
      const metadata = profile.metadata || {};
      const vibeVector = (profile.vibe_vector || []).map((v: any) => v || 0);

      const newFilters = {
        ...filters,
        // RESET interfering filters to match solver's "clean" environment
        genres: [],
        seed_games: [],
        prompt: '',
        
        // Match solver's default filtering assumptions
        english_only: true,
        remove_vr: true,
        remove_nsfw: true,
        remove_utilities: true,
        remove_unreleased: true,

        // Direct Translation: Sliders = Absolute Weights
        quality_pref: metadata.quality ?? 0,
        age_pref: metadata.age ?? 0,
        pop_pref: metadata.popularity ?? 0,
        length_pref: metadata.length ?? 0,
        difficulty_pref: metadata.difficulty ?? 0,
        
        vibe_vector: vibeVector,
        intercept: profile.intercept || 0,
        metadata_weights: metadata,
        
        alpha: metadata.semantic ?? 1.0,
        beta: metadata.tag_match ?? 1.0,
        disc_pref: metadata.discovery ?? 0,
        
        library_appids: profile.library_appids || [],
        rated_appids: profile.rated_appids || []
      };

      console.log("New Filters Prepared:", newFilters);

      // 4. Persist to session storage so RecommendationsView picks it up on mount
      sessionStorage.setItem('recommendations_filters', JSON.stringify(newFilters));
      
      // 5. Update global state and switch tab
      setAppliedProfile(profile);
      setActiveTab('recommend');
    } catch (err) {
      console.error("Failed to apply taste profile:", err);
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
