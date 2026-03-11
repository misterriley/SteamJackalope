import { useState, useEffect, useCallback } from 'react';
import Layout from './components/Layout';
import type { TabType } from './components/Layout';
import RecommendationsView from './components/RecommendationsView';
import ListsView from './components/ListsView';
import CatalogueView from './components/CatalogueView';
import PersonalizationView from './components/PersonalizationView';
import AboutView from './components/AboutView';
import MethodologyView from './components/MethodologyView';
import ChangelogView from './components/ChangelogView';
import SplashView from './components/SplashView';
import { useUser } from './context/UserContext';

import InteractiveRankingsView from './components/InteractiveRankingsView';

function App() {
  const { setSteamId: globalSetSteamId } = useUser();
  const [activeTab, setActiveTab] = useState<TabType>(() => {
    try {
      return (sessionStorage.getItem('activeTab') as TabType) || 'splash';
    } catch (e) {
      return 'splash';
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
      try {
        // Remove the massive interactive_pool before saving to localStorage to prevent QuotaExceededError
        const { interactive_pool, ...profileToSave } = appliedProfile;
        localStorage.setItem('appliedProfile', JSON.stringify(profileToSave));
      } catch (e) {
        console.error("Failed to save appliedProfile to localStorage", e);
      }
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
        const { 
          vibe_vector, 
          semantic_vibe_vector, 
          topic_vibe_vector,
          gamma_topic,
          intercept, 
          metadata_weights, 
          library_appids, 
          rated_appids, 
          ignored_appids, 
          ...rest 
        } = filters;
        sessionStorage.setItem('recommendations_filters', JSON.stringify({
          ...rest,
          profile_filter: 'none',
          library_appids: [],
          rated_appids: [],
          ignored_appids: []
        }));
      } catch (e) {}
    }
  }, []);

  const handleApplyProfile = (profile: any) => {
    console.log("!!! APP: Applying Profile START !!!", profile);
    if (!profile) {
      console.error("Apply Profile called with null profile");
      return;
    }

    try {
      // 1. Get current filters or defaults
      const savedFiltersStr = sessionStorage.getItem('recommendations_filters');
      const defaults = {
        alpha: 0.25,
        beta: 1.5,
        quality_pref: 1.0,
        age_pref: 0.0,
        pop_pref: 0.0,
        disc_pref: 0.0,
        length_pref: 0.0,
        difficulty_pref: 0.0,
        price_pref: 0.0,
        remove_vr: true,
        english_only: true,
        remove_nsfw: true,
        remove_utilities: true,
        remove_unreleased: true,
        remove_delisted: true,
        top_k: 30,
        prompt: '',
        seed_games: [],
        genres: [],
        tags: [],
        profile_filter: 'none',
        library_appids: [],
        rated_appids: [],
        ignored_appids: []
      };
      
      let currentFilters = savedFiltersStr ? { ...defaults, ...JSON.parse(savedFiltersStr) } : defaults;
      
      // 2. Extract metadata and vibe
      const meta = profile.metadata || {};
      const vibeVector = (profile.vibe_vector || []).map((v: any) => v || 0);
      const semVibeVector = (profile.semantic_vibe_vector || []).map((v: any) => v || 0);
      const topicVibeVector = (profile.topic_vibe_vector || []).map((v: any) => v || 0);

      console.log("!!! APP: Metadata Keys !!!", Object.keys(meta));
      console.log("!!! APP: Metadata Values !!!", meta);

      const newFilters = {
        ...currentFilters,
        // RESET interfering filters to match solver's "clean" environment
        genres: [],
        tags: [],
        seed_games: [],
        prompt: '',
        
        // Match solver's default filtering assumptions
        english_only: true,
        remove_vr: true,
        remove_nsfw: true,
        remove_utilities: true,
        remove_unreleased: true,
        remove_delisted: true,

        // Direct Mapping: Sliders now show the ABSOLUTE solved weights
        quality_pref: meta.quality ?? 1.0,
        age_pref: meta.age ?? 0.0,
        pop_pref: meta.popularity ?? 0.0,
        length_pref: meta.length ?? 0.0,
        difficulty_pref: meta.difficulty ?? 0.0,
        price_pref: meta.price ?? 0.0,
        alpha: meta.semantic ?? 1.0,
        beta: meta.tag_match ?? 1.0,
        gamma_topic: meta.topic_match ?? profile.gamma_topic ?? 0.5,
        
        // State
        vibe_vector: vibeVector,
        semantic_vibe_vector: semVibeVector,
        topic_vibe_vector: topicVibeVector,
        metadata_weights: meta,
        intercept: profile.intercept ?? 5.0, 
        scaling_factor: profile.scaling_factor ?? 1.0,
        disc_pref: meta.discovery ?? 0,
        
        profile_filter: 'all',
        library_appids: profile.library_appids || [],
        rated_appids: profile.rated_appids || [],
        ignored_appids: profile.ignored_appids || [],
        library_details: profile.library_details || {}
      };

      console.log("!!! APP: Final Filter Payload !!!", newFilters);

      // 4. Persist to session storage so RecommendationsView picks it up on mount
      sessionStorage.setItem('recommendations_filters', JSON.stringify(newFilters));
      
      // 5. Update global state and switch tab
      if (profile.steam_id) {
        globalSetSteamId(profile.steam_id);
      }
      setAppliedProfile(profile);
      setActiveTab('recommend');
      console.log("!!! APP: State Updated and Tab Switched !!!");
    } catch (err) {
      console.error("!!! APP: Failed to apply taste profile !!!", err);
      alert("Failed to apply taste profile. See console for details.");
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'splash': return <SplashView onTabChange={setActiveTab} />;
      case 'recommend': return (
        <RecommendationsView 
          onProfileClear={handleProfileClear} 
        />
      );
      case 'interactive': return <InteractiveRankingsView />;
      case 'lists': return <ListsView />;
      case 'catalogue': return <CatalogueView />;
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
