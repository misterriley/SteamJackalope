import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { GameMetadata, RecommendationRequest } from '../types';
import { recommend, getGenres, getTags, getTermLinks, getRandomGame, getRandomTrendingGame, getMetadata } from '../api';
import GameCard from './GameCard';
import Filters from './Filters';
import SeedSelector from './SeedSelector';
import GenreSelector from './GenreSelector';
import TagSelector from './TagSelector';
import { Search, RotateCcw, AlertCircle, Dices, Sparkles, TrendingUp, ArrowUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { GameStatus } from '../types';


const DEFAULT_GENRES = [
  "Action", "Adventure", "Casual", "Indie", "Massively Multiplayer", "RPG", "Simulation", 
  "Sports", "Strategy", "Racing", "Early Access", "Free to Play", "Violent", "Gore", 
  "Nudity", "Sexual Content", "Education", "Software Training", "Utilities", 
  "Design & Illustration", "Web Publishing", "Video Production", "Audio Production", 
  "Photo Editing", "Animation & Modeling", "Accounting"
];

interface RecommendationsViewProps {
  onProfileClear?: () => void;
}

const RecommendationsView: React.FC<RecommendationsViewProps> = ({ onProfileClear }) => {
  const [genresList, setGenresList] = useState<string[]>(DEFAULT_GENRES);
  const [tagsList, setTagsList] = useState<string[]>([]);
  const [termLinks, setTermLinks] = useState<Record<string, string>>({});
  const [recommendations, setRecommendations] = useState<GameMetadata[]>([]);
  const [seedGamesMetadata, setSeedGamesMetadata] = useState<GameMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useTrendingRandom, setUseTrendingRandom] = useState(false);
  const [showMobileFilters, setShowMobileFilters] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
  const isInitialMount = useRef(true);

  // Track window resize and scroll position
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    const handleScroll = () => setShowScrollTop(window.scrollY > 500);
    
    window.addEventListener('resize', handleResize);
    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const [filters, setFilters] = useState<RecommendationRequest>(() => {
    const defaults: RecommendationRequest = {
      alpha: 0.5,
      beta: 0.5,
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

    const saved = sessionStorage.getItem('recommendations_filters');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        console.log("RecommendationsView: Initializing filters from sessionStorage", parsed);
        return { ...defaults, ...parsed };
      } catch (e) {
        console.error("RecommendationsView: Failed to parse saved filters", e);
      }
    }
    console.log("RecommendationsView: Initializing filters with defaults");
    return defaults;
  });

  // Save filters to session storage
  useEffect(() => {
    console.log("RecommendationsView: Persisting filters to sessionStorage", filters);
    sessionStorage.setItem('recommendations_filters', JSON.stringify(filters));
  }, [filters]);

  const [localPrompt, setLocalPrompt] = useState(filters.prompt);

  // Sync local prompt when filters.prompt changes (e.g. on reset)
  useEffect(() => {
    setLocalPrompt(filters.prompt);
  }, [filters.prompt]);

  // Initial data load
  useEffect(() => {
    const fetchInitialData = async () => {
      console.log("Fetching initial data (genres, tags, links)...");
      try {
        const [genres, tags, links] = await Promise.all([
          getGenres(),
          getTags(),
          getTermLinks()
        ]);
        
        console.log("Genres received:", genres);
        if (genres && genres.length > 0) {
          setGenresList(genres);
        }

        if (tags && tags.length > 0) {
          setTagsList(tags);
        }
        
        if (links) {
          setTermLinks(links);
        }
      } catch (err) {
        console.error("Failed to load initial data", err);
        setError("Failed to connect to the discovery server. Is the backend running?");
      }
    };
    fetchInitialData();
  }, []);

  // Sync seed metadata
  useEffect(() => {
    const fetchSeedMetadata = async () => {
      if (filters.seed_games.length > 0) {
        try {
          const meta = await getMetadata(filters.seed_games);
          setSeedGamesMetadata(meta);
        } catch (err) {
          console.error("Failed to fetch seed metadata", err);
        }
      } else {
        setSeedGamesMetadata([]);
      }
    };
    fetchSeedMetadata();
  }, [filters.seed_games]);

  const lastSearchRef = useRef<string>('');
  const lastSignificantFiltersRef = useRef<string>('');

  const handleSearch = async (currentFilters: RecommendationRequest) => {
    const filterStr = JSON.stringify(currentFilters);
    if (filterStr === lastSearchRef.current) return;
    
    lastSearchRef.current = filterStr; // Update immediately
    setLoading(true);
    setError(null);
    try {
      const results = await recommend(currentFilters);
      setRecommendations(results);
      
      if (results.length === 0 && currentFilters.seed_games.length === 0 && !currentFilters.prompt) {
        // Silently handle empty state
      } else if (results.length === 0) {
        setError("No recommendations found matching your criteria. Try loosening your filters.");
      }
    } catch (err) {
      console.error("Recommendation failed", err);
      setError("An error occurred while fetching recommendations.");
      lastSearchRef.current = ''; // Reset on error to allow retry
    } finally {
      setLoading(false);
    }
  };

  // Auto-update effect with debouncing
  useEffect(() => {
    // Define which filters are 'significant' (require a backend re-score)
    // remove_nsfw (Blur) only affects local rendering.
    const { remove_nsfw, ...significantFilters } = filters;
    const sigStr = JSON.stringify(significantFilters);

    // Immediate search on mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      handleSearch(filters);
      lastSignificantFiltersRef.current = sigStr;
      return;
    }

    // Skip if only non-significant filters changed
    if (sigStr === lastSignificantFiltersRef.current) {
      return;
    }

    const timer = setTimeout(() => {
      handleSearch(filters);
      lastSignificantFiltersRef.current = sigStr;
    }, 500);

    return () => clearTimeout(timer);
  }, [filters]); // Still watch filters to ensure UI stays in sync

  const handleReset = () => {
    if (filters.vibe_vector && filters.metadata_weights) {
      // Profile-aware reset: return to solved profile weights
      const meta = filters.metadata_weights;
      setFilters(prev => ({
        ...prev,
        alpha: typeof meta.semantic === 'number' ? meta.semantic : 0.5,
        beta: typeof meta.tag_match === 'number' ? meta.tag_match : 0.5,
        gamma_topic: typeof meta.topic_match === 'number' ? meta.topic_match : 0.5,
        quality_pref: typeof meta.quality === 'number' ? meta.quality : 1.0,
        age_pref: typeof meta.age === 'number' ? meta.age : 0.0,
        pop_pref: typeof meta.popularity === 'number' ? meta.popularity : 0.0,
        disc_pref: typeof meta.discovery === 'number' ? meta.discovery : 0.0,
        length_pref: typeof meta.length === 'number' ? meta.length : 0.0,
        difficulty_pref: typeof meta.difficulty === 'number' ? meta.difficulty : 0.0,
        price_pref: typeof meta.price === 'number' ? meta.price : 0.0,
        
        // Still reset these to clean state
        prompt: '',
        seed_games: [],
        genres: [],
        tags: [],
        
        // Keep global toggles as they are, or reset to defaults?
        // Let's reset to defaults for a true "Reset All"
        remove_vr: true,
        english_only: true,
        remove_nsfw: true,
        remove_utilities: true,
        remove_unreleased: true,
        remove_delisted: true,
        
        // Ensure profile remains active
        profile_filter: 'all'
      }));
    } else {
      // Standard global reset
      setFilters({
        alpha: 0.5,
        beta: 0.5,
        gamma_topic: 0.5,
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
      });
    }
  };

  const handleRandomizeSliders = () => {
    const rand = () => parseFloat((Math.random() * 2 - 1).toFixed(2)); // -1.00 to 1.00
    const randCore = () => parseFloat(Math.random().toFixed(2)); // 0.00 to 1.00 for alpha/beta/gamma
    
    setFilters(prev => ({
      ...prev,
      alpha: randCore(),
      beta: randCore(),
      gamma_topic: randCore(),
      quality_pref: rand(),
      age_pref: rand(),
      pop_pref: rand(),
      disc_pref: 0,
      length_pref: rand(),
      difficulty_pref: rand(),
      price_pref: rand(),
    }));
  };

  const handleRandomSeed = async () => {
    try {
      const game = useTrendingRandom ? await getRandomTrendingGame() : await getRandomGame();
      setFilters(prev => ({
        ...prev,
        seed_games: [...prev.seed_games, game]
      }));
    } catch (err) {
      console.error("Failed to fetch random game", err);
    }
  };

  const handleStatusUpdate = useCallback((appid: number, status: GameStatus) => {
    // If the new status would mean the game is filtered out, remove it locally
    const isExcluded = (
      (filters.profile_filter === 'all' && (status === 'played' || status === 'rated' || status === 'backlog')) ||
      (filters.profile_filter === 'rated' && status === 'rated') ||
      (status === 'ignored') // Ignored is always excluded from results
    );

    if (isExcluded) {
      setRecommendations(prev => prev.filter(g => g.appid !== appid));
    }
  }, [filters.profile_filter]);

  const handleProfileUpload = (profile: any) => {
    if (!profile || !profile.metadata || !profile.vibe_vector) {
      alert("Invalid taste profile format.");
      return;
    }

    const meta = profile.metadata || {};
    const vibeVector = (profile.vibe_vector || []).map((v: any) => v || 0);
    const semVibeVector = (profile.semantic_vibe_vector || []).map((v: any) => v || 0);
    const topicVibeVector = (profile.topic_vibe_vector || []).map((v: any) => v || 0);
    
    console.log("RecommendationsView: Manual Profile Upload", profile);

    setFilters(prev => ({
      ...prev,
      // Sliders show the ABSOLUTE solved weights
      quality_pref: typeof meta.quality === 'number' ? meta.quality : 1.0,
      age_pref: typeof meta.age === 'number' ? meta.age : 0.0,
      pop_pref: typeof meta.popularity === 'number' ? meta.popularity : 0.0,
      length_pref: typeof meta.length === 'number' ? meta.length : 0.0,
      difficulty_pref: typeof meta.difficulty === 'number' ? meta.difficulty : 0.0,
      price_pref: typeof meta.price === 'number' ? meta.price : 0.0,
      alpha: typeof meta.semantic === 'number' ? meta.semantic : 1.0,
      beta: typeof meta.tag_match === 'number' ? meta.tag_match : 1.0,
      gamma_topic: typeof meta.topic_match === 'number' ? meta.topic_match : 0.5,
      
      vibe_vector: vibeVector,
      semantic_vibe_vector: semVibeVector,
      topic_vibe_vector: topicVibeVector,
      metadata_weights: meta,
      intercept: typeof profile.intercept === 'number' ? profile.intercept : 5.0,
      scaling_factor: typeof profile.scaling_factor === 'number' ? profile.scaling_factor : 1.0,
      disc_pref: typeof meta.discovery === 'number' ? meta.discovery : 0,
      
      profile_filter: 'all',
      library_appids: profile.library_appids || [],
      rated_appids: profile.rated_appids || [],
      ignored_appids: profile.ignored_appids || [],
      library_details: profile.library_details || {}
    }));
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8 relative">
      {/* Mobile Filter Toggle */}
      <button 
        onClick={() => setShowMobileFilters(!showMobileFilters)}
        className="lg:hidden fixed bottom-6 right-6 z-[60] bg-primary text-primary-foreground p-4 rounded-full shadow-2xl flex items-center gap-2 font-bold animate-in fade-in zoom-in"
      >
        <RotateCcw className={`transition-transform duration-500 ${showMobileFilters ? 'rotate-180' : ''}`} size={20} />
        {showMobileFilters ? 'Close Preferences' : 'Preferences'}
      </button>

      {/* Scroll to Top Button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, scale: 0.5, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.5, y: 20 }}
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className={`fixed ${windowWidth < 1024 ? 'bottom-24' : 'bottom-8'} right-8 z-[60] bg-secondary border border-border text-foreground p-3 rounded-full shadow-2xl hover:bg-secondary/80 transition-all group`}
            title="Scroll to Top"
          >
            <ArrowUp size={24} className="group-hover:-translate-y-1 transition-transform" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Sidebar - Filters */}
      <AnimatePresence>
        {(showMobileFilters || windowWidth >= 1024) && (
          <motion.aside 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className={`w-full lg:w-80 shrink-0 ${showMobileFilters ? 'fixed inset-0 z-[55] bg-background/95 backdrop-blur-md p-4 overflow-y-auto' : 'hidden lg:block'}`}
          >
            <Filters 
              filters={filters} 
              onChange={setFilters} 
              onSearch={() => {
                handleSearch(filters);
                setShowMobileFilters(false);
              }} 
              loading={loading}
              onProfileUpload={(profile) => {
                handleProfileUpload(profile);
                setShowMobileFilters(false);
              }}
              onProfileClear={onProfileClear}
            />
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-grow space-y-8">
        {/* Search Input and Selectors */}
        <section className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-6">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground" size={20} />
            <input
              type="text"
              placeholder="Describe the kind of experience you're looking for... (Press Enter to search)"
              className="w-full bg-secondary border-none rounded-lg pl-12 pr-4 py-3 text-lg focus:ring-2 focus:ring-primary/50 transition-all outline-none"
              value={localPrompt}
              onChange={(e) => setLocalPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  setFilters({ ...filters, prompt: localPrompt });
                }
              }}
              onBlur={() => {
                if (localPrompt !== filters.prompt) {
                  setFilters({ ...filters, prompt: localPrompt });
                }
              }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-2 lg:col-span-1">
              <label className="text-sm font-medium text-muted-foreground">Seed Games</label>
              <SeedSelector 
                selected={filters.seed_games} 
                onChange={(seeds) => setFilters({ ...filters, seed_games: seeds })} 
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">Genres</label>
              <GenreSelector 
                options={genresList} 
                selected={filters.genres} 
                onChange={(genres) => setFilters({ ...filters, genres: genres })} 
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">Tags (Require All)</label>
              <TagSelector 
                options={tagsList} 
                selected={filters.tags} 
                onChange={(tags) => setFilters({ ...filters, tags: tags })} 
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button 
              onClick={handleReset}
              className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
            >
              <RotateCcw size={14} />
              Reset All
            </button>
            <button 
              onClick={handleRandomizeSliders}
              className="text-sm text-muted-foreground hover:text-primary flex items-center gap-1 transition-colors"
            >
              <Dices size={14} />
              Randomize Sliders
            </button>
            <button 
              onClick={handleRandomSeed}
              className="text-sm text-muted-foreground hover:text-primary flex items-center gap-1 transition-colors"
            >
              <Sparkles size={14} />
              Random Seed Game
            </button>
            <div className="flex items-center gap-2 ml-2">
              <input
                type="checkbox"
                id="trending-random"
                checked={useTrendingRandom}
                onChange={(e) => setUseTrendingRandom(e.target.checked)}
                className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary bg-secondary"
              />
              <label htmlFor="trending-random" className="text-xs text-muted-foreground cursor-pointer hover:text-foreground flex items-center gap-1">
                <TrendingUp size={12} />
                <a 
                  href="https://store.steampowered.com/charts/mostplayed" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="hover:text-primary transition-colors underline decoration-dotted underline-offset-2"
                >
                  Trending
                </a>
              </label>
            </div>
            <div className="text-xs text-muted-foreground italic ml-auto hidden sm:block">
              Results update automatically as you adjust preferences.
            </div>
          </div>
        </section>

        {/* Results */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">
              {(recommendations || []).length > 0 ? 'Results for you' : 'Discover something new'}
            </h2>
            {(recommendations || []).length > 0 && (
              <span className="text-sm text-muted-foreground">
                Found {(recommendations || []).length} matching games
              </span>
            )}
          </div>

          {error && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 flex items-start gap-4 text-destructive mb-8">
              <AlertCircle size={24} className="shrink-0" />
              <div>
                <p className="font-bold">Something went wrong</p>
                <p className="text-sm opacity-90">{error}</p>
              </div>
            </div>
          )}

          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="bg-card rounded-lg h-80 animate-pulse border border-border" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6">
              <AnimatePresence mode="popLayout">
                {/* Seed Games */}
                {(seedGamesMetadata || []).map((game, index) => (
                  <motion.div
                    key={`seed-${game.appid}`}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    layout
                  >
                    <GameCard 
                      game={game} 
                      hideNSFW={filters.remove_nsfw}
                      isSeed={true}
                      termLinks={termLinks}
                    />
                  </motion.div>
                ))}

                {/* Recommendations */}
                {(recommendations || []).map((game, index) => (
                  <motion.div
                    key={game.appid}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: ((seedGamesMetadata || []).length + index) * 0.05 }}
                    layout
                  >
                    <GameCard 
                      game={game} 
                      hideNSFW={filters.remove_nsfw} 
                      termLinks={termLinks}
                      onStatusUpdate={handleStatusUpdate}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}

          {!loading && recommendations.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-4 bg-card/50 border border-dashed border-border rounded-2xl">
              <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center">
                <Search size={32} className="text-muted-foreground" />
              </div>
              <div>
                <h3 className="text-xl font-bold">No results found</h3>
                <p className="text-muted-foreground">Try adjusting your filters or search query.</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default RecommendationsView;
