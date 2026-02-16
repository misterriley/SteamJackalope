import React, { useState, useEffect, useRef } from 'react';
import type { GameMetadata, RecommendationRequest } from '../types';
import { recommend, getGenres, getTermLinks, getRandomGame, getRandomTrendingGame, getMetadata } from '../api';
import GameCard from './GameCard';
import Filters from './Filters';
import SeedSelector from './SeedSelector';
import GenreSelector from './GenreSelector';
import { Search, RotateCcw, AlertCircle, Dices, Sparkles, TrendingUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

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
  const [termLinks, setTermLinks] = useState<Record<string, string>>({});
  const [recommendations, setRecommendations] = useState<GameMetadata[]>([]);
  const [seedGamesMetadata, setSeedGamesMetadata] = useState<GameMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useTrendingRandom, setUseTrendingRandom] = useState(false);
  const isInitialMount = useRef(true);

  const [filters, setFilters] = useState<RecommendationRequest>(() => {
    const defaults: RecommendationRequest = {
      alpha: 1.0,
      beta: 1.0,
      quality_pref: 0.5,
      age_pref: 0.0,
      pop_pref: 0.0,
      disc_pref: 0.0,
      length_pref: 0.0,
      difficulty_pref: 0.0,
      remove_vr: true,
      english_only: true,
      remove_nsfw: true,
      remove_utilities: true,
      remove_unreleased: true,
      top_k: 30,
      prompt: '',
      seed_games: [],
      genres: [],
      debug: false
    };

    const saved = sessionStorage.getItem('recommendations_filters');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return { ...defaults, ...parsed };
      } catch (e) {}
    }
    return defaults;
  });

  // Save filters to session storage
  useEffect(() => {
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
      console.log("Fetching initial data (genres, links)...");
      try {
        const [genres, links] = await Promise.all([
          getGenres(),
          getTermLinks()
        ]);
        
        console.log("Genres received:", genres);
        if (genres && genres.length > 0) {
          setGenresList(genres);
        }
        
        if (links) {
          setTermLinks(links);
        }
      } catch (err) {
        console.error("Failed to load initial data", err);
        setError("Failed to connect to the discovery server at 127.0.0.1:8000. Is the backend running?");
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

  const handleSearch = async (currentFilters: RecommendationRequest) => {
    const filterStr = JSON.stringify(currentFilters);
    if (filterStr === lastSearchRef.current) return;
    
    lastSearchRef.current = filterStr; // Update immediately
    setLoading(true);
    setError(null);
    try {
      const results = await recommend({ ...currentFilters, remove_nsfw: false });
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
    // Immediate search on mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      handleSearch(filters);
      return;
    }

    const timer = setTimeout(() => {
      handleSearch(filters);
    }, 500);

    return () => clearTimeout(timer);
  }, [filters]); // Simplified dependency array

  const handleReset = () => {
    setFilters({
      alpha: 1.0,
      beta: 1.0,
      quality_pref: 0.5,
      age_pref: 0.0,
      pop_pref: 0.0,
      disc_pref: 0.0,
      length_pref: 0.0,
      difficulty_pref: 0.0,
      remove_vr: true,
      english_only: true,
      remove_nsfw: true,
      remove_utilities: true,
      remove_unreleased: true,
      top_k: 30,
      prompt: '',
      seed_games: [],
      genres: [],
      debug: false
    });
  };

  const handleRandomizeSliders = () => {
    const rand = () => parseFloat((Math.random() * 2 - 1).toFixed(1)); // -1.0 to 1.0
    const randCore = () => parseFloat((Math.random() * 2).toFixed(1)); // 0.0 to 2.0 for alpha/beta
    
    setFilters(prev => ({
      ...prev,
      alpha: randCore(),
      beta: randCore(),
      quality_pref: rand(),
      age_pref: rand(),
      pop_pref: rand(),
      disc_pref: rand(),
      length_pref: rand(),
      difficulty_pref: rand(),
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

  const handleProfileUpload = (profile: any) => {
    if (!profile || !profile.metadata || !profile.vibe_vector) {
      alert("Invalid taste profile format.");
      return;
    }

    // Apply solved weights (divide by multipliers to get slider positions)
    // Multipliers from common/constants.py:
    // Quality: 4.0, Age: 1.4, Pop: 1.0, Length: 0.25, Diff: 1.3
    setFilters(prev => ({
      ...prev,
      quality_pref: parseFloat((profile.metadata.quality / 4.0).toFixed(2)),
      age_pref: parseFloat((profile.metadata.age / 1.4).toFixed(2)),
      pop_pref: parseFloat((profile.metadata.popularity / 1.0).toFixed(2)),
      length_pref: parseFloat((profile.metadata.length / 0.25).toFixed(2)),
      difficulty_pref: parseFloat((profile.metadata.difficulty / 1.3).toFixed(2)),
      vibe_vector: profile.vibe_vector,
      // Reset alpha/beta to sensible defaults for personalized mode
      alpha: 1.0,
      beta: 1.5 
    }));
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8">
      {/* Sidebar - Filters */}
      <aside className="w-full lg:w-80 shrink-0">
        <Filters 
          filters={filters} 
          onChange={setFilters} 
          onSearch={() => handleSearch(filters)} 
          loading={loading}
          onProfileUpload={handleProfileUpload}
        />
      </aside>

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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
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
                Trending
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
                      debugMode={false} 
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
                      debugMode={filters.debug} 
                      hideNSFW={filters.remove_nsfw} 
                      termLinks={termLinks}
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
