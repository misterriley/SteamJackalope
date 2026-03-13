import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useUser } from '../context/UserContext';
import { useContextMenu } from '../context/ContextMenuContext';
import { API_BASE_URL } from '../api';
import { Sliders, RefreshCw, AlertCircle, Loader2, Target } from 'lucide-react';
import GameHeaderImage from './GameHeaderImage';
import TagSelector from './TagSelector';
import SeedSelector from './SeedSelector';
import GameHoverCard from './GameHoverCard';
import { type GameStatus, type GameMetadata } from '../types';

interface InteractiveGame extends Omit<GameMetadata, 'tags'> {
  appid: number;
  name: string;
  header_image: string;
  is_nsfw?: boolean;
  is_backlog?: boolean;
  is_wishlist?: boolean;
  is_free?: boolean;
  tags?: string[];
  projected_rating: number;
  features: { [key: string]: number };
  kernel_residual: number;
  raw_price?: string;
  raw_difficulty?: number;
  raw_length?: number;
}

interface TasteProfile {
  intercept: number;
  metadata: { [key: string]: number };
  interactive_pool: InteractiveGame[];
}

export default function InteractiveRankingsView() {
  const { steamId } = useUser();
  const { showContextMenu } = useContextMenu();
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blurNSFW, setBlurNSFW] = useState(true);

  // Hover state (Stable Pattern)
  const [hoverState, setHoverState] = useState<{ game: any, anchor: DOMRect } | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleGameMouseEnter = (e: React.MouseEvent, game: any) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    
    const delay = hoverState ? 50 : 150;
    
    hoverTimerRef.current = setTimeout(() => {
      setHoverState({ game, anchor: rect });
    }, delay);
  };

  const handleGameMouseLeave = () => {
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = setTimeout(() => {
      setHoverState(null);
    }, 100);
  };

  // Filters state
  const [weights, setWeights] = useState<{ [key: string]: number }>({});
  const [includedTags, setIncludedTags] = useState<string[]>([]);
  const [excludedTags, setExcludedTags] = useState<string[]>([]);
  
  // Kernel Targeting State
  const [seedGames, setSeedGames] = useState<string[]>([]);
  const [seedKernelLimit, setSeedKernelLimit] = useState<number>(250);
  const [kernelAppidScores, setKernelAppidScores] = useState<{ [appid: string]: number } | null>(null);
  const [isFetchingKernel, setIsFetchingKernel] = useState(false);

  useEffect(() => {
    const saved = sessionStorage.getItem('recommendations_filters');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.remove_nsfw !== undefined) setBlurNSFW(parsed.remove_nsfw);
      } catch (e) {}
    }
  }, []);

  const fetchProfile = useCallback(async (isManual = false) => {
    if (!steamId) return;
    setLoading(true);
    setError(null);
    try {
      const cacheBuster = isManual ? `?t=${Date.now()}` : '';
      const res = await fetch(`${API_BASE_URL}/user/insights/${steamId}${cacheBuster}`);
      if (!res.ok) throw new Error("Could not load taste profile. Have you solved your Taste DNA yet?");
      const data = await res.json();
      if (!data.interactive_pool || data.interactive_pool.length === 0) {
        throw new Error("Interactive pool not found in profile. Please re-run the solver.");
      }
      setProfile(data);
      
      const initialWeights: { [key: string]: number } = {
        quality: data.metadata.quality || 0,
        difficulty: data.metadata.difficulty || 0,
        price: data.metadata.price || 0,
        age: data.metadata.age || 0,
        length: data.metadata.length || 0,
        popularity: data.metadata.popularity || 0,
        tone: data.metadata.tone || 0,
        kernel: 1.0 
      };
      setWeights(initialWeights);
      setIncludedTags([]);
      setExcludedTags([]);
      setSeedGames([]);
      setKernelAppidScores(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [steamId]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  useEffect(() => {
    if (seedGames.length === 0) {
      setKernelAppidScores(null);
      return;
    }
    
    let isMounted = true;
    const fetchKernel = async () => {
      setIsFetchingKernel(true);
      try {
        const res = await fetch(`${API_BASE_URL}/user/interactive/kernel`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seed_games: seedGames })
        });
        if (!res.ok) throw new Error("Failed to fetch kernel sims");
        const data = await res.json();
        if (isMounted) {
          setKernelAppidScores(data);
        }
      } catch (e) {
        console.error("Kernel fetch error", e);
      } finally {
        if (isMounted) setIsFetchingKernel(false);
      }
    };
    fetchKernel();
    return () => { isMounted = false; };
  }, [seedGames]);

  const handleContextMenu = useCallback((e: React.MouseEvent, game: InteractiveGame) => {
    e.preventDefault();
    
    let currentStatus: GameStatus = 'none';
    if (game.is_backlog) currentStatus = 'backlog';
    else if (game.is_wishlist) currentStatus = 'wishlist';

    showContextMenu({
      x: e.clientX,
      y: e.clientY,
      appid: game.appid,
      steamId: steamId || '',
      currentStatus,
      onUpdate: (aid: number, status: GameStatus) => {
        setProfile((prev: any) => {
          if (!prev) return prev;
          const removalStatuses: GameStatus[] = ['played', 'rated', 'ignored'];
          
          if (removalStatuses.includes(status)) {
            return {
              ...prev,
              interactive_pool: prev.interactive_pool.filter((g: any) => g.appid !== aid)
            };
          }
          
          return {
            ...prev,
            interactive_pool: prev.interactive_pool.map((g: any) => {
              if (g.appid === aid) {
                return {
                  ...g,
                  is_backlog: status === 'backlog',
                  is_wishlist: status === 'wishlist'
                };
              }
              return g;
            })
          };
        });
      }
    });
  }, [showContextMenu, steamId]);

  const allTags = useMemo(() => {
    if (!profile?.interactive_pool) return [];
    const tags = new Set<string>();
    profile.interactive_pool.forEach(game => {
      game.tags?.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }, [profile]);

  const handleReset = () => {
    if (!profile) return;
    setWeights({
      quality: profile.metadata.quality || 0,
      difficulty: profile.metadata.difficulty || 0,
      price: profile.metadata.price || 0,
      age: profile.metadata.age || 0,
      length: profile.metadata.length || 0,
      popularity: profile.metadata.popularity || 0,
      tone: profile.metadata.tone || 0,
      kernel: 1.0
    });
    setIncludedTags([]);
    setExcludedTags([]);
    setSeedGames([]);
  };

  const rankedGames = useMemo(() => {
    if (!profile) return [];
    
    let filtered = profile.interactive_pool;

    if (seedGames.length > 0 && kernelAppidScores !== null) {
      const poolWithSims = filtered.map(game => ({
        ...game,
        _kernel_sim: kernelAppidScores[game.appid.toString()] || 0
      }));
      poolWithSims.sort((a, b) => b._kernel_sim - a._kernel_sim);
      filtered = poolWithSims.slice(0, seedKernelLimit);
    }

    if (includedTags.length > 0) {
      filtered = filtered.filter(game => 
        game.tags && includedTags.every(tag => game.tags!.includes(tag))
      );
    }
    if (excludedTags.length > 0) {
      filtered = filtered.filter(game => 
        !game.tags || !excludedTags.some(tag => game.tags!.includes(tag))
      );
    }

    const scored = filtered.map(game => {
      let score = profile.intercept;
      score += (game.features.quality || 0) * weights.quality;
      score += (game.features.difficulty || 0) * weights.difficulty;
      score += (game.features.price || 0) * weights.price;
      score += (game.features.age || 0) * weights.age;
      score += (game.features.length || 0) * weights.length;
      score += (game.features.popularity || 0) * weights.popularity;
      score += (game.features.tone || 0) * weights.tone;
      score += (game.kernel_residual || 0) * (weights.kernel ?? 1.0);
      
      score = Math.max(0, Math.min(10, score));
      return { ...game, current_score: score };
    });

    scored.sort((a, b) => b.current_score - a.current_score);
    return scored.slice(0, 100); 
  }, [profile, weights, includedTags, excludedTags, seedGames, seedKernelLimit, kernelAppidScores]);

  if (!steamId) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center h-[60vh]">
        <AlertCircle size={48} className="text-muted-foreground mb-4" />
        <h2 className="text-2xl font-bold mb-2">No Profile Found</h2>
        <p className="text-muted-foreground">Please enter your Steam ID in the Personalize tab first.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center h-[60vh]">
        <Loader2 size={48} className="animate-spin text-primary mb-4" />
        <h2 className="text-2xl font-bold mb-2">Loading Interactive Profile</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center h-[60vh]">
        <AlertCircle size={48} className="text-destructive mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error</h2>
        <p className="text-muted-foreground">{error}</p>
      </div>
    );
  }

  if (!profile) return null;

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500 relative">
      <div className="text-center space-y-4 mb-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight">
          Interactive <span className="text-primary">Rankings</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Instantly tweak the mathematical weights of your Taste DNA and watch your top backlog recommendations re-sort in real-time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1 space-y-6 bg-card border border-border/50 rounded-2xl p-6 shadow-sm h-fit sticky top-24 max-h-[calc(100vh-8rem)] overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold flex items-center gap-2">
              <Sliders size={18} /> Filters & Weights
            </h3>
            <div className="flex items-center gap-2">
              <button 
                onClick={() => fetchProfile(true)}
                className="text-xs text-primary hover:text-primary/80 flex items-center gap-1 bg-primary/10 px-2 py-1 rounded border border-primary/20 transition-colors"
              >
                <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Sync Data
              </button>
              <button 
                onClick={handleReset}
                className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 bg-background px-2 py-1 rounded border border-border transition-colors"
              >
                Reset
              </button>
            </div>
          </div>

          <div className="space-y-4 pb-4 border-b border-border/50">
            <h3 className="font-bold text-sm text-primary flex items-center gap-2">
              <Target size={14} /> Kernel Targeting
            </h3>
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label className="text-xs text-muted-foreground font-medium">Seed Games:</label>
                {isFetchingKernel && <Loader2 size={12} className="animate-spin text-primary" />}
              </div>
              <SeedSelector 
                selected={seedGames}
                onChange={setSeedGames}
                placeholder="Select seed games..."
              />
            </div>
            
            <div className="space-y-2" style={{ opacity: seedGames.length > 0 ? 1 : 0.5 }}>
              <div className="flex justify-between text-xs font-medium items-center">
                <span>Pool Size Limit</span>
                <span className="font-mono text-muted-foreground">{seedKernelLimit} games</span>
              </div>
              <input
                type="range"
                min="50"
                max="2000"
                step="50"
                value={seedKernelLimit}
                onChange={(e) => setSeedKernelLimit(parseInt(e.target.value))}
                className="w-full accent-primary"
                disabled={seedGames.length === 0}
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="font-bold text-sm text-primary">Tags</h3>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground font-medium">Must include ALL of:</label>
              <TagSelector 
                options={allTags.filter(t => !excludedTags.includes(t))}
                selected={includedTags}
                onChange={setIncludedTags}
                placeholder="Include tags..."
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground font-medium">Must NOT include ANY of:</label>
              <TagSelector 
                options={allTags.filter(t => !includedTags.includes(t))}
                selected={excludedTags}
                onChange={setExcludedTags}
                placeholder="Exclude tags..."
              />
            </div>
          </div>

          <hr className="border-border/50" />

          <div className="space-y-4">
            <h3 className="font-bold text-sm text-primary mb-4">Model Weights</h3>
            {Object.entries(weights).map(([key, val]) => {
              const displayLabels: { [key: string]: string } = {
                age: 'Release Date',
                quality: 'Quality',
                difficulty: 'Difficulty',
                price: 'Price',
                length: 'Length',
                popularity: 'Popularity',
                tone: 'Tone',
                kernel: 'Kernel Strength'
              };

              return (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between text-sm items-center">
                    <span className="font-medium">{displayLabels[key] || key}</span>
                    <span className="font-mono text-muted-foreground">{val.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.05"
                    value={val}
                    onChange={(e) => setWeights(prev => ({ ...prev, [key]: parseFloat(e.target.value) }))}
                    className="w-full accent-primary"
                  />
                </div>
              );
            })}
          </div>
        </div>

        <div className="lg:col-span-3 space-y-4">
          <div className="flex justify-between items-center mb-2 px-2">
            <h3 className="font-bold text-xl">
              Top 100 Matches
              {seedGames.length > 0 && kernelAppidScores !== null && (
                <span className="ml-2 text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                  Targeted
                </span>
              )}
            </h3>
            <span className="text-sm text-muted-foreground">Intercept: {profile.intercept.toFixed(2)}</span>
          </div>
          
          {rankedGames.length === 0 ? (
            <div className="bg-card border border-border/50 rounded-xl p-12 text-center text-muted-foreground">
              No games found matching these filters.
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3">
              {rankedGames.map((game, index) => (
                <a 
                  key={game.appid} 
                  href={`https://store.steampowered.com/app/${game.appid}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  onContextMenu={(e) => handleContextMenu(e, game as InteractiveGame)}
                  onMouseEnter={(e) => handleGameMouseEnter(e, game)}
                  onMouseLeave={handleGameMouseLeave}
                  className="flex items-center gap-4 bg-card border border-border/50 rounded-xl overflow-hidden shadow-sm hover:border-primary/50 transition-colors group relative"
                >
                  <div className="w-12 text-center font-bold text-muted-foreground group-hover:text-primary transition-colors">
                    #{index + 1}
                  </div>
                  <GameHeaderImage 
                    appid={game.appid} 
                    header_image={game.header_image}
                    isNSFW={game.is_nsfw}
                    blurNSFW={blurNSFW}
                    className="w-32 h-[4.5rem] object-cover rounded shadow-sm border border-border/50 group-hover:scale-105 transition-transform"
                  />
                  <div className="flex-grow py-2">
                    <h4 className="font-bold text-lg leading-tight truncate">
                      {game.name}
                    </h4>
                    <div className="flex items-center gap-2 mt-0.5 mb-1">
                      {game.is_backlog && <span className="text-[10px] bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">Backlog</span>}
                      {game.is_wishlist && <span className="text-[10px] bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">Wishlist</span>}
                      {game.is_free && <span className="text-[10px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">Free</span>}
                    </div>
                    {game.tags && (
                      <div className="text-xs text-muted-foreground truncate opacity-70">
                        {game.tags.slice(0, 5).join(', ')}
                      </div>
                    )}
                  </div>
                  <div className="px-6 flex flex-col items-end shrink-0">
                    <div className="text-2xl font-black text-primary">
                      {(game as any).current_score.toFixed(2)}
                    </div>
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <GameHoverCard 
        game={hoverState?.game || null} 
        anchorRect={hoverState?.anchor} 
        weights={weights}
      />
    </div>
  );
}
