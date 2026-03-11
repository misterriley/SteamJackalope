import { useState, useEffect, useMemo } from 'react';
import { useUser } from '../context/UserContext';
import { API_BASE_URL } from '../api';
import { Sliders, RefreshCw, AlertCircle, Loader2 } from 'lucide-react';
import GameHeaderImage from './GameHeaderImage';

interface InteractiveGame {
  appid: number;
  name: string;
  header_image: string;
  is_nsfw?: boolean;
  projected_rating: number;
  features: { [key: string]: number };
  kernel_residual: number;
}

interface TasteProfile {
  intercept: number;
  metadata: { [key: string]: number };
  interactive_pool: InteractiveGame[];
}

export default function InteractiveRankingsView() {
  const { steamId } = useUser();
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blurNSFW, setBlurNSFW] = useState(true);

  // Sliders state
  const [weights, setWeights] = useState<{ [key: string]: number }>({});

  useEffect(() => {
    const saved = sessionStorage.getItem('recommendations_filters');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.remove_nsfw !== undefined) setBlurNSFW(parsed.remove_nsfw);
      } catch (e) {}
    }
  }, []);

  useEffect(() => {
    if (!steamId) return;
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/user/insights/${steamId}`);
        if (!res.ok) throw new Error("Could not load taste profile. Have you solved your Taste DNA yet?");
        const data = await res.json();
        if (!data.interactive_pool || data.interactive_pool.length === 0) {
          throw new Error("Interactive pool not found in profile. Please re-run the solver.");
        }
        setProfile(data);
        
        // Initialize weights from metadata
        const initialWeights: { [key: string]: number } = {
          quality: data.metadata.quality || 0,
          difficulty: data.metadata.difficulty || 0,
          price: data.metadata.price || 0,
          age: data.metadata.age || 0,
          length: data.metadata.length || 0,
          popularity: data.metadata.popularity || 0,
          tone: data.metadata.tone || 0,
          kernel: 1.0 // Kernel residual multiplier defaults to 1.0
        };
        setWeights(initialWeights);

      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [steamId]);

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
  };

  const rankedGames = useMemo(() => {
    if (!profile) return [];
    
    const scored = profile.interactive_pool.map(game => {
      let score = profile.intercept;
      score += (game.features.quality || 0) * weights.quality;
      score += (game.features.difficulty || 0) * weights.difficulty;
      score += (game.features.price || 0) * weights.price;
      score += (game.features.age || 0) * weights.age;
      score += (game.features.length || 0) * weights.length;
      score += (game.features.popularity || 0) * weights.popularity;
      score += (game.features.tone || 0) * weights.tone;
      score += game.kernel_residual * weights.kernel;
      
      // Clamp between 0 and 10
      score = Math.max(0, Math.min(10, score));

      return { ...game, current_score: score };
    });

    scored.sort((a, b) => b.current_score - a.current_score);
    return scored.slice(0, 100); // Only render top 100 for performance
  }, [profile, weights]);

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
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-4 mb-8">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight">
          Interactive <span className="text-primary">Rankings</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Instantly tweak the mathematical weights of your Taste DNA and watch your top backlog recommendations re-sort in real-time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sliders Panel */}
        <div className="lg:col-span-1 space-y-6 bg-card border border-border/50 rounded-2xl p-6 shadow-sm h-fit sticky top-24">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold flex items-center gap-2">
              <Sliders size={18} /> Model Weights
            </h3>
            <button 
              onClick={handleReset}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 bg-background px-2 py-1 rounded border border-border"
            >
              <RefreshCw size={12} /> Reset
            </button>
          </div>

          {Object.entries(weights).map(([key, val]) => (
            <div key={key} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="capitalize font-medium">{key}</span>
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
          ))}
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex justify-between items-center mb-2 px-2">
            <h3 className="font-bold text-xl">Top 100 Matches</h3>
            <span className="text-sm text-muted-foreground">Intercept: {profile.intercept.toFixed(2)}</span>
          </div>
          
          <div className="grid grid-cols-1 gap-3">
            {rankedGames.map((game, index) => (
              <a 
                key={game.appid} 
                href={`https://store.steampowered.com/app/${game.appid}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-4 bg-card border border-border/50 rounded-xl overflow-hidden shadow-sm hover:border-primary/50 transition-colors group"
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
                  <h4 className="font-bold text-lg leading-tight truncate max-w-[200px] sm:max-w-sm md:max-w-md lg:max-w-lg" title={game.name}>
                    {game.name}
                  </h4>
                </div>
                <div className="px-6 flex flex-col items-end shrink-0">
                  <div className="text-2xl font-black text-primary">
                    {game.current_score.toFixed(2)}
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
