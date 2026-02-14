import React, { useState, useEffect } from 'react';
import type { GameMetadata, ListResponse } from '../types';
import { getList } from '../api';
import { Trophy, TrendingUp, Clock, History, Swords, Info, Tags } from 'lucide-react';

const ListsView: React.FC = () => {
  const [activeCategory, setActiveCategory] = useState('quality');
  const [discoveryPref, setDiscoveryPref] = useState(0);
  const [data, setData] = useState<ListResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const categories = [
    { id: 'quality', label: 'Rating', icon: <Trophy size={18} />, description: 'Games ranked by Bayesian quality scores.' },
    { id: 'popularity', label: 'Popularity', icon: <TrendingUp size={18} />, description: 'Games ranked by total volume of user reviews.' },
    { id: 'length', label: 'Playtime', icon: <Clock size={18} />, description: 'Games ranked by median playtime.' },
    { id: 'age', label: 'Release Age', icon: <History size={18} />, description: 'Games ranked by their release date.' },
    { id: 'difficulty', label: 'Difficulty', icon: <Swords size={18} />, description: 'Games ranked by predicted difficulty.' },
    { id: 'difficulty_tags', label: 'Difficulty Tags', icon: <Tags size={18} />, description: 'Tags that most strongly predict high or low difficulty.' },
  ];

  const fetchData = async () => {
    setLoading(true);
    try {
      // For difficulty_tags, we use the difficulty endpoint
      const endpoint = activeCategory === 'difficulty_tags' ? 'difficulty' : activeCategory;
      const result = await getList(endpoint, discoveryPref);
      setData(result);
    } catch (err) {
      console.error("Failed to fetch list", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [activeCategory, discoveryPref]);

  return (
    <div className="space-y-8">
      {/* Category Tabs */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCategory(cat.id)}
            className={`flex flex-col items-center gap-3 p-4 rounded-xl border transition-all ${
              activeCategory === cat.id 
                ? 'bg-primary/10 border-primary text-primary shadow-sm shadow-primary/5' 
                : 'bg-card border-border text-muted-foreground hover:border-muted hover:text-foreground'
            }`}
          >
            {cat.icon}
            <span className="text-sm font-bold">{cat.label}</span>
          </button>
        ))}
      </div>

      {/* Header & Controls */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <h2 className="text-2xl font-bold mb-1">{categories.find(c => c.id === activeCategory)?.label}</h2>
            <p className="text-muted-foreground text-sm">{categories.find(c => c.id === activeCategory)?.description}</p>
          </div>

          {activeCategory === 'quality' && (
            <div className="w-full md:w-64">
              <div className="flex justify-between items-center mb-2">
                <label className="text-xs font-bold uppercase tracking-widest text-muted-foreground flex items-center gap-1">
                  Discovery
                  <div className="group relative">
                    <Info size={14} className="text-muted-foreground cursor-help" />
                    <div className="absolute bottom-full right-0 mb-2 px-3 py-2 bg-popover text-popover-foreground text-xs rounded shadow-xl w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none border border-border z-50">
                      Lower = Popular games, Higher = Hidden gems.
                    </div>
                  </div>
                </label>
                <span className="text-xs font-mono text-primary font-bold">{discoveryPref.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={-1}
                max={1}
                step={0.1}
                value={discoveryPref}
                onChange={(e) => setDiscoveryPref(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {[...Array(2)].map((_, i) => (
             <div key={i} className="space-y-4">
               <div className="h-8 w-48 bg-muted animate-pulse rounded" />
               <div className="grid grid-cols-1 gap-4">
                 {[...Array(3)].map((_, j) => (
                   <div key={j} className="h-32 bg-card rounded-lg border border-border animate-pulse" />
                 ))}
               </div>
             </div>
          ))}
        </div>
      ) : data ? (
        activeCategory === 'difficulty_tags' ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-12">
            {/* Hardest Tags */}
            <div className="space-y-6">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <span className="w-2 h-8 bg-orange-500 rounded-full" />
                Hardest Predictors
              </h3>
              <div className="grid grid-cols-1 gap-4">
                {data.tag_impacts?.slice(0, 20).map((impact) => (
                  <TagImpactItem key={impact.tag} tag={impact.tag} impact={impact.impact} />
                ))}
              </div>
            </div>

            {/* Easiest Tags */}
            <div className="space-y-6">
              <h3 className="text-xl font-bold flex items-center gap-2">
                <span className="w-2 h-8 bg-cyan-500 rounded-full" />
                Easiest Predictors
              </h3>
              <div className="grid grid-cols-1 gap-4">
                {data.tag_impacts?.slice(-20).reverse().map((impact) => (
                  <TagImpactItem key={impact.tag} tag={impact.tag} impact={impact.impact} />
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-12">
                      {/* Top List */}
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <span className="w-2 h-8 bg-green-500 rounded-full" />
                          {
                            activeCategory === 'quality' ? 'Top Rated' : 
                            activeCategory === 'popularity' ? 'Most Popular' :
                            activeCategory === 'length' ? 'Longest Games' : 
                            activeCategory === 'age' ? 'Newest Releases' :
                            activeCategory === 'difficulty' ? 'Hardest Challenges' : 'Top Tier'
                          }
                        </h3>
                        <div className="grid grid-cols-1 gap-4">
                          {data.top.slice(0, 10).map((game) => (
                            <ListGameItem key={game.appid} game={game as GameMetadata} category={activeCategory} />
                          ))}
                        </div>
                      </div>
            
                      {/* Bottom List */}
                      <div className="space-y-6">
                        <h3 className="text-xl font-bold flex items-center gap-2">
                          <span className="w-2 h-8 bg-red-500 rounded-full" />
                          {
                            activeCategory === 'quality' ? 'Bottom Rated' : 
                            activeCategory === 'popularity' ? 'Least Popular' :
                            activeCategory === 'length' ? 'Shortest Games' : 
                            activeCategory === 'age' ? 'Oldest Releases' :
                            activeCategory === 'difficulty' ? 'Relaxing Experiences' : 'Niche/Lesser Known'
                          }
                        </h3>
                        <div className="grid grid-cols-1 gap-4">
                          {data.bottom.slice(0, 10).map((game) => (
                            <ListGameItem key={game.appid} game={game as GameMetadata} category={activeCategory} />
                          ))}
                        </div>
                      </div>          </div>
        )
      ) : null}
    </div>
  );
};

const TagImpactItem = ({ tag, impact }: { tag: string, impact: number }) => {
  return (
    <div className="bg-card border border-border rounded-lg p-3 flex items-center justify-between hover:border-primary/50 transition-all group">
      <div className="flex items-center gap-3">
        <div className={`w-1 h-6 rounded-full ${impact >= 0 ? 'bg-orange-500' : 'bg-cyan-500'}`} />
        <span className="font-bold text-sm text-foreground uppercase tracking-wider">{tag}</span>
      </div>
      <div className={`font-mono text-sm font-bold ${impact >= 0 ? 'text-orange-400' : 'text-cyan-400'}`}>
        {impact >= 0 ? '+' : ''}{impact.toFixed(3)}
      </div>
    </div>
  );
};

const ListGameItem = ({ game, category }: { game: GameMetadata, category: string }) => {
  const [imgError, setImgError] = useState(false);
  const iconUrl = `https://cdn.akamai.steamstatic.com/steam/apps/${game.appid}/capsule_184x69.jpg`;
  
  const getStat = () => {
    switch(category) {
      case 'quality': {
        const pts = Math.round(game.quality_score! * 100);
        return `${pts} ${Math.abs(pts) === 1 ? 'point' : 'pts'}`;
      }
      case 'popularity': {
        const count = game.total_reviews || 0;
        return `${count.toLocaleString()} ${count === 1 ? 'review' : 'reviews'}`;
      }
      case 'length': return `${Math.round(game.playtime! / 60)}h`;
      case 'age': return game.release_date;
      case 'difficulty': return `${game.difficulty_predicted?.toFixed(1)}`;
      default: return '';
    }
  }

  return (
    <a 
      href={`https://store.steampowered.com/app/${game.appid}`}
      target="_blank"
      rel="noopener noreferrer"
      className="bg-card border border-border rounded-lg p-3 flex items-center gap-4 hover:border-primary/50 transition-all group no-underline"
    >
      <div className="w-24 h-12 shrink-0 overflow-hidden rounded bg-secondary flex items-center justify-center">
        {!imgError ? (
          <img 
            src={iconUrl} 
            alt={game.name} 
            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
            onError={() => setImgError(true)}
          />
        ) : (
          <span className="text-[8px] font-bold text-muted-foreground/40 text-center uppercase leading-tight px-1 line-clamp-2">
            {game.name}
          </span>
        )}
      </div>
      <div className="flex-grow min-w-0">
        <h4 className="font-bold text-sm truncate group-hover:text-primary transition-colors">{game.name}</h4>
        <p className="text-[10px] text-muted-foreground uppercase tracking-widest">{game.appid}</p>
      </div>
      <div className="text-right">
        <div className="text-sm font-mono font-bold text-primary">{getStat()}</div>
      </div>
    </a>
  );
}

export default ListsView;
