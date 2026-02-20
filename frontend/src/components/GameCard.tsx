import React, { useState } from 'react';
import type { GameMetadata } from '../types';
import { Star, Clock, Trophy, ExternalLink, ImageOff, Bug, AlertTriangle, Sparkles } from 'lucide-react';

interface GameCardProps {
  game: GameMetadata;
  debugMode?: boolean;
  hideNSFW?: boolean;
  isSeed?: boolean;
  termLinks?: Record<string, string>;
}

const GameCard: React.FC<GameCardProps> = ({ game, debugMode, hideNSFW = true, isSeed = false, termLinks = {} }) => {
  const [imgError, setImgError] = useState(false);
  const steamUrl = `https://store.steampowered.com/app/${game.appid}/`;
  const headerUrl = `https://cdn.akamai.steamstatic.com/steam/apps/${game.appid}/header.jpg`;

  const isNSFW = !!game.is_nsfw;
  const shouldBlur = isNSFW && hideNSFW;

  // Parse genres: handle both comma-separated and list formats
  const parseGenres = (genresStr: any) => {
    if (!genresStr) return [];
    if (Array.isArray(genresStr)) return genresStr;
    
    const str = String(genresStr);
    if (str.startsWith('[') && str.endsWith(']')) {
      try {
        return str
          .slice(1, -1)
          .split(',')
          .map(g => g.trim().replace(/^['']|['"]$/g, ''))
          .filter(g => g);
      } catch (e) {
        console.error("Failed to parse genre list string", str);
      }
    }
    return str.split(',').map(g => g.trim()).filter(g => g);
  };

  // Parse tags: handle dictionary-like format {'Tag': count, ...}
  const parseTags = (tagsStr: any) => {
    if (!tagsStr) return [];
    
    // Handle array format
    if (Array.isArray(tagsStr)) return tagsStr;
    
    // Handle object/dictionary format
    if (typeof tagsStr === 'object') {
      return Object.keys(tagsStr);
    }

    try {
      const str = String(tagsStr);
      const tagMatch = str.match(/'([^']+)'(?=:)/g);
      if (tagMatch) {
        return tagMatch.map(t => t.replace(/'/g, ''));
      }
    } catch (e) {
      console.error("Failed to parse tags string", tagsStr);
    }
    return [];
  };

  const allGenres = parseGenres(game.genres);
  const allTags = parseTags(game.tags);

  const renderLinkableTerm = (term: string, isGenre: boolean) => {
    const link = termLinks[term];
    const baseClasses = isGenre 
      ? "px-1.5 py-0.5 bg-primary/10 text-[9px] font-bold text-primary rounded border border-primary/20 uppercase tracking-tighter whitespace-nowrap transition-colors"
      : "px-1.5 py-0.5 bg-secondary/50 text-[9px] font-bold text-muted-foreground rounded border border-border/50 uppercase tracking-tighter whitespace-nowrap transition-colors";
    
    if (link) {
      return (
        <a 
          key={term}
          href={link}
          target="_blank"
          rel="noopener noreferrer"
          className={`${baseClasses} hover:bg-primary/20 hover:text-primary cursor-pointer`}
          onClick={(e) => e.stopPropagation()}
        >
          {term}
        </a>
      );
    }

    return (
      <span key={term} className={baseClasses}>
        {term}
      </span>
    );
  };

  const handleCardClick = () => {
    window.open(steamUrl, '_blank', 'noopener,noreferrer');
  };

  const renderDebugInfo = () => {
    const components = [
      { label: 'Sem', z: game.z_semantic, w: game.w_semantic, color: 'text-purple-400' },
      { label: 'Tag', z: game.z_tag, w: game.w_tag, color: 'text-pink-400' },
      { label: 'Qual', z: game.z_spps, w: game.w_spps, color: 'text-yellow-400' },
      { label: 'Age', z: game.z_date, w: game.w_date, color: 'text-green-400' },
      { label: 'Pop', z: game.z_pop, w: game.w_pop, color: 'text-blue-400' },
      { label: 'Pr', z: game.z_price, w: game.w_price, color: 'text-emerald-400' },
      { label: 'Len', z: game.z_length, w: game.w_length, color: 'text-cyan-400' },
      { label: 'Diff', z: game.z_difficulty, w: game.w_difficulty, color: 'text-orange-400' },
    ];

    return (
      <div className="mt-4 pt-3 border-t border-border bg-secondary/10 -mx-4 px-4 pb-2">
        <div className="flex items-center gap-1 mb-2">
          <Bug size={10} className="text-primary" />
          <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">Weight Contributions (z * w)</span>
        </div>
        <div className="space-y-1">
          {components.map((comp) => {
            const contrib = (comp.z || 0) * (comp.w || 0);
            if (Math.abs(comp.w || 0) < 0.01) return null;
            
            return (
              <div key={comp.label} className="flex items-center justify-between text-[8px] font-mono">
                <span className={`${comp.color} font-bold w-6`}>{comp.label}</span>
                <div className="flex-grow mx-2 h-1 bg-secondary rounded-full overflow-hidden relative">
                  <div 
                    className={`absolute inset-y-0 ${contrib >= 0 ? 'bg-primary left-1/2' : 'bg-destructive right-1/2'}`}
                    style={{ 
                      width: `${Math.min(50, (Math.abs(contrib) / 8) * 50)}%`,
                    }}
                  />
                </div>
                <span className="text-muted-foreground w-8 text-right">{contrib.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div 
      onClick={handleCardClick}
      className={`bg-card rounded-lg overflow-hidden shadow-lg border border-border hover:border-primary/50 transition-all group flex flex-col h-full cursor-pointer active:scale-[0.98] ${isNSFW ? 'border-orange-500/20' : ''} ${isSeed ? 'ring-2 ring-primary/30 border-primary/40' : ''}`}
    >
      {/* Image Container */}
      <div className="relative aspect-video overflow-hidden bg-secondary/30">
        {!imgError ? (
          <img 
            src={headerUrl} 
            alt={game.name} 
            className={`w-full h-full object-cover group-hover:scale-105 transition-all duration-500 ${shouldBlur ? 'blur-2xl scale-110' : ''}`}
            onError={() => setImgError(true)}
          />
        ) : (
          <div className={`w-full h-full flex flex-col items-center justify-center p-4 text-center bg-secondary/80 ${shouldBlur ? 'blur-xl' : ''}`}>
            <span className="text-sm font-bold text-muted-foreground uppercase tracking-wider line-clamp-2 px-2">
              {game.name}
            </span>
          </div>
        )}
        
        {/* Release Year Label */}
        <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[10px] font-bold text-primary border border-primary/30 z-10">
          {game.release_year}
        </div>

        {/* Price Label */}
        {game.price && (
          <div className="absolute bottom-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[10px] font-bold text-white border border-white/20 z-10">
            {game.price}
          </div>
        )}

        {/* Seed Badge */}
        {isSeed && (
          <div className="absolute top-2 left-2 bg-primary px-2 py-1 rounded text-[10px] font-bold text-primary-foreground shadow-lg z-10 flex items-center gap-1 uppercase tracking-wider">
             <Sparkles size={10} fill="currentColor" />
             Seed Game
          </div>
        )}

        {/* NSFW Overlay */}
        {isNSFW && !isSeed && (
          <div className={`absolute top-2 left-2 px-2 py-1 rounded text-[10px] font-bold border z-10 flex items-center gap-1 ${shouldBlur ? 'bg-orange-600/80 text-white border-orange-400/50 shadow-lg animate-pulse' : 'bg-black/60 backdrop-blur-md text-orange-500 border-orange-500/30'}`}>
             <AlertTriangle size={10} />
             NSFW
          </div>
        )}

        {shouldBlur && (
          <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
            <div className="bg-black/40 backdrop-blur-sm px-4 py-2 rounded-full border border-white/10">
               <span className="text-[10px] font-bold uppercase tracking-widest text-white/80">Content Hidden</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="p-4 flex flex-col flex-grow">
        <div className="flex justify-between items-start mb-2 gap-2">
          <h3 className="text-base font-bold line-clamp-1 text-foreground group-hover:text-primary transition-colors flex-grow">
            {game.name}
          </h3>
          <div className="text-muted-foreground group-hover:text-primary transition-colors shrink-0">
            <ExternalLink size={16} />
          </div>
        </div>

        <div className="min-h-[2.5rem] mb-2">
          <p className="text-xs text-muted-foreground line-clamp-2 flex-grow">
            {game.short_description}
          </p>
        </div>

        {/* Genres Section - Single line dynamic fit */}
        {allGenres.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-1 h-5 overflow-hidden content-start">
            {allGenres.map((genre) => renderLinkableTerm(genre, true))}
          </div>
        )}

        {/* Tags Section - Single line dynamic fit */}
        {allTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-4 h-5 overflow-hidden content-start">
            {allTags.map((tag) => renderLinkableTerm(tag, false))}
          </div>
        )}

        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[10px] mt-auto">
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Star size={12} className="text-yellow-500" />
            <span>{Math.round((game.positive / (game.positive + game.negative || 1)) * 100)}% Positive</span>
          </div>
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Clock size={12} className="text-blue-500" />
            <span>{Math.round(game.estimated_playtime / 60)}h Playtime</span>
          </div>
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Trophy size={12} className="text-orange-500" />
            <span>Diff: {game.difficulty_predicted?.toFixed(1) || '0.0'}</span>
          </div>
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <span className="bg-secondary px-1.5 py-0.5 rounded text-[9px] uppercase font-bold tracking-wider">
              {game.appid}
            </span>
          </div>
        </div>

        {game.weighted_score !== undefined && game.weighted_score !== null && !isSeed && (
          <div className="mt-4 pt-3 border-t border-border flex justify-between items-center">
            <span className="text-[9px] font-bold uppercase text-muted-foreground tracking-widest">Match Score</span>
            <span className="text-primary font-bold text-sm">{(game.weighted_score || 0).toFixed(2)}</span>
          </div>
        )}

        {!isSeed && renderDebugInfo()}
      </div>
    </div>
  );
};

export default GameCard;
