import React from 'react';
import type { GameMetadata } from '../types';
import { ExternalLink, Sparkles, Star, Clock, Trophy } from 'lucide-react';
import { useContextMenu } from '../context/ContextMenuContext';
import { useUser } from '../context/UserContext';
import type { GameStatus } from '../types';

import GameHeaderImage from './GameHeaderImage';

interface GameCardProps {
  game: GameMetadata;
  hideNSFW?: boolean;
  isSeed?: boolean;
  termLinks?: Record<string, string>;
  onStatusUpdate?: (appid: number, status: GameStatus, rating?: number) => void;
  onMouseEnter?: (e: React.MouseEvent, game: any) => void;
  onMouseLeave?: () => void;
}

const GameCard: React.FC<GameCardProps> = ({ 
  game, 
  hideNSFW = true, 
  isSeed = false, 
  termLinks = {},
  onStatusUpdate,
  onMouseEnter,
  onMouseLeave
}) => {
  const { showContextMenu } = useContextMenu();
  const { steamId } = useUser();
  const steamUrl = `https://store.steampowered.com/app/${game.appid}/`;
  
  const handleMouseEnter = (e: React.MouseEvent) => {
    if (onMouseEnter) onMouseEnter(e, game);
  };

  const handleMouseLeave = () => {
    if (onMouseLeave) onMouseLeave();
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    showContextMenu({
      x: e.clientX,
      y: e.clientY,
      appid: game.appid,
      steamId: steamId || "", 
      onUpdate: onStatusUpdate
    });
  };

  const isNSFW = !!game.is_nsfw;
  const shouldBlur = isNSFW && hideNSFW;

  const parseGenres = (genresStr: any) => {
    if (!genresStr) return [];
    if (Array.isArray(genresStr)) return genresStr;
    const str = String(genresStr);
    return str.split(',').map(g => g.trim()).filter(g => g);
  };

  const parseTags = (tagsStr: any) => {
    if (!tagsStr) return [];
    if (Array.isArray(tagsStr)) return tagsStr;
    if (typeof tagsStr === 'object') return Object.keys(tagsStr);
    return [];
  };

  const allGenres = parseGenres(game.genres);
  const allTags = parseTags(game.tags);

  const renderLinkableTerm = (term: string, isGenre: boolean) => {
    const link = termLinks[term];
    const baseClasses = isGenre 
      ? "px-1.5 py-0.5 bg-primary/10 text-[9px] font-bold text-primary rounded border border-primary/20 uppercase tracking-tighter whitespace-nowrap"
      : "px-1.5 py-0.5 bg-secondary/50 text-[9px] font-bold text-muted-foreground rounded border border-border/50 uppercase tracking-tighter whitespace-nowrap";
    if (link) {
      return (
        <a key={term} href={link} target="_blank" rel="noopener noreferrer" className={`${baseClasses} hover:bg-primary/20 hover:text-primary cursor-pointer`} onClick={(e) => e.stopPropagation()}>
          {term}
        </a>
      );
    }
    return <span key={term} className={baseClasses}>{term}</span>;
  };

  const handleCardClick = () => { window.open(steamUrl, '_blank', 'noopener,noreferrer'); };

  return (
    <div 
      onClick={handleCardClick}
      onContextMenu={handleContextMenu}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`bg-card rounded-lg overflow-hidden shadow-lg border border-border hover:border-primary/50 transition-all group flex flex-col h-full cursor-pointer active:scale-[0.98] ${isNSFW ? 'border-orange-500/20' : ''} ${isSeed ? 'ring-2 ring-primary/30 border-primary/40' : ''} relative`}
    >
      <div className="relative aspect-video overflow-hidden bg-secondary/30">
        <GameHeaderImage appid={game.appid} header_image={game.header_image} isNSFW={isNSFW} blurNSFW={hideNSFW} className="w-full h-full object-cover group-hover:scale-105 transition-all duration-500" alt={game.name} />
        <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[10px] font-bold text-primary border border-primary/30 z-10">{game.release_year}</div>
        {(game.is_free || (game.price && (game.price.toLowerCase().includes("free") || game.price === ""))) && (<div className="absolute top-0 left-0 px-2 py-1 bg-green-500 text-[10px] font-black text-white rounded-br-lg shadow-2xl uppercase tracking-wider z-20">Free</div>)}
        {game.is_in_library && (<div className={`absolute ${game.is_free ? 'top-6' : 'top-0'} left-0 px-2 py-1 bg-blue-600 text-[10px] font-black text-white rounded-br-lg shadow-2xl uppercase tracking-wider z-20`}>In Library</div>)}
        {game.price && (<div className="absolute bottom-2 right-2 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-[10px] font-bold text-white border border-white/20 z-10">{game.price}</div>)}
        {isSeed && (<div className="absolute top-2 left-2 bg-primary px-2 py-1 rounded text-[10px] font-bold text-primary-foreground shadow-lg z-10 flex items-center gap-1 uppercase tracking-wider"><Sparkles size={10} fill="currentColor" />Seed Game</div>)}
        {isNSFW && !isSeed && (<div className={`absolute top-2 left-2 px-2 py-1 rounded text-[10px] font-bold border z-10 flex items-center gap-1 ${shouldBlur ? 'bg-orange-600/80 text-white border-orange-400/50' : 'bg-black/60 backdrop-blur-md text-orange-500 border-orange-500/30'}`}>NSFW</div>)}
        {shouldBlur && (<div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none"><div className="bg-black/40 backdrop-blur-sm px-4 py-2 rounded-full border border-white/10"><span className="text-[10px] font-bold uppercase tracking-widest text-white/80">Content Hidden</span></div></div>)}
      </div>
      <div className="p-4 flex flex-col flex-grow">
        <div className="flex justify-between items-start mb-2 gap-2"><h3 className="text-base font-bold line-clamp-1 text-foreground group-hover:text-primary transition-colors flex-grow">{game.name}</h3><div className="text-muted-foreground group-hover:text-primary transition-colors shrink-0"><ExternalLink size={16} /></div></div>
        <div className="min-h-[2.5rem] mb-2"><p className="text-xs text-muted-foreground line-clamp-2 flex-grow">{game.short_description}</p></div>
        {allGenres.length > 0 && (<div className="flex flex-wrap gap-1 mb-1 h-5 overflow-hidden content-start">{allGenres.slice(0, 3).map((genre) => renderLinkableTerm(genre, true))}</div>)}
        {allTags.length > 0 && (<div className="flex flex-wrap gap-1 mb-4 h-5 overflow-hidden content-start">{allTags.slice(0, 5).map((tag) => renderLinkableTerm(tag, false))}</div>)}
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[10px] mt-auto border-t border-border/30 pt-3">
          <div className="flex items-center gap-1.5 text-muted-foreground"><Star size={12} className="text-yellow-500" /><span>{game.match_percent !== undefined ? `${Math.round(game.match_percent)}% Match` : 'No Match Data'}</span></div>
          <div className="flex items-center gap-1.5 text-muted-foreground"><Clock size={12} className="text-blue-500" /><span>{Math.round(game.estimated_playtime / 60)}h</span></div>
          <div className="flex items-center gap-1.5 text-muted-foreground"><Trophy size={12} className="text-orange-500" /><span>Diff: {game.difficulty_predicted?.toFixed(1) || '0.0'}</span></div>
          <div className="flex items-center gap-1.5 text-muted-foreground"><span className="bg-secondary px-1.5 py-0.5 rounded text-[9px] uppercase font-bold tracking-wider">ID: {game.appid}</span></div>
        </div>
        {game.weighted_score !== undefined && game.weighted_score !== null && !isSeed && (<div className="mt-4 pt-3 border-t border-border flex justify-between items-center"><span className="text-[9px] font-bold uppercase text-muted-foreground tracking-widest">Match Score</span><span className="text-primary font-bold text-sm">{(game.weighted_score || 0).toFixed(2)}</span></div>)}
      </div>
    </div>
  );
};

export default GameCard;
