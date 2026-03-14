import React from 'react';
import type { GameMetadata, GameStatus } from '../types';
import { 
  Plus, 
  CheckCircle2, 
  Clock, 
  Trophy, 
  TrendingUp,
  AlertCircle,
  EyeOff,
  Star,
  BookOpen,
  Heart
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useContextMenu } from '../context/ContextMenuContext';
import GameHeaderImage from './GameHeaderImage';

interface GameCardProps {
  game: GameMetadata;
  hideNSFW?: boolean;
  isSeed?: boolean;
  onStatusUpdate?: (appid: number, status: GameStatus) => void;
  onMouseEnter?: (e: React.MouseEvent, game: any) => void;
  onMouseLeave?: () => void;
}

const GameCard: React.FC<GameCardProps> = ({ 
  game, 
  hideNSFW = true, 
  isSeed = false, 
  onStatusUpdate,
  onMouseEnter,
  onMouseLeave
}) => {
  const { showContextMenu } = useContextMenu();

  // Find current status (placeholder for now as library is managed in parents)
  const status: GameStatus = 'none';

  const handleStatusClick = (e: React.MouseEvent, newStatus: GameStatus) => {
    e.stopPropagation();
    if (onStatusUpdate) onStatusUpdate(game.appid, newStatus);
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    showContextMenu({
      x: e.clientX,
      y: e.clientY,
      appid: game.appid,
      steamId: "", 
      currentStatus: status,
      onUpdate: (aid, newStatus) => {
        if (onStatusUpdate) onStatusUpdate(aid, newStatus as GameStatus);
      }
    });
  };

  const getStatusIcon = (s: GameStatus) => {
    switch (s) {
      case 'backlog': return <BookOpen size={14} />;
      case 'played': return <CheckCircle2 size={14} />;
      case 'rated': return <Star size={14} />;
      case 'ignored': return <EyeOff size={14} />;
      case 'wishlist': return <Heart size={14} />;
      default: return <Plus size={14} />;
    }
  };

  const getStatusColor = (s: GameStatus) => {
    switch (s) {
      case 'backlog': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'played': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
      case 'rated': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'ignored': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'wishlist': return 'bg-pink-500/20 text-pink-400 border-pink-500/30';
      default: return 'bg-secondary/80 text-muted-foreground border-border/50 hover:border-primary/50';
    }
  };

  return (
    <motion.div
      onPointerEnter={(e) => {
        onMouseEnter?.(e as any, game);
      }}
      onPointerLeave={() => {
        onMouseLeave?.();
      }}
      onContextMenu={handleContextMenu}
      className={`relative bg-card border-2 rounded-2xl overflow-hidden shadow-lg transition-all group/card cursor-pointer h-full flex flex-col ${
        isSeed ? 'border-primary/40 ring-2 ring-primary/20' : 'border-border/50 hover:border-primary/30'
      }`}
      onClick={() => window.open(`https://store.steampowered.com/app/${game.appid}`, '_blank')}
    >
      {/* Header Image */}
      <div className="relative aspect-video overflow-hidden bg-secondary">
        <GameHeaderImage 
          appid={game.appid} 
          header_image={game.header_image} 
          isNSFW={game.is_nsfw}
          blurNSFW={hideNSFW}
          className="w-full h-full object-cover transition-transform duration-700 group-hover/card:scale-110"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60" />
        
        {/* Quality Score Badge */}
        <div className="absolute top-3 right-3 px-2 py-1 bg-black/60 backdrop-blur-md border border-white/10 rounded-lg flex items-center gap-1.5 shadow-2xl">
          <Trophy size={12} className="text-yellow-400" />
          <span className="text-[10px] font-mono font-black text-white">
            {Math.round((game.quality_score || 0) * 100)}
          </span>
        </div>

        {/* Action Buttons Overlay */}
        <div className="absolute bottom-3 left-3 right-3 flex justify-between items-center translate-y-2 opacity-0 group-hover/card:translate-y-0 group-hover/card:opacity-100 transition-all duration-300">
          <div className="flex gap-2">
            {(['backlog', 'wishlist', 'played'] as GameStatus[]).map((s) => (
              <button
                key={s}
                onClick={(e) => handleStatusClick(e, s)}
                className={`p-2 rounded-lg backdrop-blur-md border transition-all ${
                  status === s ? 'bg-primary text-primary-foreground border-primary shadow-lg scale-110' : 'bg-black/40 text-white/80 border-white/10 hover:bg-black/60'
                }`}
                title={`Mark as ${s}`}
              >
                {getStatusIcon(s)}
              </button>
            ))}
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); handleContextMenu(e); }}
            className="p-2 bg-black/40 text-white/80 backdrop-blur-md border border-white/10 rounded-lg hover:bg-black/60 transition-all"
          >
            <Plus size={16} />
          </button>
        </div>

        {isSeed && (
          <div className="absolute top-3 left-3 px-2 py-1 bg-primary text-primary-foreground text-[8px] font-black uppercase tracking-tighter rounded shadow-lg flex items-center gap-1">
            <TrendingUp size={10} />
            Seed Source
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4 flex flex-col flex-grow bg-gradient-to-b from-card to-background">
        <div className="mb-3">
          <h3 className="font-bold text-base leading-tight group-hover/card:text-primary transition-colors line-clamp-2 min-h-[2.5rem]">
            {game.name}
          </h3>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="text-[9px] font-mono text-muted-foreground bg-secondary/50 px-1.5 py-0.5 rounded border border-border/30">
              {game.appid}
            </span>
            <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border transition-colors ${getStatusColor(status)}`}>
              {status}
            </span>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-2 mt-auto">
          <div className="bg-secondary/30 rounded-xl p-2 flex items-center gap-2 border border-border/20">
            <Clock size={12} className="text-primary/70" />
            <div className="flex flex-col">
              <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter leading-none">Length</span>
              <span className="text-[10px] font-bold text-foreground">{(((game as any).playtime || game.estimated_playtime || 0) / 60).toFixed(1)}h</span>
            </div>
          </div>
          <div className="bg-secondary/30 rounded-xl p-2 flex items-center gap-2 border border-border/20">
            <AlertCircle size={12} className="text-primary/70" />
            <div className="flex flex-col">
              <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter leading-none">Diff</span>
              <span className="text-[10px] font-bold text-foreground">{(game.difficulty_predicted || 0).toFixed(1)}</span>
            </div>
          </div>
        </div>

        {/* Match Confidence (If available) */}
        {game.predicted_rating !== undefined && (
          <div className="mt-3 pt-3 border-t border-border/30 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Match Affinity</span>
            </div>
            <span className="text-xs font-mono font-bold text-primary">
              {Math.round(game.predicted_rating * 10)}%
            </span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default GameCard;
