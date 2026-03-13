import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { getGameMedia } from '../api';
import { GameMetadata } from '../types';
import { 
  Volume2, VolumeX, ChevronLeft, ChevronRight, 
  Clock, Trophy, Banknote, Loader2 
} from 'lucide-react';

interface GameHoverCardProps {
  game: GameMetadata;
  isVisible: boolean;
  anchorRect?: DOMRect;
  weights?: { [key: string]: number }; // Optional weights for live UI updates
}

const GameHoverCard: React.FC<GameHoverCardProps> = ({ game, isVisible, anchorRect, weights }) => {
  const [media, setMedia] = useState<{ screenshots: string[], movies: string[], error?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isMuted, setIsMuted] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Combine media into a single list
  const mediaList = useMemo(() => {
    if (!media) return [];
    const list = [
      ...(media.movies || []).map(url => ({ type: 'video' as const, url })), 
      ...(media.screenshots || []).map(url => ({ type: 'image' as const, url }))
    ];
    return list;
  }, [media]);

  const handleNext = useCallback(() => {
    if (mediaList.length > 0) {
      setCurrentIndex(prev => (prev + 1) % mediaList.length);
    }
  }, [mediaList.length]);

  const handlePrev = useCallback(() => {
    if (mediaList.length > 0) {
      setCurrentIndex(prev => (prev - 1 + mediaList.length) % mediaList.length);
    }
  }, [mediaList.length]);

  useEffect(() => {
    if (isVisible) {
      if (!media) {
        setLoading(true);
        getGameMedia(game.appid)
          .then(data => {
            setMedia(data);
            setLoading(false);
          })
          .catch(err => {
            setMedia({ screenshots: [], movies: [], error: String(err) });
            setLoading(false);
          });
      }
    }
  }, [isVisible, game.appid, media]);

  useEffect(() => {
    if (!isVisible) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }

    const currentMedia = mediaList[currentIndex];
    if (!currentMedia) return;

    if (currentMedia.type === 'image') {
      timerRef.current = setTimeout(handleNext, 2500);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isVisible, currentIndex, mediaList, handleNext]);

  if (!isVisible || !anchorRect) return null;

  // Calculate position: prefer right of the card, fallback to left
  const spaceOnRight = window.innerWidth - anchorRect.right;
  const showOnRight = spaceOnRight > 400;
  const left = showOnRight ? anchorRect.right + 15 : Math.max(10, anchorRect.left - 395);
  const top = Math.max(10, Math.min(window.innerHeight - 550, anchorRect.top));

  const currentMedia = mediaList[currentIndex];

  const renderContributions = () => {
    const features = game.features || {};
    const getW = (key: string, backup?: number) => weights ? (weights[key] || 0) : (backup || 0);

    const contributors = [
      { key: 'quality', label: 'Quality', val: (game.z_spps ?? features.quality ?? 0) * getW('quality', game.w_spps), color: 'text-blue-400' },
      { key: 'age', label: 'Release Date', val: (game.z_date ?? features.age ?? 0) * getW('age', game.w_date), color: 'text-cyan-400' },
      { key: 'difficulty', label: 'Difficulty', val: (game.z_difficulty ?? features.difficulty ?? 0) * getW('difficulty', game.w_difficulty), color: 'text-orange-400' },
      { key: 'price', label: 'Price', val: (game.z_price ?? features.price ?? 0) * getW('price', game.w_price), color: 'text-yellow-400' },
      { key: 'length', label: 'Length', val: (game.z_length ?? features.length ?? 0) * getW('length', game.w_length), color: 'text-purple-400' },
      { key: 'popularity', label: 'Popularity', val: (game.z_pop ?? features.popularity ?? 0) * getW('popularity', game.w_pop), color: 'text-pink-400' },
      { key: 'tone', label: 'Tone', val: (game.z_tone ?? features.tone ?? 0) * getW('tone', game.w_tone), color: 'text-indigo-400' },
      { key: 'kernel', label: 'Kernel Strength', val: (game.kernel_residual || 0) * getW('kernel', 1.0), color: 'text-emerald-400' }
    ].filter(c => {
      if (c.val === undefined) return false;
      return Math.abs(parseFloat(c.val.toFixed(2))) > 0;
    }).sort((a, b) => Math.abs(b.val!) - Math.abs(a.val!));

    const totalScore = game.weighted_score ?? (game as any).current_score;

    return (
      <div className="mt-4 pt-3 border-t border-border/50">
        <div className="flex justify-between items-center mb-2">
          <h4 className="text-[10px] font-bold text-primary uppercase tracking-widest">Score Breakdown</h4>
          {totalScore !== undefined && (
            <span className="text-[10px] font-mono font-bold text-primary">
              SCORE: {totalScore.toFixed(2)}
            </span>
          )}
        </div>
        
        {contributors.length > 0 ? (
          <div className="space-y-1.5">
            {contributors.map(c => (
              <div key={c.key} className="flex items-center gap-2 text-[10px]">
                <span className={`w-20 truncate font-bold ${c.color}`}>{c.label}</span>
                <div className="flex-grow h-1.5 bg-secondary/50 rounded-full overflow-hidden relative border border-border/20">
                  <div 
                    className={`absolute inset-y-0 ${c.val! >= 0 ? 'bg-primary left-1/2' : 'bg-destructive right-1/2'}`}
                    style={{ width: `${Math.min(50, (Math.abs(c.val!) / 4) * 50)}%` }}
                  />
                </div>
                <span className={`w-8 text-right font-mono ${c.val! >= 0 ? 'text-primary' : 'text-destructive'}`}>
                  {c.val! > 0 ? '+' : ''}{c.val!.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[9px] text-muted-foreground italic">
            {(Object.keys(features).length === 0 && !game.w_spps) 
              ? "No feature data found. Re-solve Taste DNA to see details." 
              : "Weights are currently near zero or non-predictive."}
          </p>
        )}
      </div>
    );
  };

  return (
    <div 
      className="fixed z-[9999] w-[380px] bg-card border border-primary/40 rounded-xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden animate-in fade-in zoom-in duration-200 pointer-events-auto"
      style={{ left, top, minHeight: '400px' }}
    >
      {/* Media Viewport - Forced height for stability */}
      <div 
        className="relative w-full bg-black flex items-center justify-center group/media border-b border-border/50 overflow-hidden"
        style={{ height: '214px' }}
      >
        {loading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="animate-spin text-primary" size={24} />
            <span className="text-[10px] font-bold text-primary/50 uppercase tracking-widest">Fetching Store Media...</span>
          </div>
        ) : currentMedia ? (
          currentMedia.type === 'video' ? (
            <video
              ref={videoRef}
              key={currentMedia.url}
              src={currentMedia.url}
              autoPlay
              muted={isMuted}
              onEnded={handleNext}
              className="w-full h-full object-contain"
            />
          ) : (
            <img 
              key={currentMedia.url}
              src={currentMedia.url} 
              className="w-full h-full object-contain" 
              alt="Screenshot" 
            />
          )
        ) : (
          <div className="relative w-full h-full">
            <img src={game.header_image} className="w-full h-full object-cover opacity-40 blur-[1px]" alt="Placeholder" />
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 p-4 text-center">
               <span className="text-[10px] font-bold uppercase tracking-widest text-white/60 border border-white/10 px-3 py-1 rounded-full mb-2">
                 {media?.error ? 'API Error' : 'No Preview Media'}
               </span>
               {media?.error && (
                 <p className="text-[9px] text-destructive font-mono max-w-[200px] break-words">
                   {media.error}
                 </p>
               )}
               {!media?.error && !loading && (
                 <p className="text-[9px] text-white/40 italic">
                   Steam API returned 0 screenshots/movies for ID {game.appid}.
                 </p>
               )}
            </div>
          </div>
        )}

        {/* Controls Overlay */}
        {!loading && mediaList.length > 1 && (
          <>
            <button 
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handlePrev(); }}
              className="absolute left-2 top-1/2 -translate-y-1/2 bg-black/60 text-white p-1.5 rounded-full opacity-0 group-hover/media:opacity-100 transition-opacity z-10 border border-white/10 hover:bg-primary"
            >
              <ChevronLeft size={18} />
            </button>
            <button 
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleNext(); }}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-black/60 text-white p-1.5 rounded-full opacity-0 group-hover/media:opacity-100 transition-opacity z-10 border border-white/10 hover:bg-primary"
            >
              <ChevronRight size={18} />
            </button>
            
            {currentMedia?.type === 'video' && (
              <button 
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setIsMuted(!isMuted); }}
                className="absolute bottom-2 right-2 bg-black/60 text-white p-1.5 rounded-full hover:bg-primary transition-colors z-10 border border-white/10"
              >
                {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
              </button>
            )}

            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1 z-10">
              {mediaList.map((_, i) => (
                <div 
                  key={i} 
                  className={`h-1 rounded-full transition-all ${i === currentIndex ? 'w-4 bg-primary' : 'w-1 bg-white/20'}`}
                />
              ))}
            </div>
          </>
        )}
      </div>

      <div className="p-4">
        <div className="mb-4">
          <h3 className="font-bold text-lg leading-tight text-foreground line-clamp-1" title={game.name}>
            {game.name}
          </h3>
          <div className="flex items-center gap-2 mt-0.5">
            <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-tighter">
              AppID: {game.appid} • {game.release_year || 'TBA'}
            </p>
            {media?.screenshots && (
              <span className="text-[9px] bg-secondary px-1 rounded text-muted-foreground">
                {media.movies.length}v / {media.screenshots.length}s
              </span>
            )}
          </div>
        </div>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-secondary/30 border border-border/50 rounded-lg p-2 flex flex-col items-center justify-center text-center shadow-inner">
            <Banknote size={12} className="text-primary/70 mb-1" />
            <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Price</span>
            <span className="text-xs font-bold truncate w-full">{game.raw_price || game.price || 'N/A'}</span>
          </div>
          <div className="bg-secondary/30 border border-border/50 rounded-lg p-2 flex flex-col items-center justify-center text-center shadow-inner">
            <Trophy size={12} className="text-primary/70 mb-1" />
            <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Diff</span>
            <span className="text-xs font-bold">{(game.raw_difficulty ?? game.difficulty_predicted ?? 0).toFixed(1)}</span>
          </div>
          <div className="bg-secondary/30 border border-border/50 rounded-lg p-2 flex flex-col items-center justify-center text-center shadow-inner">
            <Clock size={12} className="text-primary/70 mb-1" />
            <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Length</span>
            <span className="text-xs font-bold">{(game.raw_length ?? (game.estimated_playtime / 60) ?? 0).toFixed(1)}h</span>
          </div>
        </div>

        {renderContributions()}
      </div>
    </div>
  );
};

export default GameHoverCard;
