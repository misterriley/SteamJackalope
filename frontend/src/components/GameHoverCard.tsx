import React, { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { getGameMedia } from '../api';
import type { GameMetadata } from '../types';
import { 
  Clock, Trophy, Banknote, Loader2, AlertCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface GameHoverCardProps {
  game: GameMetadata | null;
  anchorRect?: DOMRect;
  weights?: { [key: string]: number };
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

const GameHoverCard: React.FC<GameHoverCardProps> = ({ 
  game, anchorRect, weights, onMouseEnter, onMouseLeave 
}) => {
  const [media, setMedia] = useState<{ screenshots: string[], movies: {url: string, poster: string}[], error?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const activeAppidRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const mediaList = useMemo(() => {
    if (!media) return [];
    const movies = (media.movies || []).map(m => ({ type: 'video' as const, url: m.url, poster: m.poster }));
    const screenshots = (media.screenshots || []).map(url => ({ type: 'image' as const, url, poster: undefined }));
    
    if (screenshots.length > 0) {
      return [
        screenshots[0],
        ...movies,
        ...screenshots.slice(1)
      ];
    }
    return movies;
  }, [media]);

  const mediaListRef = useRef(mediaList);
  useEffect(() => { mediaListRef.current = mediaList; }, [mediaList]);

  const handleNext = useCallback(() => {
    const list = mediaListRef.current;
    if (list.length > 1) {
      setCurrentIndex(prev => (prev + 1) % list.length);
    }
  }, []);

  // Heartbeat to lock state
  useLayoutEffect(() => {
    if (game?.appid) {
      onMouseEnter?.();
    }
  }, [game?.appid, onMouseEnter]);

  useEffect(() => {
    if (!game?.appid) {
      setMedia(null);
      setLoading(false);
      activeAppidRef.current = null;
      return;
    }

    if (activeAppidRef.current === game.appid) return;
    
    activeAppidRef.current = game.appid;
    let isMounted = true;
    
    setCurrentIndex(0);
    setLoading(true);

    getGameMedia(game.appid)
      .then(data => {
        if (isMounted && activeAppidRef.current === game.appid) {
          setMedia(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (isMounted && activeAppidRef.current === game.appid) {
          setMedia({ screenshots: [], movies: [], error: String(err) });
          setLoading(false);
        }
      });

    return () => { isMounted = false; };
  }, [game?.appid]);

  useEffect(() => {
    if (!game || mediaList.length <= 1) return;
    const currentMedia = mediaList[currentIndex];
    if (!currentMedia) return;

    if (timerRef.current) clearTimeout(timerRef.current);

    if (currentMedia.type === 'image') {
      timerRef.current = setTimeout(handleNext, 2000);
    } else {
      timerRef.current = setTimeout(handleNext, 10000);
    }
    
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [game?.appid, currentIndex, mediaList.length, handleNext]);

  const getPositionData = () => {
    const cardWidth = 380;
    const cardHeight = 540; 
    const overlap = 5; // 5px overlap to guarantee hit
    if (!anchorRect) return { left: 0, top: 0 };
    
    const spaceOnRight = window.innerWidth - anchorRect.right;
    const showOnRight = spaceOnRight > (cardWidth + 40);
    
    let left = showOnRight ? anchorRect.right - overlap : anchorRect.left - cardWidth + overlap;
    let top = anchorRect.top + (anchorRect.height / 2) - (cardHeight / 2);
    top = Math.max(80, Math.min(window.innerHeight - cardHeight - 20, top));
    
    return { left, top };
  };

  const { left, top } = getPositionData();

  const renderContributions = () => {
    if (!game) return null;
    const features = (game as any).features || {};
    const getW = (key: string, backup?: number) => weights ? (weights[key] || 0) : (backup || 0);
    const zKeyMap: Record<string, string> = { quality: 'z_spps', age: 'z_date', difficulty: 'z_difficulty', price: 'z_price', length: 'z_length', popularity: 'z_pop', tone: 'z_tone' };
    const getZ = (key: string) => {
        const zKey = zKeyMap[key];
        const val = (game as any)[zKey];
        if (val !== undefined && val !== 0) return val;
        if (features[key] !== undefined && features[key] !== 0) return features[key];
        return 0;
    };

    const contributors = [
      { key: 'quality', label: 'Quality', val: getZ('quality') * getW('quality', (game as any).w_spps), color: 'text-blue-400' },
      { key: 'age', label: 'Release Date', val: getZ('age') * getW('age', (game as any).w_date), color: 'text-cyan-400' },
      { key: 'difficulty', label: 'Difficulty', val: getZ('difficulty') * getW('difficulty', (game as any).w_difficulty), color: 'text-orange-400' },
      { key: 'price', label: 'Price', val: getZ('price') * getW('price', (game as any).w_price), color: 'text-yellow-400' },
      { key: 'length', label: 'Length', val: getZ('length') * getW('length', (game as any).w_length), color: 'text-purple-400' },
      { key: 'popularity', label: 'Popularity', val: getZ('popularity') * getW('popularity', (game as any).w_pop), color: 'text-pink-400' },
      { key: 'tone', label: 'Tone', val: getZ('tone') * getW('tone', (game as any).w_tone), color: 'text-indigo-400' },
      { key: 'kernel', label: 'Kernel Strength', val: ((game as any).kernel_residual || 0) * getW('kernel', 1.0), color: 'text-emerald-400' }
    ].filter(c => Math.abs(c.val || 0) > 0.01).sort((a, b) => Math.abs(b.val!) - Math.abs(a.val!));

    const totalScore = (game as any).current_score ?? (game as any).predicted_rating ?? (game as any).weighted_score;

    return (
      <div className="mt-4 pt-3 border-t border-border/50">
        <div className="flex justify-between items-center mb-2">
          <h4 className="text-[10px] font-bold text-primary uppercase tracking-widest">Score Breakdown</h4>
          {totalScore !== undefined && (
            <span className="text-[10px] font-mono font-bold text-primary px-2 py-0.5 bg-primary/10 rounded border border-primary/20">
              SCORE: {typeof totalScore === 'number' ? totalScore.toFixed(2) : totalScore}
            </span>
          )}
        </div>
        {contributors.length > 0 ? (
          <div className="space-y-1.5">
            {contributors.map(c => (
              <div key={c.key} className="flex items-center gap-2 text-[10px]">
                <span className={`w-20 truncate font-bold ${c.color}`}>{c.label}</span>
                <div className="flex-grow h-1 bg-secondary/50 rounded-full overflow-hidden relative border border-border/10">
                  <div className={`absolute inset-y-0 ${c.val! >= 0 ? 'bg-primary left-1/2' : 'bg-destructive right-1/2'}`} style={{ width: `${Math.min(50, (Math.abs(c.val!) / 4) * 50)}%` }} />
                </div>
                <span className={`w-8 text-right font-mono ${c.val! >= 0 ? 'text-primary' : 'text-destructive'}`}>{c.val! > 0 ? '+' : ''}{c.val!.toFixed(2)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-2 bg-secondary/20 p-3 rounded-lg border border-border/30 text-[9px] text-muted-foreground italic">
            <div className="flex items-center gap-2 font-bold text-orange-500/80 uppercase"><AlertCircle size={14} /><span>Context Unavailable</span></div>
            <p>Score details only available for solved Taste DNA profiles.</p>
          </div>
        )}
      </div>
    );
  };

  const currentMedia = mediaList[currentIndex];

  const content = (
    <AnimatePresence mode="wait">
      {game && anchorRect && (
        <motion.div 
          key={`hover-card-${game.appid}`}
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.15 }}
          onPointerEnter={onMouseEnter}
          onPointerLeave={onMouseLeave}
          onPointerMove={onMouseEnter}
          className="fixed z-[999999] w-[380px] bg-card border-2 border-primary/50 rounded-2xl shadow-[0_40px_80px_-15px_rgba(0,0,0,0.9)] overflow-hidden pointer-events-auto"
          style={{ left, top }}
        >
          {/* Hitbox Padding Buffer */}
          <div className="absolute -inset-4 z-[-1] pointer-events-auto" onPointerEnter={onMouseEnter} />

          <div className="relative w-full bg-black flex items-center justify-center border-b border-border/50 overflow-hidden h-[214px] min-h-[214px]">
            {loading && !media ? (
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="animate-spin text-primary" size={24} />
                <span className="text-[10px] font-bold text-primary/50 uppercase tracking-widest">Loading Media...</span>
              </div>
            ) : currentMedia ? (
              <>
                <div className="absolute top-2 right-2 z-20 px-2 py-0.5 bg-black/60 backdrop-blur-md border border-white/10 rounded text-[9px] font-mono font-bold text-white/80">
                  {currentIndex + 1} / {mediaList.length}
                </div>
                
                {currentMedia.type === 'video' ? (
                  <video
                    ref={videoRef}
                    key={`${game.appid}-v-${currentIndex}`}
                    autoPlay
                    muted
                    playsInline
                    preload="auto"
                    onEnded={handleNext}
                    onPlay={() => {
                        if (timerRef.current) clearTimeout(timerRef.current);
                    }}
                    onError={() => {
                      console.warn(`[HoverCard] Video failed to load at index ${currentIndex}, skipping in 1s...`);
                      setTimeout(handleNext, 1000);
                    }}
                    className="w-full h-full object-contain"
                    poster={currentMedia.poster || game.header_image}
                  >
                    <source src={currentMedia.url} type="video/mp4" />
                  </video>
                ) : (
                  <img 
                    key={`${game.appid}-i-${currentIndex}`}
                    src={currentMedia.url} 
                    className="w-full h-full object-contain" 
                    alt="Screenshot" 
                    onError={handleNext}
                  />
                )}
              </>
            ) : (
              <div className="relative w-full h-full">
                <img src={game.header_image} className="w-full h-full object-cover opacity-40 blur-[1px]" alt="Placeholder" />
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40 p-4 text-center">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-white/60 border border-white/10 px-3 py-1 rounded-full">
                    {media?.error ? `Error: ${media.error}` : 'Static Preview'}
                  </span>
                </div>
              </div>
            )}

            {!loading && mediaList.length > 1 && (
              <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1 z-10">
                {mediaList.map((_, i) => (
                  <div key={i} className={`h-1 rounded-full transition-all ${i === currentIndex ? 'w-4 bg-primary' : 'w-1 bg-white/20'}`} />
                ))}
              </div>
            )}
          </div>

          <div className="p-4 bg-gradient-to-b from-card to-background">
            <div className="mb-4">
              <h3 className="font-bold text-xl leading-tight text-foreground truncate">{game.name}</h3>
              <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-tighter mt-1 opacity-70">AppID: {game.appid} • {(game as any).release_year || 'TBA'}</p>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-secondary/40 border border-border/40 rounded-xl p-2 flex flex-col items-center justify-center text-center">
                <div className="text-primary/70 mb-1"><Banknote size={12} /></div>
                <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Price</span>
                <span className="text-xs font-bold truncate w-full text-foreground">{(game as any).raw_price || (game as any).price || 'N/A'}</span>
              </div>
              <div className="bg-secondary/40 border border-border/40 rounded-xl p-2 flex flex-col items-center justify-center text-center">
                <div className="text-primary/70 mb-1"><Trophy size={12} /></div>
                <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Diff</span>
                <span className="text-xs font-bold text-foreground">{((game as any).raw_difficulty ?? game.difficulty_predicted ?? 0).toFixed(1)}</span>
              </div>
              <div className="bg-secondary/40 border border-border/40 rounded-xl p-2 flex flex-col items-center justify-center text-center">
                <div className="text-primary/70 mb-1"><Clock size={12} /></div>
                <span className="text-[9px] text-muted-foreground uppercase font-black tracking-tighter">Length</span>
                <span className="text-xs font-bold text-foreground">{(((game as any).raw_length ?? (game.estimated_playtime / 60)) || 0).toFixed(1)}h</span>
              </div>
            </div>
            {renderContributions()}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return createPortal(content, document.body);
};

export default GameHoverCard;
