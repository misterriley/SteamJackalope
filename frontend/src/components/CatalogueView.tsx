import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { 
  Trash2,
  RefreshCw,
  ExternalLink,
  Library,
  BookOpen,
  CheckCircle2,
  Star,
  EyeOff,
  Heart
} from 'lucide-react';
import { getMetadata, API_BASE_URL } from '../api';
import { useContextMenu, ContextMenuProvider } from '../context/ContextMenuContext';
import { useUser } from '../context/UserContext';
import { type GameStatus, type GameMetadata } from '../types';

import GameAddControl from './GameAddControl';
import GameHeaderImage from './GameHeaderImage';
import GameHoverCard from './GameHoverCard';

interface CatalogueEntry extends GameMetadata {
  appid: number;
  name: string;
  predicted_rating: number;
  actual_rating: number;
  ignore: boolean;
  status: GameStatus;
  playtime_forever: number;
  header_image?: string;
  is_manual?: boolean;
  is_nsfw?: boolean;
  is_free?: boolean;
  price?: string;
  notes?: string;
}

const ROWS_PER_PAGE = 50;

const getStatusIcon = (status: GameStatus) => {
  switch (status) {
    case 'backlog': return <BookOpen size={14} />;
    case 'played': return <CheckCircle2 size={14} />;
    case 'rated': return <Star size={14} />;
    case 'ignored': return <EyeOff size={14} />;
    case 'wishlist': return <Heart size={14} />;
    default: return null;
  }
};

interface CatalogueRowProps {
  entry: CatalogueEntry;
  blurNSFW: boolean;
  onStatusChange: (appid: number, status: GameStatus) => void;
  onRatingChange: (appid: number, rating: number) => void;
  onDelete: (appid: number) => void;
  onContextMenu: (e: React.MouseEvent, appid: number) => void;
  onMouseEnter: (e: React.MouseEvent, game: any) => void;
  onMouseLeave: () => void;
}

const CatalogueRow = React.memo(({ entry, blurNSFW, onStatusChange, onRatingChange, onDelete, onContextMenu, onMouseEnter, onMouseLeave }: CatalogueRowProps) => {
  const isRated = entry.status === 'rated';
  return (
    <tr 
      onContextMenu={(e) => onContextMenu(e, entry.appid)}
      onMouseEnter={(e) => onMouseEnter(e, entry)}
      onMouseLeave={onMouseLeave}
      className={`hover:bg-secondary/30 transition-colors ${entry.status === 'ignored' ? 'opacity-40' : ''}`}
    >
      <td className="px-6 py-3">
        <div className="block hover:opacity-80 transition-opacity relative">
          <GameHeaderImage appid={entry.appid} header_image={entry.header_image} isNSFW={entry.is_nsfw} blurNSFW={blurNSFW} className="w-16 h-8 object-cover rounded shadow-sm border border-border/50" alt={entry.name} />
        </div>
      </td>
      <td className="px-6 py-3"><div className="flex flex-col"><div className="font-bold text-sm leading-tight">{entry.name}</div><div className="flex items-center gap-2 mt-1"><span className="text-[10px] text-muted-foreground font-mono">AppID: {entry.appid}</span>{entry.playtime_forever > 0 && <span className="text-[10px] text-primary/70 font-bold bg-primary/5 px-1.5 rounded">{(entry.playtime_forever / 60).toFixed(1)}h</span>}</div></div></td>
      <td className="px-6 py-3"><div className="flex items-center bg-secondary/50 rounded-lg p-1 w-fit">{(['backlog', 'played', 'rated', 'ignored', 'wishlist'] as GameStatus[]).map((s) => (<button key={s} onClick={(e) => { e.stopPropagation(); onStatusChange(entry.appid, s); }} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${entry.status === s ? 'bg-primary text-primary-foreground shadow-md' : 'text-muted-foreground hover:text-foreground'}`}>{getStatusIcon(s)}<span className={entry.status === s ? 'inline' : 'hidden md:inline'}>{s}</span></button>))}</div></td>
      <td className="px-6 py-3 min-w-[160px]"><div className={`flex items-center gap-3 transition-all ${!isRated ? 'opacity-20 pointer-events-none grayscale' : ''}`} onClick={(e) => e.stopPropagation()}><input type="range" min="0" max="10" step="1" disabled={!isRated} className="w-full accent-primary h-1 cursor-pointer" value={entry.actual_rating} onChange={(e) => onRatingChange(entry.appid, parseInt(e.target.value))} /><span className="w-4 text-xs font-bold text-primary text-center">{Math.round(entry.actual_rating)}</span></div></td>
      <td className="px-6 py-3 text-center"><div className="flex items-center justify-center gap-2"><a href={`https://store.steampowered.com/app/${entry.appid}`} target="_blank" rel="noopener noreferrer" className="p-2 text-muted-foreground hover:text-primary"><ExternalLink size={16} /></a><button onClick={(e) => { e.stopPropagation(); onDelete(entry.appid); }} className="p-2 text-muted-foreground hover:text-destructive"><Trash2 size={16} /></button></div></td>
    </tr>
  );
});

const CatalogueViewContent: React.FC = () => {
  const [steamId] = useState<string>(() => {
    const saved = sessionStorage.getItem('personalization_state');
    if (saved) { try { return JSON.parse(saved).steamId || ''; } catch (e) { return ''; } }
    return '';
  });

  const [loading, setLoading] = useState(false);
  const [entries, setEntries] = useState<CatalogueEntry[]>([]);
  const [visibleCount, setVisibleCount] = useState(ROWS_PER_PAGE);
  const [isSaving, setIsSaving] = useState(false);
  
  // Sticky State Machine
  const [hoverState, setHoverState] = useState<{ game: any, anchor: DOMRect } | null>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isOverSystemRef = useRef<boolean>(false);
  const activeAppidRef = useRef<number | null>(null);

  const handleGameMouseEnter = (e: React.MouseEvent, game: any) => {
    isOverSystemRef.current = true;
    
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }

    if (activeAppidRef.current === game.appid) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const delay = activeAppidRef.current ? 0 : 150; 
    
    const mount = () => {
      activeAppidRef.current = game.appid;
      setHoverState({ game, anchor: rect });
    };

    if (delay === 0) mount();
    else hoverTimerRef.current = setTimeout(mount, delay);
  };

  const handleGameMouseLeave = () => {
    isOverSystemRef.current = false;
    
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    
    hoverTimerRef.current = setTimeout(() => {
      if (!isOverSystemRef.current) {
        activeAppidRef.current = null;
        setHoverState(null);
      }
    }, 800);
  };

  const handleCardMouseEnter = () => {
    isOverSystemRef.current = true;
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
  };

  const handleCardMouseLeave = () => {
    isOverSystemRef.current = false;
    handleGameMouseLeave();
  };

  const STATUS_PRIORITY: Record<GameStatus, number> = { 'backlog': 0, 'wishlist': 1, 'rated': 2, 'played': 3, 'ignored': 4, 'none': 5, 'deleted': 6 };
  const { showContextMenu } = useContextMenu();
  const { steamId: globalSteamId, setSteamId: globalSetSteamId } = useUser();

  const handleStatusChange = useCallback((appid: number, status: GameStatus) => {
    setEntries(prev => prev.map(entry => {
      if (entry.appid === appid) {
        let ignore = entry.ignore;
        if (status === 'ignored') ignore = true;
        else if (['rated', 'played', 'backlog', 'wishlist'].includes(status)) ignore = false;
        return { ...entry, status, ignore };
      }
      return entry;
    }));
  }, []);

  const handleRatingChange = useCallback((appid: number, rating: number) => {
    setEntries(prev => prev.map(entry => entry.appid === appid ? { ...entry, actual_rating: rating, status: 'rated' as GameStatus, ignore: false } : entry));
  }, []);

  const handleContextMenu = useCallback((e: React.MouseEvent, appid: number) => {
    e.preventDefault();
    showContextMenu({
      x: e.clientX, y: e.clientY, appid, steamId: globalSteamId || "",
      onUpdate: (aid, status, rating) => {
        if (status === 'deleted') setEntries(prev => prev.filter(e => e.appid !== aid));
        else if (status === 'rated' && rating !== undefined) handleRatingChange(aid, rating);
        else handleStatusChange(aid, status as GameStatus);
      }
    });
  }, [globalSteamId, showContextMenu, handleStatusChange, handleRatingChange]);

  useEffect(() => { if (steamId && steamId.length >= 17 && steamId !== globalSteamId) globalSetSteamId(steamId); }, [steamId, globalSteamId, globalSetSteamId]);
  useEffect(() => {
    const handleScroll = () => { if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 500) setVisibleCount(prev => prev + ROWS_PER_PAGE); };
    window.addEventListener('scroll', handleScroll); return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  useEffect(() => { if (steamId) fetchCatalogue(); }, [steamId]);

  const fetchCatalogue = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/user/catalogue/${steamId}`);
      if (!res.ok) throw new Error("Failed to fetch catalogue.");
      const data: CatalogueEntry[] = await res.json();
      setEntries(data.sort((a, b) => (STATUS_PRIORITY[a.status] ?? 99) - (STATUS_PRIORITY[b.status] ?? 99)));
    } catch (err: any) { console.error(err.message); } finally { setLoading(false); }
  };

  const handleManualAdd = async (gameName: string) => {
    try {
      const meta = await getMetadata([gameName]);
      if (meta && meta.length > 0) {
        const game = meta[0];
        if (entries.some(e => e.appid === game.appid)) return alert("Game already in catalogue!");
        setEntries(prev => [{ appid: game.appid, name: game.name, predicted_rating: 5, actual_rating: 5, ignore: true, status: 'backlog', playtime_forever: 0, is_manual: true, is_nsfw: game.is_nsfw } as CatalogueEntry, ...prev]);
      }
    } catch (err) {}
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const updates = entries.map(e => ({ steam_id: steamId, appid: e.appid, actual_rating: e.actual_rating, ignore: e.ignore, status: e.status, notes: e.notes }));
      await fetch(`${API_BASE_URL}/user/verify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(updates) });
      alert("Catalogue saved!");
    } catch (err: any) { alert("Error saving: " + err.message); } finally { setIsSaving(false); }
  };

  const visibleEntries = useMemo(() => entries.slice(0, visibleCount), [entries, visibleCount]);

  if (!steamId) return <div className="py-20 text-center"><Library size={40} className="mx-auto text-muted-foreground mb-4" /><h2>No Library Loaded</h2></div>;

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20 relative">
      <div className="sticky top-16 z-40 bg-background/95 backdrop-blur-xl py-6 border-b border-border/50 flex justify-between items-center px-4">
        <h2 className="text-3xl font-bold">Catalogue</h2>
        <div className="flex gap-3"><GameAddControl onAdd={handleManualAdd} /><button onClick={handleSave} className="bg-primary text-primary-foreground px-6 py-2 rounded-xl font-bold">{isSaving ? 'Saving...' : 'Save'}</button></div>
      </div>
      {loading ? <div className="py-40 text-center"><RefreshCw className="animate-spin text-primary mx-auto" size={48} /></div> : (
        <div className="bg-card border border-border rounded-2xl overflow-hidden shadow-xl">
          <table className="w-full text-left">
            <thead className="bg-secondary/50 border-b border-border"><tr><th className="px-6 py-4">Img</th><th className="px-6 py-4">Game</th><th className="px-6 py-4">Status</th><th className="px-6 py-4">Rating</th><th className="px-6 py-4 text-center">Action</th></tr></thead>
            <tbody className="divide-y divide-border">{visibleEntries.map((entry) => (<CatalogueRow key={entry.appid} entry={entry} blurNSFW={true} onStatusChange={handleStatusChange} onRatingChange={handleRatingChange} onDelete={(aid) => setEntries(prev => prev.filter(e => e.appid !== aid))} onContextMenu={handleContextMenu} onMouseEnter={handleGameMouseEnter} onMouseLeave={handleGameMouseLeave} />))}</tbody>
          </table>
        </div>
      )}
      <GameHoverCard 
        game={hoverState?.game || null} 
        anchorRect={hoverState?.anchor} 
        onMouseEnter={handleCardMouseEnter}
        onMouseLeave={handleCardMouseLeave}
      />
    </div>
  );
};

export default function CatalogueView() { return <ContextMenuProvider><CatalogueViewContent /></ContextMenuProvider>; }
