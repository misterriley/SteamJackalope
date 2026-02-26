import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { 
  Search, 
  Table as TableIcon, 
  ChevronDown,
  Trash2,
  Plus,
  RefreshCcw,
  ExternalLink,
  Save,
  Library,
  BookOpen,
  CheckCircle2,
  Star,
  EyeOff,
  Heart,
  ArrowUp,
  RotateCcw
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchGames, getMetadata, API_BASE_URL } from '../api';

type GameStatus = 'backlog' | 'played' | 'rated' | 'ignored' | 'wishlist';

interface CatalogueEntry {
  appid: number;
  name: string;
  predicted_rating: number;
  actual_rating: number;
  ignore: boolean;
  status: GameStatus;
  playtime_forever: number;
  is_manual?: boolean;
  is_nsfw?: boolean;
  notes?: string;
}

const ROWS_PER_PAGE = 50;

// --- Sub-components for Performance ---

const getStatusIcon = (status: GameStatus) => {
  switch (status) {
    case 'backlog': return <BookOpen size={14} />;
    case 'played': return <CheckCircle2 size={14} />;
    case 'rated': return <Star size={14} />;
    case 'ignored': return <EyeOff size={14} />;
    case 'wishlist': return <Heart size={14} />;
  }
};

interface CatalogueRowProps {
  entry: CatalogueEntry;
  blurNSFW: boolean;
  onStatusChange: (appid: number, status: GameStatus) => void;
  onRatingChange: (appid: number, rating: number) => void;
  onDelete: (appid: number) => void;
}

const CatalogueRow = React.memo(({ entry, blurNSFW, onStatusChange, onRatingChange, onDelete }: CatalogueRowProps) => {
  return (
    <tr className={`hover:bg-secondary/30 transition-colors ${entry.status === 'ignored' ? 'opacity-40' : ''}`}>
      <td className="px-6 py-3">
        <a href={`https://store.steampowered.com/app/${entry.appid}`} target="_blank" rel="noopener noreferrer" className="block hover:opacity-80 transition-opacity">
          <img 
            src={`https://cdn.akamai.steamstatic.com/steam/apps/${entry.appid}/header.jpg`} 
            alt={entry.name} 
            className={`w-16 h-8 object-cover rounded shadow-sm border border-border/50 ${entry.is_nsfw && blurNSFW ? 'blur-sm scale-110' : ''}`}
            onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E')}
          />
        </a>
      </td>
      <td className="px-6 py-3">
        <div className="flex flex-col">
          <a 
            href={`https://store.steampowered.com/app/${entry.appid}`} 
            target="_blank" 
            rel="noopener noreferrer"
            className="font-bold text-sm leading-tight hover:text-primary transition-colors w-fit"
          >
            {entry.name}
          </a>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] text-muted-foreground font-mono">AppID: {entry.appid}</span>
            {entry.playtime_forever > 0 && (
              <span className="text-[10px] text-primary/70 font-bold bg-primary/5 px-1.5 rounded">{(entry.playtime_forever / 60).toFixed(1)}h</span>
            )}
            {entry.is_manual && <span className="text-[8px] font-bold uppercase tracking-tighter bg-secondary px-1 rounded text-muted-foreground">Manual</span>}
            {entry.notes && <span className="text-[8px] font-bold uppercase tracking-tighter bg-blue-500/10 text-blue-400 px-1 rounded border border-blue-500/20">{entry.notes}</span>}
          </div>
        </div>
      </td>
      <td className="px-6 py-3">
        <div className="flex items-center bg-secondary/50 rounded-lg p-1 w-fit border border-border/50">
          {(['backlog', 'played', 'rated', 'ignored', 'wishlist'] as GameStatus[]).map((s) => (
            <button
              key={s}
              onClick={() => onStatusChange(entry.appid, s)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
                entry.status === s 
                  ? 'bg-primary text-primary-foreground shadow-md scale-105' 
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`}
              title={s.charAt(0).toUpperCase() + s.slice(1)}
            >
              {getStatusIcon(s)}
              <span className={entry.status === s ? 'inline' : 'hidden md:inline'}>{s}</span>
            </button>
          ))}
        </div>
      </td>
      <td className="px-6 py-3 min-w-[160px]">
        <div className={`flex items-center gap-3 transition-opacity ${entry.status === 'ignored' ? 'opacity-20 pointer-events-none' : ''}`}>
          <input 
            type="range" min="0" max="10" step="1" className="w-full accent-primary h-1 cursor-pointer"
            value={entry.actual_rating}
            onChange={(e) => onRatingChange(entry.appid, parseInt(e.target.value))}
          />
          <span className="w-4 text-xs font-bold text-primary text-center">{Math.round(entry.actual_rating)}</span>
        </div>
      </td>
      <td className="px-6 py-3 text-center">
        <div className="flex items-center justify-center gap-2">
          <a href={`https://store.steampowered.com/app/${entry.appid}`} target="_blank" rel="noopener noreferrer" className="p-2 text-muted-foreground hover:text-primary transition-colors">
            <ExternalLink size={16} />
          </a>
          <button 
            onClick={() => onDelete(entry.appid)}
            className="p-2 text-muted-foreground hover:text-destructive transition-colors"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </td>
    </tr>
  );
});

CatalogueRow.displayName = 'CatalogueRow';

interface ManualAddControlProps {
  onAdd: (gameName: string) => Promise<void>;
}

const ManualAddControl = ({ onAdd }: ManualAddControlProps) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (activeIndex >= 0 && resultsRef.current) {
      const activeItem = resultsRef.current.children[activeIndex] as HTMLElement;
      if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIndex]);

  const handleSearch = (val: string) => {
    setQuery(val);
    setActiveIndex(-1);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    if (val.length > 1) {
      timeoutRef.current = setTimeout(async () => {
        const res = await searchGames(val);
        setResults(res);
        setShowResults(true);
      }, 300);
    } else {
      setResults([]);
      setShowResults(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showResults || results.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex(prev => (prev + 1) % results.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex(prev => (prev - 1 + results.length) % results.length);
    } else if (e.key === 'Enter' && activeIndex >= 0) {
      e.preventDefault();
      onAdd(results[activeIndex]);
      setQuery('');
      setShowResults(false);
    } else if (e.key === 'Escape') {
      setShowResults(false);
    }
  };

  return (
    <div className="relative max-w-xs" ref={containerRef}>
      <div className="relative">
        <Plus className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={16} />
        <input 
          type="text" placeholder="Add any game..."
          className="w-full bg-card border border-border rounded-xl pl-10 pr-4 py-3 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all"
          value={query} onChange={(e) => handleSearch(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query.length > 1 && setShowResults(true)}
        />
      </div>
      <AnimatePresence>
        {showResults && results.length > 0 && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            ref={resultsRef}
            className="absolute z-50 w-full mt-1 bg-card border border-border rounded-lg shadow-2xl max-h-60 overflow-y-auto"
          >
            {results.map((res, idx) => (
              <button 
                key={res} 
                onClick={() => { onAdd(res); setQuery(''); setShowResults(false); }}
                className={`w-full text-left px-4 py-2 text-sm transition-colors border-b border-border/50 last:border-0 ${
                  idx === activeIndex ? 'bg-primary text-primary-foreground' : 'hover:bg-secondary'
                }`}
              >
                {res}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// --- Main View ---

const CatalogueView: React.FC = () => {
  const [steamId] = useState<string>(() => {
    const saved = sessionStorage.getItem('personalization_state');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return parsed.steamId || '';
      } catch (e) {
        return '';
      }
    }
    return '';
  });

  const [loading, setLoading] = useState(false);
  const [entries, setEntries] = useState<CatalogueEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [visibleCount, setVisibleCount] = useState(ROWS_PER_PAGE);
  const [isSaving, setIsSaving] = useState(false);
  const [blurNSFW, setBlurNSFW] = useState(true);
  const [sortConfig, setSortConfig] = useState<{ key: keyof CatalogueEntry; direction: 'asc' | 'desc' }>({
    key: 'status',
    direction: 'asc'
  });

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 400);
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 500) {
        setVisibleCount(prev => prev + ROWS_PER_PAGE);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (steamId) fetchCatalogue();
  }, [steamId]);

  useEffect(() => {
    const savedFilters = sessionStorage.getItem('recommendations_filters');
    if (savedFilters) {
      try {
        const parsed = JSON.parse(savedFilters);
        if (parsed.remove_nsfw !== undefined) setBlurNSFW(parsed.remove_nsfw);
      } catch (e) {}
    }
  }, []);

  const fetchCatalogue = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/user/catalogue/${steamId}`);
      if (!res.ok) throw new Error("Failed to fetch catalogue.");
      const data: CatalogueEntry[] = await res.json();
      const sorted = data.sort((a, b) => {
        if (a.status === b.status) return a.name.localeCompare(b.name);
        return a.status.localeCompare(b.status);
      });
      setEntries(sorted);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = useCallback((appid: number, status: GameStatus) => {
    setEntries(prev => prev.map(entry => {
      if (entry.appid === appid) {
        let ignore = entry.ignore;
        if (status === 'ignored') ignore = true;
        else if (status === 'rated' || status === 'played') ignore = false;
        else ignore = true; 
        return { ...entry, status, ignore };
      }
      return entry;
    }));
  }, []);

  const handleRatingChange = useCallback((appid: number, rating: number) => {
    setEntries(prev => prev.map(entry => {
      if (entry.appid === appid) {
        return { ...entry, actual_rating: rating, status: 'rated' as GameStatus, ignore: false };
      }
      return entry;
    }));
  }, []);

  const handleSort = (key: keyof CatalogueEntry) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') direction = 'desc';
    setSortConfig({ key, direction });
    
    const sorted = [...entries].sort((a, b) => {
      const valA = a[key];
      const valB = b[key];
      if (valA === valB) return a.name.localeCompare(b.name);
      if (typeof valA === 'string' && typeof valB === 'string') return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      if (typeof valA === 'number' && typeof valB === 'number') return direction === 'asc' ? valA - valB : valB - valA;
      if (typeof valA === 'boolean' && typeof valB === 'boolean') return direction === 'asc' ? (valA === valB ? 0 : valA ? 1 : -1) : (valA === valB ? 0 : valB ? 1 : -1);
      return 0;
    });
    setEntries(sorted);
  };

  const handleManualAdd = async (gameName: string) => {
    try {
      const meta = await getMetadata([gameName]);
      if (meta && meta.length > 0) {
        const game = meta[0];
        if (entries.some(e => e.appid === game.appid)) {
          alert("Game already in catalogue!");
          return;
        }
        setEntries(prev => [{
          appid: game.appid,
          name: game.name,
          predicted_rating: 5,
          actual_rating: 5,
          ignore: true,
          status: 'backlog',
          playtime_forever: 0,
          is_manual: true,
          is_nsfw: game.is_nsfw
        }, ...prev]);
      }
    } catch (err) {
      console.error("Failed to add manual game", err);
    }
  };

  const handleDeleteEntry = useCallback((appid: number) => {
    if (!window.confirm("Remove this entry?")) return;
    setEntries(prev => prev.filter(e => e.appid !== appid));
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const updates = entries.map(e => ({ 
        steam_id: steamId, 
        appid: e.appid, 
        actual_rating: e.actual_rating, 
        ignore: e.ignore, 
        status: e.status,
        notes: e.notes
      }));
      const res = await fetch(`${API_BASE_URL}/user/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      if (!res.ok) throw new Error("Failed to save catalogue");
      const savedPersonalization = sessionStorage.getItem('personalization_state');
      if (savedPersonalization) {
        const parsed = JSON.parse(savedPersonalization);
        parsed.games = []; 
        sessionStorage.setItem('personalization_state', JSON.stringify(parsed));
      }
      alert("Catalogue saved successfully!");
    } catch (err: any) {
      alert("Error saving: " + err.message);
    } finally {
      setIsSaving(false);
    }
  };

  const filteredEntries = useMemo(() => {
    if (!searchQuery) return entries;
    const lower = searchQuery.toLowerCase();
    return entries.filter(e => e.name.toLowerCase().includes(lower) || e.appid.toString().includes(lower));
  }, [entries, searchQuery]);

  useEffect(() => { setVisibleCount(ROWS_PER_PAGE); }, [searchQuery, sortConfig]);

  const visibleEntries = useMemo(() => filteredEntries.slice(0, visibleCount), [filteredEntries, visibleCount]);

  if (!steamId) {
    return (
      <div className="max-w-2xl mx-auto py-20 text-center space-y-6">
        <div className="w-20 h-20 bg-secondary rounded-full flex items-center justify-center mx-auto">
          <Library size={40} className="text-muted-foreground" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-bold">No Library Loaded</h2>
          <p className="text-muted-foreground">Please go to the <strong>Personalize</strong> tab and enter your SteamID first.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      <div className="sticky top-16 z-40 bg-background/95 backdrop-blur-xl py-6 -mx-4 px-4 border-b border-border/50 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="space-y-4 flex-grow">
            <div>
              <h2 className="text-3xl font-bold">Game Catalogue</h2>
              <p className="text-muted-foreground italic text-sm">Track your collection and organize your backlog.</p>
            </div>
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-grow max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={18} />
                <input 
                  type="text" placeholder="Search your catalogue..."
                  className="w-full bg-card border border-border rounded-xl pl-10 pr-4 py-3 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                  value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <ManualAddControl onAdd={handleManualAdd} />
            </div>
          </div>
          <div className="flex gap-3">
            <button 
              onClick={fetchCatalogue} disabled={loading}
              className="p-3 bg-secondary hover:bg-secondary/80 rounded-xl text-muted-foreground transition-colors border border-border/50"
              title="Refresh from Server"
            >
              <RefreshCcw size={20} className={loading ? 'animate-spin' : ''} />
            </button>
            <button 
              onClick={handleSave} disabled={isSaving || loading}
              className="bg-primary text-primary-foreground px-8 py-3 rounded-xl font-bold flex items-center gap-2 hover:opacity-90 shadow-lg shadow-primary/20 disabled:opacity-50 transition-all"
            >
              {isSaving ? <RefreshCcw size={20} className="animate-spin" /> : <Save size={20} />}
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-40 space-y-4">
          <RefreshCcw size={48} className="text-primary animate-spin" />
          <p className="text-muted-foreground font-bold uppercase tracking-widest text-xs">Loading Catalogue...</p>
        </div>
      ) : (
        <div className="bg-card border border-border rounded-2xl overflow-hidden shadow-xl overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead className="bg-secondary/50 border-b border-border">
              <tr>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground w-20">Img</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground cursor-pointer hover:bg-secondary" onClick={() => handleSort('name')}>
                  <div className="flex items-center gap-2">Game {sortConfig.key === 'name' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
                </th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground cursor-pointer hover:bg-secondary" onClick={() => handleSort('status')}>
                  <div className="flex items-center gap-2">Status {sortConfig.key === 'status' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
                </th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground">Rating</th>
                <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {visibleEntries.map((entry) => (
                <CatalogueRow 
                  key={entry.appid} 
                  entry={entry} 
                  blurNSFW={blurNSFW}
                  onStatusChange={handleStatusChange}
                  onRatingChange={handleRatingChange}
                  onDelete={handleDeleteEntry}
                />
              ))}
            </tbody>
          </table>
          {filteredEntries.length === 0 && (
            <div className="py-20 text-center text-muted-foreground italic">No games matching your search.</div>
          )}
          {visibleCount < filteredEntries.length && (
            <div className="py-8 text-center border-t border-border/50">
              <p className="text-xs text-muted-foreground uppercase tracking-widest font-bold">Scrolling for more... ({visibleCount} of {filteredEntries.length} shown)</p>
            </div>
          )}
        </div>
      )}

      {/* Floating Action Buttons */}
      <div className="fixed bottom-8 right-8 z-50 flex flex-col gap-4">
        <AnimatePresence>
          {showScrollTop && (
            <motion.button
              initial={{ opacity: 0, y: 20, scale: 0.8 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.8 }}
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="p-4 bg-secondary text-foreground rounded-full shadow-2xl border border-border/50 hover:bg-secondary/80 transition-all group"
              title="Scroll to Top"
            >
              <ArrowUp size={24} className="group-hover:-translate-y-1 transition-transform" />
            </motion.button>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {showScrollTop && (
            <motion.button
              initial={{ opacity: 0, y: 20, scale: 0.8 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 20, scale: 0.8 }}
              onClick={handleSave} disabled={isSaving || loading}
              className="bg-primary text-primary-foreground p-4 rounded-full shadow-2xl flex items-center justify-center hover:opacity-90 transition-all disabled:opacity-50 group"
              title="Save All Changes"
            >
              {isSaving ? <RefreshCcw size={24} className="animate-spin" /> : (
                <div className="flex items-center gap-2 px-2"><Save size={24} /><span className="font-bold text-sm">Save Changes</span></div>
              )}
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default CatalogueView;
