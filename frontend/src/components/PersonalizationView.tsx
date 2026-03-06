import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  User, 
  Search, 
  Table as TableIcon, 
  LineChart, 
  ChartBar,
  ArrowRight, 
  RefreshCcw, 
  ExternalLink,
  ChevronDown,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Download,
  RotateCcw,
  ThumbsUp,
  ThumbsDown,
  Hash,
  Compass,
  Anchor,
  Sparkles,
  Library
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getMetadata, getTermLinks, API_BASE_URL } from '../api';
import ExplainabilityChart from './ExplainabilityChart';
import ViolinPlot from './ViolinPlot';

import GameAddControl from './GameAddControl';

interface GameHeaderImageProps {
  appid: number;
  isNSFW?: boolean;
  blurNSFW?: boolean;
  className?: string;
}

const GameHeaderImage: React.FC<GameHeaderImageProps> = ({ appid, isNSFW, blurNSFW, className }) => {
  const [src, setSrc] = useState(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`);
  const [retryCount, setRetryCount] = useState(0);

  const handleError = () => {
    if (retryCount === 0) {
      // Try capsule image if header fails
      setSrc(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/capsule_231x87.jpg`);
      setRetryCount(1);
    } else if (retryCount === 1) {
      // Try another capsule format
      setSrc(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/capsule_184x69.jpg`);
      setRetryCount(2);
    } else {
      // Final fallback to placeholder
      setSrc('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E');
    }
  };

  return (
    <img
      src={src}
      className={`${className} ${isNSFW && blurNSFW ? 'blur-sm scale-110' : ''}`}
      onError={handleError}
      alt="Game Header"
    />
  );
};

interface GameVerification {
  appid: number;
  name: string;
  predicted_rating: number;
  actual_rating: number;
  ignore: boolean;
  user_review_text?: string;
  user_voted_up?: boolean;
  playtime_forever: number;
  status: string;
  is_manual?: boolean;
  is_nsfw?: boolean;
}

interface PersonalizationViewProps {
  onApply?: (profile: any) => void;
}

// --- Sub-components for Performance ---

interface VerificationRowProps {
  game: GameVerification;
  blurNSFW: boolean;
  showPlaytime: boolean;
  onRatingChange: (appid: number, rating: number) => void;
  onIgnoreChange: (appid: number, ignore: boolean) => void;
  onDelete?: (appid: number) => void;
}

const VerificationRow = React.memo(({ game, blurNSFW, showPlaytime, onRatingChange, onIgnoreChange, onDelete }: VerificationRowProps) => {
  return (
    <tr className={`hover:bg-secondary/30 transition-colors ${game.ignore ? 'opacity-40' : ''}`}>
      <td className="px-6 py-3">
        <a href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="block hover:opacity-80 transition-opacity">
          <GameHeaderImage 
            appid={game.appid} 
            isNSFW={game.is_nsfw}
            blurNSFW={blurNSFW}
            className="w-16 h-8 object-cover rounded shadow-sm border border-border/50"
          />
        </a>
      </td>
      <td className="px-6 py-3">
        <a href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="group">
          <div className="font-bold text-sm leading-tight group-hover:text-primary transition-colors">{game.name}</div>
          <div className="text-[10px] text-muted-foreground">AppID: {game.appid}</div>
        </a>
      </td>
      <td className="px-6 py-3 text-center">
        <span className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-widest ${
          game.status === 'ignored' ? 'bg-red-500/10 text-red-500' :
          game.status === 'rated' ? 'bg-green-500/10 text-green-500' :
          game.status === 'played' ? 'bg-blue-500/10 text-blue-500' :
          'bg-secondary text-muted-foreground'
        }`}>
          {game.status}
        </span>
      </td>
      {showPlaytime && (
        <td className="px-6 py-3 text-xs font-mono text-muted-foreground">
          {(game.playtime_forever / 60).toFixed(1)}h
        </td>
      )}
      <td className="px-6 py-3 text-center">
        <span className="px-2 py-1 bg-secondary rounded text-[10px] font-mono">
          {Math.round(game.predicted_rating)}
        </span>
      </td>
      <td className="px-6 py-3 min-w-[180px]">
        <div className="flex items-center gap-3">
          <input 
            type="range" min="0" max="10" step="1" className="w-full accent-primary h-1"
            value={game.actual_rating}
            onChange={(e) => onRatingChange(game.appid, parseInt(e.target.value))}
          />
          <span className="w-4 text-xs font-bold text-primary text-center">{Math.round(game.actual_rating)}</span>
        </div>
      </td>
      <td className="px-6 py-3 text-center">
        <input 
          type="checkbox" checked={game.ignore}
          onChange={(e) => onIgnoreChange(game.appid, e.target.checked)}
          className="w-4 h-4 rounded border-border text-primary"
        />
      </td>
      <td className="px-6 py-3 text-center">
        {game.is_manual ? (
          <button onClick={() => onDelete?.(game.appid)} className="text-muted-foreground hover:text-destructive transition-colors">
            <Trash2 size={16} />
          </button>
        ) : (
          <span className="text-[10px] text-muted-foreground/30 font-bold uppercase">Library</span>
        )}
      </td>
    </tr>
  );
});

VerificationRow.displayName = 'VerificationRow';

interface VerificationTableProps {
  data: GameVerification[];
  title: string;
  showPlaytime?: boolean;
  blurNSFW?: boolean;
  visibleCount?: number;
  sortConfig: { key: keyof GameVerification; direction: 'asc' | 'desc' };
  onSort: (key: keyof GameVerification) => void;
  onRatingChange: (appid: number, rating: number) => void;
  onIgnoreChange: (appid: number, ignore: boolean) => void;
  onDelete?: (appid: number) => void;
}

const VerificationTable = ({ 
  data, title, showPlaytime = true, blurNSFW = true, visibleCount = 50, sortConfig, onSort, onRatingChange, onIgnoreChange, onDelete 
}: VerificationTableProps) => (
  <div className="space-y-4">
    <h3 className="text-sm font-bold uppercase tracking-widest text-primary px-2">{title} ({data.length})</h3>
    <div className="bg-card border border-border rounded-2xl overflow-hidden shadow-xl overflow-x-auto">
      <table className="w-full text-left">
        <thead className="bg-secondary/50 border-b border-border">
          <tr>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground w-20">Img</th>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground cursor-pointer hover:bg-secondary" onClick={() => onSort('name')}>
              <div className="flex items-center gap-2">Game {sortConfig.key === 'name' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
            </th>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground text-center">Status</th>
            {showPlaytime && (
              <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground cursor-pointer hover:bg-secondary" onClick={() => onSort('playtime_forever')}>
                <div className="flex items-center gap-2">Playtime {sortConfig.key === 'playtime_forever' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
              </th>
            )}
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground text-center cursor-pointer hover:bg-secondary" onClick={() => onSort('predicted_rating')}>
              <div className="flex items-center justify-center gap-2">Pred {sortConfig.key === 'predicted_rating' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
            </th>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground cursor-pointer hover:bg-secondary" onClick={() => onSort('actual_rating')}>
              <div className="flex items-center gap-2">Your Rating {sortConfig.key === 'actual_rating' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
            </th>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground text-center cursor-pointer hover:bg-secondary" onClick={() => onSort('ignore')}>
              <div className="flex items-center justify-center gap-2">Ignore {sortConfig.key === 'ignore' && (sortConfig.direction === 'asc' ? <ChevronDown size={14} className="rotate-180" /> : <ChevronDown size={14} />)}</div>
            </th>
            <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-muted-foreground text-center">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {data.slice(0, visibleCount).map((game) => (
            <VerificationRow 
              key={game.appid} 
              game={game} 
              blurNSFW={blurNSFW} 
              showPlaytime={showPlaytime}
              onRatingChange={onRatingChange}
              onIgnoreChange={onIgnoreChange}
              onDelete={onDelete}
            />
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

// --- Main View ---

const PersonalizationView: React.FC<PersonalizationViewProps> = ({ onApply }) => {
  const [termLinks, setTermLinks] = useState<Record<string, string>>({});
  const [hoveredWeight, setHoveredWeight] = useState<string | null>(null);
  const [hoveredDimension, setHoveredDimension] = useState<string | null>(null);
  const [hoveredSemanticDimension, setHoveredSemanticDimension] = useState<string | null>(null);
  const [hoveredTag, setHoveredTag] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [links] = await Promise.all([getTermLinks()]);
        setTermLinks(links);
      } catch (err) {
        console.error("Failed to fetch term links", err);
      }
    };
    fetchData();
  }, []);

  // Persistence Helper
  const getSaved = () => {
    const saved = sessionStorage.getItem('personalization_state');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    return null;
  };

  const savedState = getSaved();

  const [step, setStep] = useState<number>(savedState?.step || 1);
  const [steamId, setSteamId] = useState<string>(savedState?.steamId || '');
  const [reviewHtml, setReviewHtml] = useState<string>(savedState?.reviewHtml || '');
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<any>(savedState?.status || null);
  const [games, setGames] = useState<GameVerification[]>(savedState?.games || []);
  const [insights, setInsights] = useState<any>(savedState?.insights || null);
  const [error, setError] = useState<string | null>(null);
  const [solverStatus, setSolverStatus] = useState<string>('');
  const [visibleCount, setVisibleCount] = useState(50);
  const [blurNSFW, setBlurNSFW] = useState(true);
  const [sortConfig, setSortConfig] = useState<{ key: keyof GameVerification; direction: 'asc' | 'desc' }>({
    key: 'predicted_rating',
    direction: 'desc'
  });

  // Effects for persistence and environment sync
  useEffect(() => {
    const state = { step, steamId, reviewHtml, games, insights, status };
    sessionStorage.setItem('personalization_state', JSON.stringify(state));
  }, [step, steamId, reviewHtml, games, insights, status]);

  useEffect(() => {
    if (steamId && step === 2 && games.length === 0) fetchVerificationData(steamId);
  }, [step]);

  useEffect(() => {
    const handleScroll = () => {
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 500) {
        setVisibleCount(prev => prev + 50);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const checkFilters = () => {
      const saved = sessionStorage.getItem('recommendations_filters');
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (parsed.remove_nsfw !== undefined) setBlurNSFW(parsed.remove_nsfw);
        } catch (e) {}
      }
    };
    checkFilters();
    window.addEventListener('storage', checkFilters);
    const interval = setInterval(checkFilters, 1000);
    return () => { window.removeEventListener('storage', checkFilters); clearInterval(interval); };
  }, []);

  useEffect(() => {
    let interval: any;
    if (step === 1.5 && steamId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/user/status/${steamId}`);
          const data = await res.json();
          setStatus(data);
          if (data.has_soft_labels) fetchVerificationData(steamId);
        } catch (err) {}
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [step, steamId]);

  const handleFetch = async () => {
    let cleanId = steamId.trim().replace(/\/$/, ""); 
    if (cleanId.includes('steamcommunity.com')) {
      const profileMatch = cleanId.match(/profiles\/(\d+)/);
      const idMatch = cleanId.match(/id\/([^/]+)/);
      if (profileMatch) cleanId = profileMatch[1];
      else if (idMatch) cleanId = idMatch[1];
    }
    
    setLoading(true);
    setError(null);

    try {
      // Build 68: Check for existing data before starting a new fetch
      const statusRes = await fetch(`${API_BASE_URL}/user/status/${cleanId}`);
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setStatus(statusData);
        if (statusData.has_profile) {
          const insightRes = await fetch(`${API_BASE_URL}/user/insights/${cleanId}`);
          if (insightRes.ok) {
            const insightData = await insightRes.json();
            setInsights(insightData);
            setSteamId(cleanId);
            setStep(3);
            setLoading(false);
            return;
          }
        }
        if (statusData.has_soft_labels) {
          setSteamId(cleanId);
          await fetchVerificationData(cleanId);
          setLoading(false);
          return;
        }
      }

      const res = await fetch(`${API_BASE_URL}/user/fetch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steam_id: cleanId, review_html: reviewHtml })
      });
      if (!res.ok) throw new Error("Failed to start acquisition");
      setSteamId(cleanId);
      setStep(1.5);
    } catch (err: any) { setError(err.message); } finally { setLoading(false); }
  };

  const fetchVerificationData = async (sid: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/user/verify/${sid}`);
      if (!res.ok) return;
      const data = await res.json();
      setGames(data);
      if (step === 1.5) setStep(2);
    } catch (err) {}
  };

  const handleManualAdd = async (name: string) => {
    try {
      const meta = await getMetadata([name]);
      if (meta && meta.length > 0) {
        const game = meta[0];
        if (games.some(g => g.appid === game.appid)) { alert("Game already in verification list!"); return; }
        const newEntry = { appid: game.appid, name: game.name, predicted_rating: 5, actual_rating: 5, ignore: true, playtime_forever: 0, status: 'backlog', is_manual: true, is_nsfw: game.is_nsfw };
        setGames(prev => [newEntry, ...prev]);
        await fetch(`${API_BASE_URL}/user/verify`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([{ steam_id: steamId, appid: game.appid, actual_rating: 5, ignore: true, status: 'backlog' }]) });
      }
    } catch (err) {}
  };

  const handleSort = (key: keyof GameVerification) => {
    let direction: 'asc' | 'desc' = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') direction = 'asc';
    setSortConfig({ key, direction });
    setGames([...games].sort((a, b) => {
      const valA = a[key], valB = b[key];
      if (typeof valA === 'string' && typeof valB === 'string') return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      if (typeof valA === 'number' && typeof valB === 'number') return direction === 'asc' ? valA - valB : valB - valA;
      if (typeof valA === 'boolean' && typeof valB === 'boolean') return direction === 'asc' ? (valA === valB ? 0 : valA ? 1 : -1) : (valA === valB ? 0 : valB ? 1 : -1);
      return 0;
    }));
  };

  const handleIgnoreChange = useCallback((appid: number, ignore: boolean) => {
    setGames(prev => prev.map(g => {
      if (g.appid === appid) {
        let newStatus = g.status;
        if (ignore) newStatus = 'ignored';
        else {
          if (g.actual_rating > 0) newStatus = 'rated';
          else if (g.playtime_forever > 0) newStatus = 'played';
          else newStatus = 'backlog';
        }
        return { ...g, ignore, status: newStatus };
      }
      return g;
    }));
  }, []);

  const handleRatingChange = useCallback((appid: number, actual_rating: number) => {
    setGames(prev => prev.map(g => g.appid === appid ? { ...g, actual_rating, status: 'rated' } : g));
  }, []);

  const handleDeleteManual = useCallback(async (appid: number) => {
    if (!window.confirm("Remove this manual entry?")) return;
    setGames(prev => prev.filter(g => g.appid !== appid));
    try {
      await fetch(`${API_BASE_URL}/user/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([{ steam_id: steamId, appid: appid, actual_rating: 0, ignore: true }])
      });
    } catch (err) {}
  }, [steamId]);

  const handleSaveAndSolve = async () => {
    setLoading(true); setSolverStatus('Uploading ratings...');
    try {
      const updates = games.map(g => ({ steam_id: steamId, appid: g.appid, actual_rating: g.actual_rating, ignore: g.ignore, status: g.status }));
      await fetch(`${API_BASE_URL}/user/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      setSolverStatus('Solving Taste DNA...');
      const res = await fetch(`${API_BASE_URL}/user/solve/${steamId}`, { method: 'POST' });
      if (!res.ok) throw new Error("Solver failed");
      const insightRes = await fetch(`${API_BASE_URL}/user/insights/${steamId}`);
      const data = await insightRes.json();
      setInsights(data);
      setStep(3);
    } catch (err: any) { alert(err.message); } finally { setLoading(false); setSolverStatus(''); }
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(insights, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `user_${steamId}_taste_profile.json`;
    a.click(); URL.revokeObjectURL(url);
  };

  const manualGames = useMemo(() => games.filter(g => g.is_manual), [games]);
  const libraryGames = useMemo(() => games.filter(g => !g.is_manual), [games]);

  return (
    <div className="max-w-6xl mx-auto space-y-12 pb-20">
      <div className="flex justify-center mb-8">
        {[1, 2, 3].map((s) => {
          const isActive = step >= s; const isCurrent = Math.floor(step) === s;
          return (
            <React.Fragment key={s}>
              <button 
                onClick={() => step > s && setStep(s)}
                className={`flex flex-col items-center gap-2 group transition-all ${step > s ? 'cursor-pointer' : 'cursor-default'}`}
              >
                <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                  isCurrent ? 'bg-primary border-primary text-primary-foreground shadow-lg scale-110' :
                  isActive ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary border-border text-muted-foreground'
                }`}>
                  {s === 1 ? <Search size={20} /> : s === 2 ? <TableIcon size={20} /> : <ChartBar size={20} />}
                </div>
                <span className={`text-[10px] font-bold uppercase tracking-widest ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>
                  {s === 1 ? 'Acquire' : s === 2 ? 'Verify' : 'Solve'}
                </span>
              </button>
              {s < 3 && <div className={`w-12 h-0.5 mt-6 ${step > s ? 'bg-primary' : 'bg-muted-foreground/30'}`} />}
            </React.Fragment>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {step === 1 && (
          <motion.div 
            key="step1" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            className="bg-card border border-border rounded-2xl p-8 shadow-xl space-y-8 max-w-2xl mx-auto"
          >
            <div className="text-center space-y-2">
              <h2 className="text-3xl font-bold">Personalize Your Experience</h2>
              <p className="text-muted-foreground italic">By analyzing your Steam library and reviews, we can solve for your personal "Taste DNA."</p>
            </div>
            <div className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground flex items-center gap-2"><User size={14} />SteamID64 or Custom URL</label>
                <div className="flex gap-2">
                  <input 
                    type="text" placeholder="e.g., 76561198039155404"
                    className="flex-grow bg-secondary border-none rounded-xl px-4 py-3 text-lg outline-none focus:ring-2 focus:ring-primary/50"
                    value={steamId} onChange={(e) => setSteamId(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') handleFetch(); }}
                  />
                  {steamId && (
                    <button onClick={() => setSteamId('')} className="bg-secondary hover:bg-secondary/80 p-3 rounded-xl text-muted-foreground transition-colors">
                      <RotateCcw size={20} />
                    </button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-muted-foreground flex items-center gap-2"><Search size={14} />Optional: Paste Review HTML</label>
                  <a href={`https://steamcommunity.com/id/${steamId}/recommended/`} target="_blank" rel="noopener noreferrer" className="text-[10px] text-primary hover:underline flex items-center gap-1">Find my reviews <ExternalLink size={10} /></a>
                </div>
                <textarea 
                  placeholder="Paste HTML from your Steam Reviews page..."
                  className="w-full bg-secondary border-none rounded-xl px-4 py-3 text-sm h-32 outline-none focus:ring-2 focus:ring-primary/50 font-mono"
                  value={reviewHtml} onChange={(e) => setReviewHtml(e.target.value)}
                />
              </div>
              <button 
                onClick={handleFetch} disabled={!steamId || loading}
                className="w-full bg-primary text-primary-foreground py-4 rounded-xl font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg shadow-primary/20"
              >
                {loading ? <RefreshCcw size={20} className="animate-spin" /> : <ArrowRight size={20} />}
                {loading ? 'Starting Acquisition...' : 'Start Analysis'}
              </button>
              {error && <div className="bg-destructive/10 text-destructive text-sm p-4 rounded-xl flex items-center gap-2"><AlertCircle size={16} />{error}</div>}
            </div>
          </motion.div>
        )}

        {step === 1.5 && (
          <motion.div key="step1.5" className="flex flex-col items-center justify-center py-20 space-y-6 text-center">
            <div className="relative">
              <div className="w-24 h-24 border-4 border-primary/20 rounded-full animate-pulse" />
              <div className="absolute inset-0 flex items-center justify-center"><RefreshCcw size={40} className="text-primary animate-spin" /></div>
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Acquiring Data...</h2>
              <p className="text-muted-foreground max-w-sm">We're fetching your library, parsing your reviews, and generating initial predictions.</p>
            </div>
          </motion.div>
        )}

        {step === 2 && (
          <motion.div key="step2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
              <div className="space-y-4 flex-grow">
                <div>
                  <h2 className="text-2xl font-bold">Verify Your Ratings</h2>
                  <p className="text-sm text-muted-foreground italic">Correct the predicted ratings where needed. Add games manually if you want to broaden the training set.</p>
                </div>
                <GameAddControl onAdd={handleManualAdd} placeholder="Add a game manually (e.g. Elden Ring)" />
              </div>
              <div className="flex flex-col items-end gap-3">
                <button onClick={handleSaveAndSolve} disabled={loading || games.length < 10} className="bg-primary text-primary-foreground px-8 py-4 rounded-xl font-bold flex items-center gap-2 hover:scale-105 transition-transform shadow-lg shadow-primary/20 disabled:opacity-50 disabled:scale-100">
                  {loading ? <RefreshCcw size={20} className="animate-spin" /> : <Sparkles size={20} />}
                  Solve Taste DNA
                </button>
                {loading && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="text-xs font-bold text-primary uppercase tracking-widest flex items-center gap-2">
                    <RefreshCcw size={12} className="animate-spin" />
                    {solverStatus}
                  </motion.div>
                )}
              </div>
            </div>

            {manualGames.length > 0 && (
              <VerificationTable 
                data={manualGames} title="Manual Additions" showPlaytime={false} blurNSFW={blurNSFW} sortConfig={sortConfig} onSort={handleSort}
                onRatingChange={handleRatingChange} onIgnoreChange={handleIgnoreChange} onDelete={handleDeleteManual} visibleCount={visibleCount}
              />
            )}
            <VerificationTable 
              data={libraryGames} title="Library Games" blurNSFW={blurNSFW} sortConfig={sortConfig} onSort={handleSort}
              onRatingChange={handleRatingChange} onIgnoreChange={handleIgnoreChange} visibleCount={visibleCount}
            />
          </motion.div>
        )}

        {step === 3 && insights && (
          <motion.div key="step3" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="space-y-12">
            <div className="bg-gradient-to-br from-primary/20 to-card border border-primary/20 rounded-3xl p-8 flex flex-col md:flex-row items-center gap-8 shadow-2xl">
              <div className="w-32 h-32 bg-primary rounded-full flex items-center justify-center shadow-inner shrink-0">
                <CheckCircle2 size={64} className="text-primary-foreground" />
              </div>
              <div className="space-y-4">
                <h2 className="text-4xl font-bold">Taste DNA Solved</h2>
                <p className="text-muted-foreground italic text-lg max-w-2xl">We've identified the underlying patterns that drive your enjoyment. Your profile is now optimized with an R² of {insights.r2?.toFixed(2) || '0.55'}.</p>
                <div className="flex flex-wrap gap-4">
                  <button onClick={() => { if (onApply) onApply(insights); }} className="bg-primary text-primary-foreground px-8 py-3 rounded-xl font-bold hover:scale-105 transition-transform shadow-lg shadow-primary/20">Apply to Recommender</button>
                  <button onClick={() => setStep(2)} className="bg-secondary text-foreground px-6 py-3 rounded-xl font-bold flex items-center gap-2 hover:bg-secondary/80 border border-border/50">
                    <TableIcon size={18} />
                    Back to Verify
                  </button>
                  <button onClick={handleExport} className="bg-secondary text-foreground px-6 py-3 rounded-xl font-bold flex items-center gap-2 hover:bg-secondary/80 border border-border/50"><Download size={18} />Export Profile</button>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-8">
                {/* LOVE LIST */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">  
                  <h3 className="text-lg font-bold flex items-center gap-2 text-blue-500"><ThumbsUp size={18} />Games You'll Love</h3>
                  <div className="space-y-3">
                    {insights.top_recommendations?.slice(0, 30).map((game: any, idx: number) => (      
                      <a
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-blue-500/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-blue-500/10 rounded-full text-[10px] font-bold text-blue-500 shrink-0">{idx + 1}</div>
                        <GameHeaderImage
                          appid={game.appid}
                          isNSFW={game.is_nsfw}
                          blurNSFW={blurNSFW}
                          className="w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform"
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-blue-500 transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-blue-500">{game.predicted_rating?.toFixed(1)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>

                {/* UPCOMING GAMES */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">  
                  <h3 className="text-lg font-bold flex items-center gap-2 text-purple-500"><Sparkles size={18} />Upcoming Games</h3>
                  <div className="space-y-3">
                    {(insights.upcoming_recommendations || insights.bottom_recommendations)?.slice(0, 10).map((game: any, idx: number) => (   
                      <a
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-purple-500/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-purple-500/10 rounded-full text-[10px] font-bold text-purple-500 shrink-0">{idx + 1}</div>
                        <GameHeaderImage
                          appid={game.appid}
                          isNSFW={game.is_nsfw}
                          blurNSFW={blurNSFW}
                          className="w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform"
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-purple-500 transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-purple-500">{game.predicted_rating?.toFixed(1)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6 space-y-6 relative">
                  <h3 className="text-lg font-bold flex items-center gap-2"><LineChart size={18} className="text-primary" />Metadata Weights</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {Object.entries(insights.metadata || {}).filter(([key, val]) => typeof val === 'number' && !['best_q_idx', 'oos_r2'].includes(key)).map(([key, val]: [string, any]) => {
                      const labels: any = { 
                        quality: 'Quality', 
                        age: 'Release Date', 
                        popularity: 'Popularity', 
                        length: 'Playtime', 
                        difficulty: 'Difficulty', 
                        price: 'Price', 
                        tone: 'Tonal Spirit', 
                        semantic: 'Theme Match', 
                        tag_match: 'Vibe Match',
                        kernel_match: 'Mechanical Sim',
                        graph_match: 'Behavioral Sim'
                      };
                      const descriptions: any = { 
                        quality: 'Preference for high critic/user consensus.', 
                        age: 'Preference for newer vs. classic titles.', 
                        popularity: 'Preference for mainstream vs. niche gems.', 
                        length: 'Preference for short vs. long experiences.', 
                        difficulty: 'Preference for relaxed vs. hard games.', 
                        price: 'Sensitivity to game price.', 
                        tone: 'Preference for Bizarre/Absurd (Positive) vs. Serious/Grounded (Negative) spirit.', 
                        semantic: 'Weight of the descriptive theme model.', 
                        tag_match: 'Weight of the categorical tag model.',
                        kernel_match: 'Weight of the high-fidelity mechanical similarity engine.',
                        graph_match: 'Weight of the behavioral graph resonance model.'
                      };
                      return (
                        <div key={key} className="p-4 bg-secondary/30 rounded-xl space-y-2 group cursor-help relative" onMouseEnter={() => setHoveredWeight(key)} onMouseLeave={() => setHoveredWeight(null)}>
                          <div className="flex justify-between items-center"><span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">{labels[key] || key}</span><span className={`text-sm font-mono font-bold ${(val as number) > 0 ? 'text-green-500' : (val as number) < 0 ? 'text-red-500' : 'text-muted-foreground'}`}>{(val as number).toFixed(2)}</span></div>
                          {key === 'tone' ? (
                            <div className="space-y-1">
                              <div className="h-1.5 bg-secondary rounded-full overflow-hidden relative">
                                <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-border z-10" />
                                <div 
                                  className={`h-full transition-all duration-1000 ${(val as number) > 0 ? 'bg-purple-500' : 'bg-blue-500'}`} 
                                  style={{ 
                                    width: `${Math.min(50, Math.abs((val as number) * 50))}%`, 
                                    marginLeft: (val as number) > 0 ? '50%' : `${50 - Math.min(50, Math.abs((val as number) * 50))}%` 
                                  }} 
                                />
                              </div>
                              <div className="flex justify-between text-[8px] font-bold uppercase tracking-tighter text-muted-foreground/50">
                                <span>Grounded</span>
                                <span>Bizarre</span>
                              </div>
                            </div>
                          ) : (
                            <div className="h-1.5 bg-secondary rounded-full overflow-hidden"><div className={`h-full transition-all duration-1000 ${(val as number) > 0 ? 'bg-green-500' : 'bg-red-500'}`} style={{ width: `${Math.min(100, Math.abs((val as number) * 50))}%`, marginLeft: (val as number) > 0 ? '0' : 'auto' }} /></div>
                          )}
                          <AnimatePresence>{hoveredWeight === key && <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute z-50 bottom-full left-0 mb-2 w-48 p-2 bg-popover text-popover-foreground text-[10px] rounded shadow-xl border border-border pointer-events-none">{descriptions[key]}</motion.div>}</AnimatePresence>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6 space-y-6 relative">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Compass size={18} className="text-primary" />Vibe Dimensions</h3>
                  <div className="space-y-4">
                    {(insights.tag_dimensions?.top_dims || []).map((dim: any) => {
                      const dimId = dim.index.toString();
                      const verified = insights.tag_dimensions?.verified_tags?.[dimId];
                      let displayDesc = verified?.dynamic_label || `Dimension ${dimId}`;
                      let displayTopPos = verified?.positive || [];
                      let displayTopNeg = verified?.negative || [];
                      const val = dim.weight;
                      if (val < 0) {
                        if (displayDesc.includes(' vs. ')) displayDesc = displayDesc.split(' vs. ').reverse().join(' vs. ');
                        [displayTopPos, displayTopNeg] = [displayTopNeg, displayTopPos];
                      }
                      const chartData = (insights.tag_dimensions?.correlations?.[dimId] || []).map((c: any) => ({ x: val < 0 ? -c.x : c.x, y: c.y, name: c.name }));
                      return (
                        <div key={dimId} className="space-y-2 relative group" onMouseEnter={() => setHoveredDimension(dimId)} onMouseLeave={() => setHoveredDimension(null)}>
                          <div className="flex justify-between items-center"><span className="text-xs font-bold text-foreground truncate max-w-[80%]">{displayDesc}</span><span className="text-xs font-mono font-bold text-primary">{(Math.abs(val) * 100).toFixed(0)}%</span></div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden"><div className="h-full bg-primary transition-all duration-1000" style={{ width: `${Math.min(100, Math.abs(val * 100))}%` }} /></div>
                          <AnimatePresence>
                            {hoveredDimension === dimId && (
                              <motion.div initial={{ opacity: 0, x: 20, scale: 0.9 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 20, scale: 0.9 }} className="absolute left-[calc(100%+1rem)] top-0 z-[110] w-[400px] md:w-[500px] pointer-events-none">
                                <div className="bg-card border border-primary/30 rounded-2xl shadow-2xl p-6 pb-8 backdrop-blur-2xl space-y-6">
                                  <div className="space-y-4">
                                    <div className="text-xs font-bold uppercase tracking-widest text-primary border-b border-primary/10 pb-2">{displayDesc}</div>
                                    <div className="space-y-3">
                                      <div className="flex gap-2">{displayTopPos.map((tag: string) => <span key={tag} className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[9px] font-bold text-green-500 whitespace-nowrap">{tag}</span>)}</div>
                                      <div className="flex gap-2">{displayTopNeg.map((tag: string) => <span key={tag} className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[9px] font-bold text-red-500 whitespace-nowrap">{tag}</span>)}</div>
                                    </div>
                                  </div>
                                  <ExplainabilityChart data={chartData} title={displayDesc} xLabel="Dimension Loading" type="scatter" showTrendline={true} />
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6 space-y-6 relative">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Sparkles size={18} className="text-primary" />Thematic Dimensions</h3>
                  <div className="space-y-4">
                    {(insights.semantic_dimensions?.top_dims || []).map((dim: any) => {
                      const dimId = dim.index.toString();
                      const labels = insights.semantic_dimensions?.labels?.[dimId];
                      let displayDesc = labels?.dynamic_label || `Semantic Dim ${dimId}`;
                      let displayTopPos = labels?.positive || [];
                      let displayTopNeg = labels?.negative || [];
                      const val = dim.weight;
                      if (val < 0) {
                        if (displayDesc.includes(' vs ')) displayDesc = displayDesc.split(' vs ').reverse().join(' vs ');
                        [displayTopPos, displayTopNeg] = [displayTopNeg, displayTopPos];
                      }
                      const chartData = (insights.semantic_dimensions?.correlations?.[dimId] || []).map((c: any) => ({ x: val < 0 ? -c.x : c.x, y: c.y, name: c.name }));
                      return (
                        <div key={dimId} className="space-y-2 relative group" onMouseEnter={() => setHoveredSemanticDimension(dimId)} onMouseLeave={() => setHoveredSemanticDimension(null)}>
                          <div className="flex justify-between items-center"><span className="text-xs font-bold text-foreground truncate max-w-[80%]">{displayDesc}</span><span className="text-xs font-mono font-bold text-primary">{(Math.abs(val) * 100).toFixed(0)}%</span></div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden"><div className="h-full bg-primary transition-all duration-1000" style={{ width: `${Math.min(100, Math.abs(val * 100))}%` }} /></div>
                          <AnimatePresence>
                            {hoveredSemanticDimension === dimId && (
                              <motion.div initial={{ opacity: 0, x: 20, scale: 0.9 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 20, scale: 0.9 }} className="absolute left-[calc(100%+1rem)] top-0 z-[110] w-[400px] md:w-[500px] pointer-events-none">
                                <div className="bg-card border border-primary/30 rounded-2xl shadow-2xl p-6 pb-8 backdrop-blur-2xl space-y-6">
                                  <div className="space-y-4">
                                    <div className="text-xs font-bold uppercase tracking-widest text-primary border-b border-primary/10 pb-2">{displayDesc}</div>
                                    <div className="space-y-3">
                                      <div className="flex gap-2">{displayTopPos.map((tag: string) => <span key={tag} className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[9px] font-bold text-green-500 whitespace-nowrap">{tag}</span>)}</div>
                                      <div className="flex gap-2">{displayTopNeg.map((tag: string) => <span key={tag} className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[9px] font-bold text-red-500 whitespace-nowrap">{tag}</span>)}</div>
                                    </div>
                                  </div>
                                  <ExplainabilityChart data={chartData} title={displayDesc} xLabel="Dimension Loading" type="scatter" showTrendline={true} />
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6 space-y-6 relative">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Hash size={18} className="text-primary" />Predictive Tags</h3>
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-green-500">Like...</div>
                      <div className="flex flex-wrap gap-2">
                        {(insights.associative_tags?.top || insights.top_tags || []).map((t: any) => (
                          <div key={t.tag} className="relative group/tag" onMouseEnter={() => setHoveredTag(t.tag)} onMouseLeave={() => setHoveredTag(null)}>
                            {termLinks[t.tag] ? <a href={termLinks[t.tag]} target="_blank" rel="noopener noreferrer" className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-medium text-green-500 hover:bg-green-500/20 transition-colors block">{t.tag}</a> : <div className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-medium text-green-500">{t.tag}</div>}
                            <AnimatePresence>{hoveredTag === t.tag && t.ratings_with && <motion.div initial={{ opacity: 0, x: 20, scale: 0.9 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 20, scale: 0.9 }} className="absolute left-[calc(100%+0.5rem)] top-0 z-[120] pointer-events-none"><ViolinPlot ratingsWith={t.ratings_with} ratingsWithout={t.ratings_without} tagName={t.tag} /></motion.div>}</AnimatePresence>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-red-500">Dislike...</div>
                      <div className="flex flex-wrap gap-2">
                        {(insights.associative_tags?.bottom || insights.bottom_tags || []).map((t: any) => (
                          <div key={t.tag} className="relative group/tag" onMouseEnter={() => setHoveredTag(t.tag)} onMouseLeave={() => setHoveredTag(null)}>
                            {termLinks[t.tag] ? <a href={termLinks[t.tag]} target="_blank" rel="noopener noreferrer" className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[10px] font-medium text-red-500 hover:bg-red-500/20 transition-colors block">{t.tag}</a> : <div className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[10px] font-medium text-red-500">{t.tag}</div>}
                            <AnimatePresence>{hoveredTag === t.tag && t.ratings_with && <motion.div initial={{ opacity: 0, x: 20, scale: 0.9 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 20, scale: 0.9 }} className="absolute left-[calc(100%+0.5rem)] top-0 z-[120] pointer-events-none"><ViolinPlot ratingsWith={t.ratings_with} ratingsWithout={t.ratings_without} tagName={t.tag} /></motion.div>}</AnimatePresence>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-8">
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">  
                  <h3 className="text-lg font-bold flex items-center gap-2 text-primary"><Library size={18} />Backlog Priority</h3>
                  <div className="space-y-3">
                    {(insights.backlog_recommendations || []).slice(0, 30).map((game: any, idx: number) => (      
                      <a
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-primary/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-primary/10 rounded-full text-[10px] font-bold text-primary shrink-0">{idx + 1}</div>
                        <GameHeaderImage
                          appid={game.appid}
                          isNSFW={game.is_nsfw}
                          blurNSFW={blurNSFW}
                          className="w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform"
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-primary transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-primary">{game.predicted_rating?.toFixed(1)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>

                {/* FREE LIST */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">  
                  <h3 className="text-lg font-bold flex items-center gap-2 text-green-500"><Download size={18} />Top Free Games</h3>
                  <div className="space-y-3">
                    {(insights.free_recommendations || []).slice(0, 10).map((game: any, idx: number) => (      
                      <a
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-green-500/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-green-500/10 rounded-full text-[10px] font-bold text-green-500 shrink-0">{idx + 1}</div>
                        <GameHeaderImage
                          appid={game.appid}
                          isNSFW={game.is_nsfw}
                          blurNSFW={blurNSFW}
                          className="w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform"
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-green-500 transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-green-500">{game.predicted_rating?.toFixed(1)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>

                <div className="bg-card border border-border rounded-2xl p-6 space-y-6 relative">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Anchor size={18} className="text-primary" />North Stars</h3>
                  <p className="text-xs text-muted-foreground italic">Games that perfectly match your solved Vibe DNA (ignoring age/quality/price).</p>
                  <div className="grid grid-cols-1 gap-4">
                    {(insights.north_stars || []).map((game: any) => (
                      <a key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="bg-secondary/30 border border-border/50 rounded-xl p-4 flex items-center gap-4 hover:border-primary/50 transition-all group">
                        <div className="w-24 h-12 rounded overflow-hidden shrink-0">
                          <GameHeaderImage 
                            appid={game.appid} 
                            isNSFW={game.is_nsfw}
                            blurNSFW={blurNSFW}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" 
                          />
                        </div>
                        <div className="flex-grow min-w-0"><div className="font-bold text-sm truncate group-hover:text-primary transition-colors">{game.name}</div><div className="text-[10px] text-muted-foreground uppercase tracking-widest">Match Score: {game.alignment?.toFixed(2)}</div></div>
                        <ExternalLink size={14} className="text-muted-foreground group-hover:text-primary transition-colors" />
                      </a>
                    ))}
                  </div>
                </div>

              </div>

                {/* TAG-SPECIFIC RECOMMENDATIONS */}
                {insights.associative_tags?.top?.some((t: any) => t.top_games && t.top_games.length > 0) && (
                  <div className="md:col-span-2 space-y-8 mt-4">
                    <div className="flex items-center gap-2">
                      <Sparkles size={20} className="text-primary" />
                      <h3 className="text-xl font-bold">Top Recommendations by Tag</h3>   
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {insights.associative_tags.top
                        .filter((t: any) => t.top_games && t.top_games.length > 0)        
                        .map((t: any) => (
                          <div key={t.tag} className="bg-card border border-border rounded-2xl p-5 space-y-4 shadow-sm hover:shadow-md transition-shadow">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                {termLinks[t.tag] ? (
                                  <a
                                    href={termLinks[t.tag]} target="_blank" rel="noopener noreferrer"
                                    className="px-2.5 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-bold text-green-500 hover:bg-green-500/20 transition-colors uppercase tracking-wider"
                                  >
                                    {t.tag}
                                  </a>
                                ) : (
                                  <div className="px-2.5 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-bold text-green-500 uppercase tracking-wider"> 
                                    {t.tag}
                                  </div>
                                )}
                              </div>
                              <span className="text-[8px] font-bold text-muted-foreground uppercase tracking-widest bg-secondary/50 px-2 py-0.5 rounded">Tag Expert</span>
                            </div>

                            <div className="flex gap-3 overflow-x-auto pb-1 no-scrollbar">
                              {t.top_games.map((game: any) => (
                                <a
                                  key={game.appid}
                                  href={`https://store.steampowered.com/app/${game.appid}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex-shrink-0 w-32 group/tag-game"
                                >
                                  <div className="relative aspect-video rounded-lg overflow-hidden border border-border/50 group-hover/tag-game:border-primary/50 transition-colors">
                                    <GameHeaderImage
                                      appid={game.appid}
                                      isNSFW={game.is_nsfw}
                                      blurNSFW={blurNSFW}
                                      className="w-full h-full object-cover group-hover/tag-game:scale-105 transition-transform"     
                                    />
                                  </div>
                                  <div className="mt-1.5 px-0.5">
                                    <div className="text-[10px] font-bold truncate group-hover/tag-game:text-primary transition-colors leading-tight">{game.name}</div>
                                  </div>
                                </a>
                              ))}
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* FAVORITE-SEED RECOMMENDATIONS */}
                {insights.favorite_game_recommendations?.length > 0 && (
                  <div className="md:col-span-2 space-y-8 mt-12">
                    <div className="flex items-center gap-2">
                      <ThumbsUp size={20} className="text-primary" />
                      <h3 className="text-xl font-bold">Similar to Your Favorites</h3>    
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {insights.favorite_game_recommendations.map((fav: any) => (
                        <div key={fav.seed_appid} className="bg-card border border-border rounded-2xl p-5 space-y-4 shadow-sm hover:shadow-md transition-shadow">
                          <div className="flex items-center justify-between gap-4">       
                            <div className="flex items-center gap-2 min-w-0">
                              <GameHeaderImage
                                appid={fav.seed_appid}
                                className="w-10 h-5 object-cover rounded shadow-sm border border-border/50 shrink-0"
                              />
                              <div className="text-[10px] font-bold truncate text-muted-foreground uppercase tracking-wider">{fav.seed_name}</div>
                            </div>
                            <span className="text-[8px] font-bold text-primary uppercase tracking-widest bg-primary/10 px-2 py-0.5 rounded whitespace-nowrap">Seed Source</span>    
                          </div>

                          <div className="flex gap-3 overflow-x-auto pb-1 no-scrollbar">  
                            {fav.top_games.map((game: any) => (
                              <a
                                key={game.appid}
                                href={`https://store.steampowered.com/app/${game.appid}`} 
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex-shrink-0 w-32 group/fav-game"
                              >
                                <div className="relative aspect-video rounded-lg overflow-hidden border border-border/50 group-hover/fav-game:border-primary/50 transition-colors"> 
                                  <GameHeaderImage
                                    appid={game.appid}
                                    isNSFW={game.is_nsfw}
                                    blurNSFW={blurNSFW}
                                    className="w-full h-full object-cover group-hover/fav-game:scale-105 transition-transform"       
                                  />
                                </div>
                                <div className="mt-1.5 px-0.5">
                                  <div className="text-[10px] font-bold truncate group-hover/fav-game:text-primary transition-colors leading-tight">{game.name}</div>
                                </div>
                              </a>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PersonalizationView;
