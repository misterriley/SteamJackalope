import React, { useState, useEffect, useRef } from 'react';
import { 
  User, 
  Search, 
  Table as TableIcon, 
  LineChart, 
  ArrowRight, 
  RefreshCcw, 
  Save, 
  ExternalLink,
  ChevronDown,
  Trash2,
  AlertCircle,
  CheckCircle2,
  Download,
  Plus,
  RotateCcw,
  ThumbsUp,
  ThumbsDown,
  Hash,
  Compass,
  Anchor
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchGames, getMetadata, getTermLinks, API_BASE_URL } from '../api';

interface GameVerification {
  appid: number;
  name: string;
  predicted_rating: number;
  actual_rating: number;
  ignore: boolean;
  user_review_text?: string;
  user_voted_up?: boolean;
  playtime_forever: number;
  is_manual?: boolean;
  is_nsfw?: boolean;
}

interface PersonalizationViewProps {
  onApply?: (profile: any) => void;
}

interface VerificationTableProps {
  data: GameVerification[];
  title: string;
  showPlaytime?: boolean;
  blurNSFW?: boolean;
  sortConfig: { key: keyof GameVerification; direction: 'asc' | 'desc' };
  onSort: (key: keyof GameVerification) => void;
  onRatingChange: (appid: number, rating: number) => void;
  onIgnoreChange: (appid: number, ignore: boolean) => void;
  onDelete?: (appid: number) => void;
}

const VerificationTable: React.FC<VerificationTableProps> = ({ 
  data, title, showPlaytime = true, blurNSFW = true, sortConfig, onSort, onRatingChange, onIgnoreChange, onDelete 
}) => (
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
          {data.map((game) => (
            <tr key={game.appid} className={`hover:bg-secondary/30 transition-colors ${game.ignore ? 'opacity-40' : ''}`}>
              <td className="px-6 py-3">
                <a href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="block hover:opacity-80 transition-opacity">
                  <img 
                    src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.appid}/header.jpg`} 
                    alt={game.name} 
                    className={`w-16 h-8 object-cover rounded shadow-sm border border-border/50 ${game.is_nsfw && blurNSFW ? 'blur-sm scale-110' : ''}`}
                    onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E')}
                  />
                </a>
              </td>
              <td className="px-6 py-3">
                <a href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="group">
                  <div className="font-bold text-sm leading-tight group-hover:text-primary transition-colors">{game.name}</div>
                  <div className="text-[10px] text-muted-foreground">AppID: {game.appid}</div>
                </a>
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
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const PersonalizationView: React.FC<PersonalizationViewProps> = ({ onApply }) => {
  const [termLinks, setTermLinks] = useState<Record<string, string>>({});

  useEffect(() => {
    const fetchLinks = async () => {
      try {
        const links = await getTermLinks();
        setTermLinks(links);
      } catch (err) {
        console.error("Failed to fetch term links", err);
      }
    };
    fetchLinks();
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
  const [sortConfig, setSortConfig] = useState<{ key: keyof GameVerification; direction: 'asc' | 'desc' }>({
    key: 'predicted_rating',
    direction: 'desc'
  });

  const [manualSearch, setManualSearch] = useState('');
  const [manualSearchResults, setManualSearchResults] = useState<string[]>([]);
  const [showManualResults, setShowManualResults] = useState(false);
  const manualSearchRef = useRef<HTMLDivElement>(null);

  // Sync blur setting from session storage (where Filters.tsx saves it)
  const [blurNSFW, setBlurNSFW] = useState(true);
  useEffect(() => {
    const checkFilters = () => {
      const saved = sessionStorage.getItem('recommendations_filters');
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (parsed.remove_nsfw !== undefined) {
            setBlurNSFW(parsed.remove_nsfw);
          }
        } catch (e) {}
      }
    };
    checkFilters();
    // Also listen for storage changes in other tabs or components
    window.addEventListener('storage', checkFilters);
    // Polling as a fallback for same-tab session storage updates
    const interval = setInterval(checkFilters, 1000);
    return () => {
      window.removeEventListener('storage', checkFilters);
      clearInterval(interval);
    };
  }, []);

  // Persistence: Save state on change
  useEffect(() => {
    const state = { step, steamId, reviewHtml, games, insights, status };
    sessionStorage.setItem('personalization_state', JSON.stringify(state));
  }, [step, steamId, reviewHtml, games, insights, status]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (manualSearchRef.current && !manualSearchRef.current.contains(event.target as Node)) {
        setShowManualResults(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    let interval: any;
    if (step === 1.5 && steamId) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/user/status/${steamId}`);
          const data = await res.json();
          setStatus(data);
          if (data.has_soft_labels) {
            fetchVerificationData(steamId);
          }
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
      if (profileMatch) {
        cleanId = profileMatch[1];
      } else if (idMatch) {
        cleanId = idMatch[1];
      }
    }

    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/user/fetch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steam_id: cleanId, review_html: reviewHtml })
      });
      if (!res.ok) throw new Error("Failed to start acquisition");
      
      const data = await res.json();
      const sid = data.resolved_as || cleanId;
      setSteamId(sid); 
      setStep(1.5); 
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchVerificationData = async (sid: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/user/verify/${sid}`);
      const data = await res.json();
      const sorted = [...data].sort((a, b) => b.predicted_rating - a.predicted_rating);
      setGames(sorted);
      setStep(2);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleManualAdd = async (gameName: string) => {
    try {
      const meta = await getMetadata([gameName]);
      if (meta && meta.length > 0) {
        const game = meta[0];
        if (games.some(g => g.appid === game.appid)) {
          alert("Game already in list!");
          return;
        }
        
        const newEntry: GameVerification = {
          appid: game.appid,
          name: game.name,
          predicted_rating: 5,
          actual_rating: 7,
          ignore: false,
          playtime_forever: 0,
          is_manual: true
        };
        
        setGames(prev => [newEntry, ...prev]);
        setManualSearch('');
        setShowManualResults(false);

        await fetch(`${API_BASE_URL}/user/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify([{
            steam_id: steamId,
            appid: game.appid,
            actual_rating: 7,
            ignore: false
          }])
        });
      }
    } catch (err) {
      console.error("Failed to add manual game", err);
    }
  };

  const handleManualSearch = async (val: string) => {
    setManualSearch(val);
    if (val.length > 1) {
      const results = await searchGames(val);
      setManualSearchResults(results);
      setShowManualResults(true);
    } else {
      setManualSearchResults([]);
      setShowManualResults(false);
    }
  };

  const handleSort = (key: keyof GameVerification) => {
    let direction: 'asc' | 'desc' = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
    
    const sortedGames = [...games].sort((a, b) => {
      const valA = a[key];
      const valB = b[key];
      if (typeof valA === 'string' && typeof valB === 'string') return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      if (typeof valA === 'number' && typeof valB === 'number') return direction === 'asc' ? valA - valB : valB - valA;
      if (typeof valA === 'boolean' && typeof valB === 'boolean') return direction === 'asc' ? (valA === valB ? 0 : valA ? 1 : -1) : (valA === valB ? 0 : valB ? 1 : -1);
      return 0;
    });
    setGames(sortedGames);
  };

  const handleIgnoreChange = (appid: number, ignore: boolean) => {
    setGames(prev => prev.map(g => g.appid === appid ? { ...g, ignore } : g));
  };

  const handleRatingChange = (appid: number, actual_rating: number) => {
    setGames(prev => prev.map(g => g.appid === appid ? { ...g, actual_rating } : g));
  };

  const handleDeleteManual = async (appid: number) => {
    if (!window.confirm("Remove this manual entry?")) return;
    setGames(prev => prev.filter(g => g.appid !== appid));
    
    // Sync to server immediately so it's removed from the ground truth file
    try {
      await fetch(`${API_BASE_URL}/user/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([{
          steam_id: steamId,
          appid: appid,
          actual_rating: 0,
          ignore: true // Marking as ignore on the server effectively removes it from solver
        }])
      });
    } catch (err) {
      console.error("Failed to sync deletion to server", err);
    }
  };

  const handleSaveAndSolve = async () => {
    setLoading(true);
    try {
      const updates = games.map(g => ({ steam_id: steamId, appid: g.appid, actual_rating: g.actual_rating, ignore: g.ignore }));
      await fetch(`${API_BASE_URL}/user/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      const sRes = await fetch(`${API_BASE_URL}/user/solve/${steamId}`, { method: 'POST' });
      if (!sRes.ok) throw new Error("Solver failed");
      const iRes = await fetch(`${API_BASE_URL}/user/insights/${steamId}`);
      const iData = await iRes.json();
      setInsights(iData);
      setStep(3);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!insights) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(insights, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `user_${steamId}_taste_profile.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const manualGames = games.filter(g => g.is_manual);
  const libraryGames = games.filter(g => !g.is_manual);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-center gap-4 mb-12">
        {[1, 2, 3].map((s) => {
          const isCompleted = (s === 1 && status?.has_soft_labels) || (s === 2 && insights);
          const isCurrent = step === s || (step === 1.5 && s === 1);
          const canJump = isCompleted || s < step;

          return (
            <React.Fragment key={s}>
              <button 
                onClick={() => canJump && setStep(s)}
                disabled={!canJump}
                className={`flex items-center gap-2 transition-all ${isCurrent ? 'text-primary' : canJump ? 'text-muted-foreground hover:text-primary' : 'text-muted-foreground/40'}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${isCurrent ? 'border-primary bg-primary/10' : canJump ? 'border-muted-foreground bg-secondary/50' : 'border-muted-foreground/20'}`}>
                  {s === 1 ? <User size={16} /> : s === 2 ? <TableIcon size={16} /> : <LineChart size={16} />}
                </div>
                <span className="text-sm font-bold uppercase tracking-wider hidden sm:inline">
                  {s === 1 ? 'Acquire' : s === 2 ? 'Verify' : 'Insights'}
                </span>
              </button>
              {s < 3 && <div className={`w-12 h-0.5 ${step > s ? 'bg-primary' : 'bg-muted-foreground/30'}`} />}
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
                    type="text" placeholder="e.g., 76561198039155404 or your vanity name"
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
                  placeholder="Paste the HTML from your Steam Reviews page to include hard labels (Up/Down votes)..."
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
              <p className="text-muted-foreground max-w-sm">We're fetching your library, parsing your reviews, and generating initial predictions. This usually takes 30-60 seconds.</p>
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
                <div className="relative max-w-md" ref={manualSearchRef}>
                  <div className="relative">
                    <Plus className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" size={16} />
                    <input 
                      type="text" placeholder="Add a game manually (e.g. Elden Ring)"
                      className="w-full bg-card border border-border rounded-lg pl-10 pr-4 py-2 text-sm outline-none focus:ring-2 focus:ring-primary/50"
                      value={manualSearch} onChange={(e) => handleManualSearch(e.target.value)}
                      onFocus={() => manualSearch.length > 1 && setShowManualResults(true)}
                    />
                  </div>
                  <AnimatePresence>
                    {showManualResults && manualSearchResults.length > 0 && (
                      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                        className="absolute z-50 w-full mt-1 bg-card border border-border rounded-lg shadow-2xl max-h-60 overflow-y-auto"
                      >
                        {manualSearchResults.map(res => (
                          <button key={res} onClick={() => handleManualAdd(res)} className="w-full text-left px-4 py-2 text-sm hover:bg-secondary transition-colors border-b border-border/50 last:border-0">
                            {res}
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
              <button 
                onClick={handleSaveAndSolve} disabled={loading}
                className="bg-primary text-primary-foreground px-8 py-4 rounded-xl font-bold flex items-center gap-2 hover:opacity-90 shadow-lg shadow-primary/20 disabled:opacity-50 shrink-0"
              >
                <Save size={20} /> {loading ? 'Solving DNA...' : 'Solve My Taste DNA'}
              </button>
            </div>

            {manualGames.length > 0 && (
              <VerificationTable 
                data={manualGames} 
                title="Manual Additions" 
                showPlaytime={false} 
                blurNSFW={blurNSFW}
                sortConfig={sortConfig}
                onSort={handleSort}
                onRatingChange={handleRatingChange}
                onIgnoreChange={handleIgnoreChange}
                onDelete={handleDeleteManual}
              />
            )}
            <VerificationTable 
              data={libraryGames} 
              title="Library Games" 
              blurNSFW={blurNSFW}
              sortConfig={sortConfig}
              onSort={handleSort}
              onRatingChange={handleRatingChange}
              onIgnoreChange={handleIgnoreChange}
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

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
              <div className="lg:col-span-1 space-y-8">
                {/* Metadata Weights */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">
                  <h3 className="text-lg font-bold flex items-center gap-2"><LineChart size={18} className="text-primary" />Metadata Weights</h3>
                  <div className="space-y-4">
                    {Object.entries(insights.metadata || {}).filter(([key]) => key !== 'semantic').map(([key, val]: [string, any]) => {
                      const denominator = key === 'discovery' ? 1.0 : 3.0;
                      return (
                        <div key={key} className="space-y-1">
                          <div className="flex justify-between text-xs uppercase tracking-widest font-bold">
                            <span>{key}</span>
                            <span className={(val || 0) >= 0 ? 'text-green-500' : 'text-red-500'}>{(val || 0) >= 0 ? '+' : ''}{(val || 0).toFixed(4)}</span>
                          </div>
                          <div className="h-2 bg-secondary rounded-full overflow-hidden">
                            <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(100, (Math.abs(val || 0) / denominator) * 100)}%` }} className={`h-full ${(val || 0) >= 0 ? 'bg-green-500' : 'bg-red-500'}`} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Predictive Tags */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Hash size={18} className="text-primary" />Predictive Tags</h3>
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-green-500">Like...</div>
                      <div className="flex flex-wrap gap-2">
                        {insights.top_tags?.map((t: any) => (
                          termLinks[t.tag] ? (
                            <a 
                              key={t.tag} href={termLinks[t.tag]} target="_blank" rel="noopener noreferrer"
                              className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-medium text-green-500 hover:bg-green-500/20 transition-colors"
                            >
                              {t.tag}
                            </a>
                          ) : (
                            <div key={t.tag} className="px-2 py-1 bg-green-500/10 border border-green-500/20 rounded-lg text-[10px] font-medium text-green-500">
                              {t.tag}
                            </div>
                          )
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-red-500">Dislike...</div>
                      <div className="flex flex-wrap gap-2">
                        {insights.bottom_tags?.map((t: any) => (
                          termLinks[t.tag] ? (
                            <a 
                              key={t.tag} href={termLinks[t.tag]} target="_blank" rel="noopener noreferrer"
                              className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[10px] font-medium text-red-500 hover:bg-red-500/20 transition-colors"
                            >
                              {t.tag}
                            </a>
                          ) : (
                            <div key={t.tag} className="px-2 py-1 bg-red-500/10 border border-red-500/20 rounded-lg text-[10px] font-medium text-red-500">
                              {t.tag}
                            </div>
                          )
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Taste Anchors */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">
                  <h3 className="text-lg font-bold flex items-center gap-2"><Compass size={18} className="text-primary" />Taste Anchors</h3>
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-primary flex items-center gap-1"><Compass size={10} />North Stars</div>
                      <div className="space-y-2">
                        {insights.north_stars?.map((game: any) => (
                          <a key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="block p-2 bg-secondary/30 rounded-lg border border-border/50 hover:border-primary/30 transition-colors">
                            <div className="font-bold text-[10px] truncate">{game.name}</div>
                            <div className="text-[8px] text-muted-foreground uppercase">Alignment: {(game.alignment * 100).toFixed(0)}%</div>
                          </a>
                        ))}
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground flex items-center gap-1"><Anchor size={10} />The Abyss</div>
                      <div className="space-y-2">
                        {insights.abyssal_games?.map((game: any) => (
                          <a key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer" className="block p-2 bg-secondary/30 rounded-lg border border-border/50 hover:border-red-500/30 transition-colors opacity-60">
                            <div className="font-bold text-[10px] truncate">{game.name}</div>
                            <div className="text-[8px] text-muted-foreground uppercase">Anti-Alignment: {(Math.abs(game.alignment) * 100).toFixed(0)}%</div>
                          </a>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* LOVE LIST */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">
                  <h3 className="text-lg font-bold flex items-center gap-2 text-primary"><ThumbsUp size={18} />Games You'll Love</h3>
                  <div className="space-y-3">
                    {insights.top_recommendations?.map((game: any, idx: number) => (
                      <a 
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-primary/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-primary/10 rounded-full text-[10px] font-bold text-primary shrink-0">{idx + 1}</div>
                        <img 
                          src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.appid}/header.jpg`} 
                          className={`w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform ${game.is_nsfw && blurNSFW ? 'blur-sm' : ''}`} 
                          onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E')}
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-primary transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-primary">{Math.round(game.predicted_rating)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>

                {/* HATE LIST */}
                <div className="bg-card border border-border rounded-2xl p-6 space-y-6">
                  <h3 className="text-lg font-bold flex items-center gap-2 text-red-500"><ThumbsDown size={18} />Games You'll Hate</h3>
                  <div className="space-y-3">
                    {insights.bottom_recommendations?.map((game: any, idx: number) => (
                      <a 
                        key={game.appid} href={`https://store.steampowered.com/app/${game.appid}`} target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-secondary/20 p-2 rounded-xl border border-border/30 hover:border-red-500/30 transition-colors group"
                      >
                        <div className="w-6 h-6 flex items-center justify-center bg-red-500/10 rounded-full text-[10px] font-bold text-red-500 shrink-0">{idx + 1}</div>
                        <img 
                          src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.appid}/header.jpg`} 
                          className={`w-12 h-6 object-cover rounded shadow-sm group-hover:scale-105 transition-transform ${game.is_nsfw && blurNSFW ? 'blur-sm' : ''}`} 
                          onError={(e) => (e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E')}
                        />
                        <div className="flex-grow min-w-0">
                          <div className="font-bold text-[11px] truncate group-hover:text-red-500 transition-colors">{game.name}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-red-500">{Math.round(game.predicted_rating)}</div>
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PersonalizationView;
