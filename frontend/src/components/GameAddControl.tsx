import React, { useState, useEffect, useRef } from 'react';
import { Plus, RefreshCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchGames } from '../api';

interface GameAddControlProps {
  onAdd: (gameName: string) => Promise<void>;
  placeholder?: string;
  className?: string;
}

const GameAddControl: React.FC<GameAddControlProps> = ({ onAdd, placeholder, className = "max-w-md" }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<any>(null);

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
      setLoading(true);
      timeoutRef.current = setTimeout(async () => {
        try {
          const res = await searchGames(val);
          setResults(res);
          setShowResults(true);
        } catch (err) {
          console.error("Search failed", err);
          setResults([]);
        } finally {
          setLoading(false);
        }
      }, 300);
    } else {
      setResults([]);
      setShowResults(false);
      setLoading(false);
    }
  };

  const handleSelect = async (name: string) => {
    setIsAdding(true);
    try {
      await onAdd(name);
      setQuery('');
      setShowResults(false);
    } catch (err) {
      console.error("Failed to add game", err);
    } finally {
      setIsAdding(false);
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
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const selected = activeIndex >= 0 ? results[activeIndex] : results[0];
      if (selected) {
        handleSelect(selected);
      }
    } else if (e.key === 'Escape') {
      setShowResults(false);
    }
  };

  return (
    <div className={`relative ${className}`} ref={containerRef}>
      <div className="relative">
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground flex items-center justify-center w-5 h-5">
          {loading || isAdding ? (
            <RefreshCcw size={16} className="animate-spin text-primary" />
          ) : (
            <Plus size={16} />
          )}
        </div>
        <input 
          type="text" 
          placeholder={placeholder || "Add any game..."}
          className="w-full bg-card border border-border rounded-xl pl-10 pr-4 py-3 text-sm outline-none focus:ring-2 focus:ring-primary/50 transition-all shadow-sm disabled:opacity-50"
          value={query} 
          onChange={(e) => handleSearch(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query.length > 1 && results.length > 0 && setShowResults(true)}
          disabled={isAdding}
        />
      </div>
      <AnimatePresence>
        {showResults && (query.length > 1) && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }} 
            animate={{ opacity: 1, y: 0 }} 
            exit={{ opacity: 0, y: -10 }}
            ref={resultsRef}
            className="absolute z-50 w-full mt-1 bg-card border border-border rounded-xl shadow-2xl max-h-60 overflow-y-auto custom-scrollbar"
          >
            {loading ? (
              <div className="px-4 py-3 text-sm text-muted-foreground italic flex items-center gap-2">
                <RefreshCcw size={14} className="animate-spin" />
                Searching...
              </div>
            ) : results.length > 0 ? (
              results.map((res, idx) => (
                <button 
                  key={res} 
                  onClick={() => handleSelect(res)}
                  className={`w-full text-left px-4 py-2.5 text-sm transition-colors border-b border-border/50 last:border-0 ${
                    idx === activeIndex ? 'bg-primary text-primary-foreground' : 'hover:bg-secondary'
                  }`}
                >
                  {res}
                </button>
              ))
            ) : (
              <div className="px-4 py-3 text-sm text-muted-foreground italic">
                No games found
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GameAddControl;
