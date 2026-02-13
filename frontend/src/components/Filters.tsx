import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { RecommendationRequest } from '../types';
import { Settings2, Info } from 'lucide-react';

// Robust Tooltip component using Portals to avoid overflow clipping
const Tooltip = ({ text }: { text: string }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  const getPosition = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const tooltipWidth = 192; // w-48 = 12rem = 192px
      
      // Default: Center above the icon
      let left = rect.left + rect.width / 2;
      let top = rect.top - 8;

      // Boundary Detection
      if (left - tooltipWidth / 2 < 10) left = tooltipWidth / 2 + 10;
      if (left + tooltipWidth / 2 > window.innerWidth - 10) left = window.innerWidth - tooltipWidth / 2 - 10;

      return { top, left };
    }
    return null;
  };

  const handleMouseEnter = () => {
    const pos = getPosition();
    if (pos) {
      setCoords(pos);
      setIsVisible(true);
    }
  };

  useEffect(() => {
    if (isVisible) {
      const update = () => setCoords(getPosition());
      window.addEventListener('scroll', update, true);
      window.addEventListener('resize', update);
      return () => {
        window.removeEventListener('scroll', update, true);
        window.removeEventListener('resize', update);
      };
    }
  }, [isVisible]);

  return (
    <div 
      ref={triggerRef} 
      className="inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => {
        setIsVisible(false);
        setCoords(null);
      }}
    >
      <Info size={12} className="text-muted-foreground cursor-help hover:text-primary transition-colors" />
      {isVisible && coords && createPortal(
        <div 
          className="fixed z-[10000] px-3 py-2 bg-popover text-popover-foreground text-[10px] rounded-lg shadow-2xl w-48 -translate-x-1/2 -translate-y-full pointer-events-none border border-border animate-in fade-in zoom-in duration-150"
          style={{ top: coords.top, left: coords.left }}
        >
          {text}
          {/* Arrow */}
          <div 
            className="absolute top-full border-4 border-transparent border-t-popover"
            style={{ 
              left: `calc(50% + ${triggerRef.current ? (triggerRef.current.getBoundingClientRect().left + triggerRef.current.getBoundingClientRect().width/2) - coords.left : 0}px)`, 
              transform: 'translateX(-50%)' 
            }}
          />
        </div>,
        document.body
      )}
    </div>
  );
};

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  tooltip?: string;
}

const Slider = ({ label, value, min, max, step, onChange, tooltip }: SliderProps) => {
  const [localValue, setLocalValue] = React.useState(value);

  React.useEffect(() => {
    setLocalValue(value);
  }, [value]);

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-foreground flex items-center gap-1">
          {label}
          {tooltip && <Tooltip text={tooltip} />}
        </label>
        <span className="text-[10px] font-mono text-primary font-bold">{localValue.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={localValue}
        onInput={(e) => setLocalValue(parseFloat((e.target as HTMLInputElement).value))}
        onChange={(e) => onChange(parseFloat((e.target as HTMLInputElement).value))}
        className="w-full h-1 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
      />
    </div>
  );
};

const Toggle = ({ label, checked, onChange }: { label: string, checked: boolean, onChange: (v: boolean) => void }) => (
  <label className="flex items-center justify-between cursor-pointer mb-2 group">
    <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">{label}</span>
    <div className="relative inline-flex items-center">
      <input
        type="checkbox"
        className="sr-only peer"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <div className="w-7 h-4 bg-secondary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-primary"></div>
    </div>
  </label>
);

interface FiltersProps {
  filters: RecommendationRequest;
  onChange: (filters: RecommendationRequest) => void;
  onSearch: () => void;
  loading: boolean;
}

const Filters: React.FC<FiltersProps> = ({ filters, onChange }) => {
  const handleChange = (key: keyof RecommendationRequest, value: any) => {
    onChange({ ...filters, [key]: value });
  };

  return (
    <div className="bg-card border border-border rounded-xl shadow-sm sticky top-20 h-[calc(100vh-6rem)] flex flex-col">
      <div className="p-4 border-b border-border flex items-center gap-2 shrink-0">
        <Settings2 size={18} className="text-primary" />
        <h2 className="text-base font-bold">Preferences</h2>
      </div>

      <div className="p-4 overflow-y-auto flex-grow custom-scrollbar space-y-4">
        <div>
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Core Weights</h3>
          <Slider 
            label="Semantic (Prompt)" 
            value={filters.alpha} 
            min={0} max={2} step={0.1} 
            onChange={(v: number) => handleChange('alpha', v)}
            tooltip="How much to weight the text prompt or seed game descriptions."
          />
          <Slider 
            label="Tags (Seed Games)" 
            value={filters.beta} 
            min={0} max={2} step={0.1} 
            onChange={(v: number) => handleChange('beta', v)}
            tooltip="How much to weight game tags from seed games."
          />
          <Slider 
            label="Quality" 
            value={filters.quality_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('quality_pref', v)}
            tooltip="Weight for reviews and rating quality."
          />
          <Slider 
            label="Discovery" 
            value={filters.disc_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('disc_pref', v)}
            tooltip="Negative = Popular/Safe, Positive = Hidden Gems/Niche."
          />
        </div>

        <div className="pt-2 border-t border-border/50">
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Game Stats</h3>
          <Slider 
            label="Popularity" 
            value={filters.pop_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('pop_pref', v)}
            tooltip="Negative = Niche titles, Positive = Blockbusters."
          />
          <Slider 
            label="Age" 
            value={filters.age_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('age_pref', v)}
            tooltip="Negative = Classic, Positive = Recent."
          />
          <Slider 
            label="Length" 
            value={filters.length_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('length_pref', v)}
            tooltip="Negative = Short sessions, Positive = Long epics."
          />
          <Slider 
            label="Difficulty" 
            value={filters.difficulty_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('difficulty_pref', v)}
            tooltip="Negative = Relaxing, Positive = Challenging."
          />
        </div>

        <div className="pt-2 border-t border-border/50">
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Filters</h3>
          <Toggle label="English Only" checked={filters.english_only} onChange={(v: boolean) => handleChange('english_only', v)} />
          <Toggle label="Hide VR-Only" checked={filters.remove_vr} onChange={(v: boolean) => handleChange('remove_vr', v)} />
          <Toggle label="Hide NSFW" checked={filters.remove_nsfw} onChange={(v: boolean) => handleChange('remove_nsfw', v)} />
          <Toggle label="Hide Utilities" checked={filters.remove_utilities} onChange={(v: boolean) => handleChange('remove_utilities', v)} />
          <Toggle label="Released Only" checked={filters.remove_unreleased} onChange={(v: boolean) => handleChange('remove_unreleased', v)} />
          <Toggle label="Visualize Contributions" checked={!!filters.debug} onChange={(v: boolean) => handleChange('debug', v)} />
        </div>
      </div>

      <div className="p-3 bg-secondary/20 border-t border-border mt-auto shrink-0">
        <div className="text-[9px] text-muted-foreground text-center uppercase tracking-tighter">
          Live Updating Enabled
        </div>
      </div>
    </div>
  );
};

export default Filters;
