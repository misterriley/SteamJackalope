import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { RecommendationRequest } from '../types';
import { Settings2, Info, Upload, UserCheck, RotateCcw } from 'lucide-react';

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
  onReset?: () => void;
  tooltip?: string;
}

const Slider = ({ label, value, min, max, step, onChange, onReset, tooltip }: SliderProps) => {
  const [localValue, setLocalValue] = React.useState(value || 0);

  React.useEffect(() => {
    setLocalValue(value || 0);
  }, [value]);

  return (
    <div className="mb-3 group/slider">
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs font-medium text-foreground flex items-center gap-1">
          {label}
          {tooltip && <Tooltip text={tooltip} />}
        </label>
        <div className="flex items-center gap-2">
          {onReset && (
            <button 
              onClick={(e) => {
                e.preventDefault();
                onReset();
              }}
              className="opacity-0 group-hover/slider:opacity-100 transition-opacity text-muted-foreground hover:text-primary p-0.5"
              title="Reset to default"
            >
              <RotateCcw size={10} />
            </button>
          )}
          <span className="text-[10px] font-mono text-primary font-bold min-w-8 text-right">
            {(localValue || 0).toFixed(2)}
          </span>
        </div>
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

const SegmentedControl = ({ label, options, value, onChange }: { label: string, options: { label: string, value: string }[], value: string, onChange: (v: any) => void }) => (
  <div className="mb-4">
    <label className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2 block">{label}</label>
    <div className="grid grid-cols-3 bg-secondary/50 p-1 rounded-lg border border-border/50">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`py-1 text-[10px] font-bold rounded-md transition-all ${
            value === opt.value 
              ? 'bg-primary text-primary-foreground shadow-sm' 
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  </div>
);

interface FiltersProps {
  filters: RecommendationRequest;
  onChange: (filters: RecommendationRequest) => void;
  onSearch: () => void;
  loading: boolean;
  onProfileUpload?: (profile: any) => void;
  onProfileClear?: () => void;
}

const Filters: React.FC<FiltersProps> = ({ filters, onChange, onProfileUpload, onProfileClear }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (key: keyof RecommendationRequest, value: any) => {
    onChange({ ...filters, [key]: value });
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && onProfileUpload) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const profile = JSON.parse(e.target?.result as string);
          onProfileUpload(profile);
        } catch (err) {
          alert("Failed to parse taste profile JSON.");
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="bg-card border border-border rounded-xl shadow-sm lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)] flex flex-col overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Settings2 size={18} className="text-primary" />
          <h2 className="text-base font-bold">Preferences</h2>
        </div>
        {filters.vibe_vector && (
          <div className="flex items-center gap-1 text-[10px] text-green-500 font-bold bg-green-500/10 px-2 py-0.5 rounded-full border border-green-500/20">
            <UserCheck size={10} />
            PERSONALIZED
          </div>
        )}
      </div>

      <div className="p-4 lg:overflow-y-auto flex-grow custom-scrollbar space-y-4">
        <div className="pb-2 border-b border-border/50">
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">Personalization</h3>
          <input 
            type="file" 
            accept=".json" 
            className="hidden" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-secondary/50 hover:bg-secondary rounded-lg text-xs font-medium transition-colors border border-border/50"
          >
            <Upload size={14} />
            {filters.vibe_vector ? "Change Taste Profile" : "Upload Taste Profile"}
          </button>
          {filters.vibe_vector && (
            <button 
              onClick={() => {
                // Reset to default weights while removing personalization
                onChange({ 
                  ...filters,
                  alpha: 1.0,
                  beta: 1.0,
                  quality_pref: 1.0,
                  age_pref: 0.0,
                  pop_pref: 0.0,
                  disc_pref: 0.0,
                  length_pref: 0.0,
                  difficulty_pref: 0.0,
                  price_pref: 0.0,
                  vibe_vector: undefined, 
                  semantic_vibe_vector: undefined,
                  metadata_weights: undefined, 
                  library_appids: [], 
                  rated_appids: [], 
                  profile_filter: 'none' 
                });
                if (onProfileClear) onProfileClear();
              }}
              className="w-full mt-2 text-[10px] text-muted-foreground hover:text-destructive underline transition-colors"
            >
              Clear personalization
            </button>
          )}
        </div>
        <div>
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Core Weights</h3>
          <Slider 
            label="Semantic (Prompt)" 
            value={filters.alpha} 
            min={0} max={2} step={0.01} 
            onChange={(v: number) => handleChange('alpha', v)}
            onReset={() => handleChange('alpha', 1.0)}
            tooltip="Absolute points contributed by the text prompt match."
          />
          <Slider 
            label="Tag Match (Vibes)" 
            value={filters.beta} 
            min={0} max={2} step={0.01} 
            onChange={(v: number) => handleChange('beta', v)}
            onReset={() => handleChange('beta', 1.0)}
            tooltip="Absolute points contributed by the tag vector match (Seeds/DNA)."
          />
          <Slider 
            label="Quality" 
            value={filters.quality_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('quality_pref', v)}
            onReset={() => handleChange('quality_pref', 1.0)}
            tooltip="Absolute points per SD of Bayesian Quality score."
          />
          <Slider 
            label="Discovery" 
            value={filters.disc_pref} 
            min={-1} max={1} step={0.1} 
            onChange={(v: number) => handleChange('disc_pref', v)}
            onReset={() => handleChange('disc_pref', 0.0)}
            tooltip="Controls Bayesian regularization strength (Left = Safe/Mainstream, Right = Wild Cards/Discovery)."
          />
        </div>

        <div className="pt-2 border-t border-border/50">
          <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3">Game Stats</h3>
          <Slider 
            label="Popularity" 
            value={filters.pop_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('pop_pref', v)}
            onReset={() => handleChange('pop_pref', 0.0)}
            tooltip="Points per SD of Review Count."
          />
          <Slider 
            label="Price" 
            value={filters.price_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('price_pref', v)}
            onReset={() => handleChange('price_pref', 0.0)}
            tooltip="Points per SD of Price (Left = Cheap, Right = Expensive)."
          />
          <Slider 
            label="Release Date" 
            value={filters.age_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('age_pref', v)}
            onReset={() => handleChange('age_pref', 0.0)}
            tooltip="Points per SD of Release Date."
          />
          <Slider 
            label="Length" 
            value={filters.length_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('length_pref', v)}
            onReset={() => handleChange('length_pref', 0.0)}
            tooltip="Points per SD of Estimated Playtime."
          />
          <Slider 
            label="Difficulty" 
            value={filters.difficulty_pref} 
            min={-2} max={2} step={0.01} 
            onChange={(v: number) => handleChange('difficulty_pref', v)}
            onReset={() => handleChange('difficulty_pref', 0.0)}
            tooltip="Points per SD of Predicted Difficulty."
          />
        </div>

        <div className="pt-2 border-t border-border/50">
          <div className="flex justify-between items-center mb-1">
             <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Filters</h3>
             {filters.library_appids && filters.library_appids.length > 0 && (
               <span className="text-[8px] font-bold text-green-500 bg-green-500/10 px-1.5 py-0.5 rounded border border-green-500/20">
                 {filters.library_appids.length} games in profile
               </span>
             )}
          </div>
          <SegmentedControl 
            label="Filter Profile Games"
            options={[
              { label: 'None', value: 'none' },
              { label: `Rated (${filters.rated_appids?.length || 0})`, value: 'rated' },
              { label: `All (${filters.library_appids?.length || 0})`, value: 'all' }
            ]}
            value={filters.profile_filter || 'none'}
            onChange={(v) => handleChange('profile_filter', v)}
          />
          <Toggle label="English Only" checked={filters.english_only} onChange={(v: boolean) => handleChange('english_only', v)} />
          <Toggle label="Hide VR-Only" checked={filters.remove_vr} onChange={(v: boolean) => handleChange('remove_vr', v)} />
          <Toggle label="Blur NSFW" checked={filters.remove_nsfw} onChange={(v: boolean) => handleChange('remove_nsfw', v)} />
          <Toggle label="Hide Utilities" checked={filters.remove_utilities} onChange={(v: boolean) => handleChange('remove_utilities', v)} />
          <Toggle label="Released Only" checked={filters.remove_unreleased} onChange={(v: boolean) => handleChange('remove_unreleased', v)} />
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
