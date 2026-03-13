import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle, 
  Star, 
  XCircle,
  ChevronRight,
  Clock,
  Heart,
  Trash2
} from 'lucide-react';
import { updateUserVerify } from '../api';
import { type GameStatus } from '../types';

interface ContextMenuProps {
  x: number;
  y: number;
  appid: number;
  steamId: string;
  currentStatus?: GameStatus;
  onClose: () => void;
  onUpdate?: (appid: number, status: GameStatus, rating?: number) => void;
}

const ContextMenu: React.FC<ContextMenuProps> = ({ x, y, appid, steamId, currentStatus, onClose, onUpdate }) => {
  const [showRatingSubmenu, setShowRatingSubmenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  const handleStatusChange = async (status: GameStatus, rating: number = 5) => {
    try {
      const isIgnore = status === 'ignored';
      const isDelete = status === 'deleted';
      await updateUserVerify(steamId, appid, rating, isIgnore, status, "", isDelete);
      if (onUpdate) {
        onUpdate(appid, status, rating);
      }
      onClose();
    } catch (err) {
      console.error("Failed to update game status", err);
    }
  };

  const handleDelete = () => {
    if (window.confirm("Are you sure you want to remove this game from your catalogue? This will delete your manual rating/status for this game.")) {
      handleStatusChange('deleted');
    }
  };

  const menuItems = [
    { id: 'backlog', label: 'Add to Backlog', icon: <Clock size={14} />, color: 'text-primary' },
    { id: 'wishlist', label: 'Add to Wishlist', icon: <Heart size={14} />, color: 'text-pink-500' },
    { id: 'played', label: 'Mark as Played', icon: <CheckCircle size={14} />, color: 'text-green-500' },
    { id: 'ignored', label: 'Ignore Game', icon: <XCircle size={14} />, color: 'text-red-500' },
  ];

  const ratings = Array.from({ length: 11 }, (_, i) => 10 - i);

  // Adjust position if menu goes off screen
  const menuWidth = 180;
  const menuHeight = 200;
  const adjustedX = Math.min(x, window.innerWidth - menuWidth - 20);
  const adjustedY = Math.min(y, window.innerHeight - menuHeight - 20);

  return (
    <motion.div
      ref={menuRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      style={{ top: adjustedY, left: adjustedX }}
      className="fixed z-[1000] w-[180px] bg-popover/95 backdrop-blur-xl border border-border shadow-2xl rounded-xl py-1.5"
    >
      {!steamId ? (
        <div className="px-4 py-3 text-center space-y-2">
          <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Library Required</div>
          <p className="text-[9px] text-muted-foreground leading-tight">Please enter your SteamID in the <strong>Personalize</strong> tab to rate games.</p>
        </div>
      ) : (
        <>
          {menuItems.map((item) => {
            const isActive = currentStatus === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleStatusChange(item.id as GameStatus)}
                className={`w-full px-3 py-2 flex items-center justify-between text-xs font-medium hover:bg-primary/10 transition-colors ${isActive ? 'bg-primary/5 text-primary' : 'text-foreground'}`}
              >
                <div className="flex items-center gap-2.5">
                  <span className={isActive ? 'text-primary' : item.color}>{item.icon}</span>
                  {item.label}
                </div>
                {isActive && <CheckCircle size={12} className="text-primary" />}
              </button>
            );
          })}

          <div className="h-px bg-border my-1" />

          <div 
            className="relative"
            onMouseEnter={() => setShowRatingSubmenu(true)}
            onMouseLeave={() => setShowRatingSubmenu(false)}
          >
            <button
              className={`w-full px-3 py-2 flex items-center justify-between text-xs font-medium transition-colors ${showRatingSubmenu || currentStatus === 'rated' ? 'bg-primary/10' : 'hover:bg-primary/10'} ${currentStatus === 'rated' ? 'text-primary' : 'text-foreground'}`}
            >
              <div className="flex items-center gap-2.5">
                <Star size={14} className={currentStatus === 'rated' ? 'text-primary' : 'text-yellow-500'} />
                <span>Rate Game</span>
              </div>
              <div className="flex items-center gap-1.5">
                {currentStatus === 'rated' && <CheckCircle size={12} className="text-primary" />}
                <ChevronRight size={14} className={`opacity-40 transition-transform ${showRatingSubmenu ? 'rotate-90' : ''}`} />
              </div>
            </button>

            <AnimatePresence>
              {showRatingSubmenu && (
                <motion.div
                  initial={{ opacity: 0, x: x > window.innerWidth - 250 ? 10 : -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: x > window.innerWidth - 250 ? 10 : -10 }}
                  className={`absolute top-0 w-[60px] bg-popover/95 backdrop-blur-xl border border-border shadow-2xl rounded-xl py-1.5 flex flex-col items-center z-[1010] ${x > window.innerWidth - 250 ? 'right-full mr-0.5' : 'left-full ml-0.5'}`}
                >
                  {ratings.map((r) => (
                    <button
                      key={r}
                      onClick={() => handleStatusChange('rated', r)}
                      className="w-full px-3 py-1.5 text-center text-xs font-bold hover:bg-primary/10 transition-colors text-foreground"
                    >
                      {r}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="h-px bg-border my-1" />

          <button
            onClick={handleDelete}
            className="w-full px-3 py-2 flex items-center gap-2.5 text-xs font-medium text-red-500 hover:bg-red-500/10 transition-colors"
          >
            <Trash2 size={14} />
            <span>Remove from Catalogue</span>
          </button>
        </>
      )}
    </motion.div>
  );
};

export default ContextMenu;
