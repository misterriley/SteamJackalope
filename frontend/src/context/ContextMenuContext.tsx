import React, { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import ContextMenu from '../components/ContextMenu';
import { AnimatePresence } from 'framer-motion';
import { type GameStatus } from '../types';

interface ContextMenuState {
  x: number;
  y: number;
  appid: number;
  steamId: string;
  currentStatus?: GameStatus;
  onUpdate?: (appid: number, status: GameStatus, rating?: number) => void;
}

interface ContextMenuContextType {
  showContextMenu: (state: ContextMenuState) => void;
  hideContextMenu: () => void;
}

const ContextMenuContext = createContext<ContextMenuContextType | undefined>(undefined);

export const useContextMenu = () => {
  const context = useContext(ContextMenuContext);
  if (!context) {
    throw new Error('useContextMenu must be used within a ContextMenuProvider');
  }
  return context;
};

export const ContextMenuProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);

  const showContextMenu = useCallback((state: ContextMenuState) => {
    setContextMenu(state);
  }, []);

  const hideContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  return (
    <ContextMenuContext.Provider value={{ showContextMenu, hideContextMenu }}>
      {children}
      <AnimatePresence>
        {contextMenu && (
          <ContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            appid={contextMenu.appid}
            steamId={contextMenu.steamId}
            currentStatus={contextMenu.currentStatus}
            onClose={hideContextMenu}
            onUpdate={contextMenu.onUpdate}
          />
        )}
      </AnimatePresence>
    </ContextMenuContext.Provider>
  );
};
