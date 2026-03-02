import React, { createContext, useContext, useState, type ReactNode, useEffect } from 'react';

interface UserContextType {
  steamId: string | null;
  setSteamId: (id: string | null) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

export const UserProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [steamId, setSteamId] = useState<string | null>(() => {
    const savedProfile = localStorage.getItem('appliedProfile');
    if (savedProfile) {
      try {
        const profile = JSON.parse(savedProfile);
        return profile.steam_id || null;
      } catch (e) {
        return null;
      }
    }
    return localStorage.getItem('lastSteamId') || null;
  });

  useEffect(() => {
    if (steamId) {
      localStorage.setItem('lastSteamId', steamId);
    }
  }, [steamId]);

  return (
    <UserContext.Provider value={{ steamId, setSteamId }}>
      {children}
    </UserContext.Provider>
  );
};
