import React from 'react';
import { Sparkles, BarChart2, Github, Info, BookOpen, History } from 'lucide-react';

export type TabType = 'recommend' | 'lists' | 'about' | 'methodology' | 'changelog';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, activeTab, onTabChange }) => {
  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30 selection:text-primary">
      {/* Header */}
      <header className="border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 shrink-0">
            <div className="w-8 h-8 rounded-full overflow-hidden border border-primary/20">
              <img src="/assets/jackalopeVR.jpg" alt="Jackalope" className="w-full h-full object-cover" />
            </div>
            <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary to-yellow-200">
              SteamJackalope
            </h1>
          </div>

          <nav className="flex items-center gap-1 bg-secondary/50 p-1 rounded-xl overflow-x-auto no-scrollbar max-w-[calc(100vw-12rem)]">
            <button
              onClick={() => onTabChange('recommend')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === 'recommend' 
                  ? 'bg-card text-primary shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Sparkles size={16} />
              Recommendations
            </button>
            <button
              onClick={() => onTabChange('lists')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === 'lists' 
                  ? 'bg-card text-primary shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <BarChart2 size={16} />
              Analysis
            </button>
            <button
              onClick={() => onTabChange('about')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === 'about' 
                  ? 'bg-card text-primary shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Info size={16} />
              About
            </button>
            <button
              onClick={() => onTabChange('methodology')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === 'methodology' 
                  ? 'bg-card text-primary shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <BookOpen size={16} />
              Methodology
            </button>
          </nav>

          <div className="flex items-center gap-2 sm:gap-4">
             <button
              onClick={() => onTabChange('changelog')}
              className={`flex items-center gap-1.5 px-2 sm:px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'changelog'
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
              }`}
             >
              <History size={16} />
              <span className="hidden sm:inline">Changelog</span>
             </button>
             <a 
              href="https://github.com/misterriley/SteamJackalope" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors p-2 rounded-lg hover:bg-secondary/50"
              title="View on GitHub"
             >
              <Github size={20} />
             </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-20 py-10 bg-card/30">
        <div className="container mx-auto px-4 flex flex-col items-center gap-4 text-center">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full overflow-hidden border border-primary/20">
              <img src="/assets/jackalopeVR.jpg" alt="Jackalope" className="w-full h-full object-cover" />
            </div>
            <span className="font-bold">SteamJackalope</span>
          </div>
          <p className="text-sm text-muted-foreground max-w-md">
            A vector-based discovery engine for Steam games. 
            Finding hidden gems through semantic similarity and statistical analysis.
          </p>
          <div className="text-[10px] text-muted-foreground uppercase tracking-widest mt-4">
            &copy; 2026 SteamJackalope Discovery
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
