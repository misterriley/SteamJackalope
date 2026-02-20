import React from 'react';
import { Sparkles, BarChart2, UserCheck, Search, BookOpen, Info } from 'lucide-react';
import type { TabType } from './Layout';

interface SplashViewProps {
  onTabChange: (tab: TabType) => void;
}

const SplashView: React.FC<SplashViewProps> = ({ onTabChange }) => {
  const tools = [
    {
      id: 'recommend' as TabType,
      title: 'Discovery Engine',
      description: 'Find your next favorite game using semantic search, tag analysis, and Bayesian quality scoring.',
      icon: <Search className="text-primary" size={32} />,
      color: 'hover:border-primary/50',
      action: 'Find Games'
    },
    {
      id: 'personalize' as TabType,
      title: 'Taste DNA Solver',
      description: 'Analyze your Steam library to solve for your personal preference weights and mathematical taste profile.',
      icon: <UserCheck className="text-green-400" size={32} />,
      color: 'hover:border-green-400/50',
      action: 'Analyze My Catalogue'
    },
    {
      id: 'lists' as TabType,
      title: 'Steam Data Analyzer',
      description: 'Explore the Steam library through statistical lenses: hardest games, longest experiences, and hidden gems.',
      icon: <BarChart2 className="text-blue-400" size={32} />,
      color: 'hover:border-blue-400/50',
      action: 'Explore Data'
    }
  ];

  const info = [
    { id: 'methodology' as TabType, title: 'Methodology', icon: <BookOpen size={16} /> },
    { id: 'about' as TabType, title: 'About', icon: <Info size={16} /> }
  ];

  return (
    <div className="max-w-5xl mx-auto py-12 px-4">
      <div className="text-center mb-16 space-y-4">
        <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-bold uppercase tracking-widest mb-4">
          <Sparkles size={16} />
          Welcome to SteamJackalope
        </div>
        <h1 className="text-5xl md:text-6xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-b from-foreground to-muted-foreground">
          The Future of Game Discovery
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          A high-fidelity recommendation engine that understands vibes, math, and your personal gaming history.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
        {tools.map((tool) => (
          <div 
            key={tool.id}
            onClick={() => onTabChange(tool.id)}
            className={`group bg-card border border-border rounded-2xl p-8 cursor-pointer transition-all hover:shadow-2xl hover:-translate-y-1 flex flex-col h-full ${tool.color}`}
          >
            <div className="mb-6 p-3 bg-secondary/50 rounded-xl w-fit group-hover:scale-110 transition-transform">
              {tool.icon}
            </div>
            <h2 className="text-2xl font-bold mb-4">{tool.title}</h2>
            <p className="text-muted-foreground mb-8 flex-grow leading-relaxed">
              {tool.description}
            </p>
            <button className="w-full py-3 bg-secondary hover:bg-primary hover:text-primary-foreground rounded-xl font-bold transition-all flex items-center justify-center gap-2">
              {tool.action}
            </button>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        {info.map((item) => (
          <button 
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className="flex items-center gap-2 px-6 py-2 rounded-full border border-border hover:bg-secondary transition-colors text-sm font-medium"
          >
            {item.icon}
            {item.title}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SplashView;
