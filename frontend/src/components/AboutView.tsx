import React, { useState, useEffect } from 'react';
import MarkdownView from './MarkdownView';
import { getAbout } from '../api';
import { Loader2 } from 'lucide-react';

const AboutView: React.FC = () => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAbout = async () => {
      try {
        setLoading(true);
        const data = await getAbout();
        setContent(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch about:', err);
        setError('Failed to load about content.');
      } finally {
        setLoading(false);
      }
    };

    fetchAbout();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <p className="text-muted-foreground animate-pulse">Loading about...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-destructive/10 border border-destructive/20 rounded-xl p-8 max-w-2xl mx-auto text-center">
        <p className="text-destructive font-bold mb-2">Error</p>
        <p className="text-muted-foreground">{error}</p>
      </div>
    );
  }

  return <MarkdownView content={content} />;
};

export default AboutView;
