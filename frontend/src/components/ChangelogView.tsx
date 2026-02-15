import React, { useState, useEffect } from 'react';
import MarkdownView from './MarkdownView';
import { getChangelog } from '../api';
import { Loader2 } from 'lucide-react';

const ChangelogView: React.FC = () => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchChangelog = async () => {
      try {
        const md = await getChangelog();
        setContent(md);
      } catch (err) {
        console.error('Failed to fetch changelog:', err);
        setError('Failed to load changelog content.');
      } finally {
        setLoading(false);
      }
    };

    fetchChangelog();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <p className="text-muted-foreground animate-pulse">Loading changelog...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-destructive/10 border border-destructive/20 rounded-xl p-8 text-center">
        <p className="text-destructive font-medium">{error}</p>
        <p className="text-sm text-muted-foreground mt-2">Please ensure the backend server is running and CHANGELOG.md exists.</p>
      </div>
    );
  }

  return <MarkdownView content={content} />;
};

export default ChangelogView;
