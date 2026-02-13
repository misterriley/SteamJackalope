import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MarkdownViewProps {
  content: string;
}

const MarkdownView: React.FC<MarkdownViewProps> = ({ content }) => {
  return (
    <div className="bg-card border border-border rounded-xl p-8 shadow-sm max-w-4xl mx-auto overflow-hidden text-left">
      <div className="prose prose-invert prose-sm sm:prose-base max-w-none 
        prose-headings:text-foreground prose-headings:font-bold prose-headings:tracking-tight
        prose-h1:text-3xl prose-h1:mb-8 prose-h1:mt-4
        prose-h2:text-2xl prose-h2:mt-10 prose-h2:mb-4 prose-h2:border-b prose-h2:border-border prose-h2:pb-2
        prose-h3:text-xl prose-h3:mt-8 prose-h3:mb-3
        prose-p:text-muted-foreground prose-p:leading-relaxed prose-p:mb-6
        prose-a:text-primary prose-a:underline prose-a:underline-offset-4 prose-a:decoration-primary/60 hover:prose-a:decoration-primary transition-all font-bold
        prose-strong:text-foreground prose-strong:font-black
        prose-code:text-primary prose-code:bg-secondary/50 prose-code:px-1 prose-code:rounded
        prose-ul:text-muted-foreground prose-ol:text-muted-foreground prose-ul:mb-6 prose-ul:list-disc prose-ul:pl-6
        prose-ol:list-decimal prose-ol:pl-6
        prose-li:my-3 prose-li:leading-relaxed
        prose-table:border prose-table:border-border prose-th:bg-secondary/50 prose-th:p-3 prose-td:p-3 prose-td:border-t prose-td:border-border prose-table:my-8
        prose-img:rounded-xl prose-img:border prose-img:border-border prose-img:my-8
        ">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
          components={{
            // Ensure links open in new tab
            a: ({ node, ...props }) => (
              <a 
                {...props} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-primary font-bold underline underline-offset-4 decoration-primary/60 hover:decoration-primary transition-all"
              />
            ),
            // Explicit paragraph spacing and text styling
            p: ({ node, ...props }) => (
              <p {...props} className="mb-6 last:mb-0 text-muted-foreground leading-relaxed" />
            ),
            // Ensure list styling is respected
            ul: ({ node, ...props }) => (
              <ul {...props} className="mb-6 list-disc pl-6 space-y-2" />
            ),
            ol: ({ node, ...props }) => (
              <ol {...props} className="mb-6 list-decimal pl-6 space-y-2" />
            ),
            li: ({ node, ...props }) => (
              <li {...props} className="text-muted-foreground leading-relaxed" />
            )
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
};

export default MarkdownView;
