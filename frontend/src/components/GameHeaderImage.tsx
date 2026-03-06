import React, { useState, useEffect } from 'react';

interface GameHeaderImageProps {
  appid: number;
  isNSFW?: boolean;
  blurNSFW?: boolean;
  className?: string;
  alt?: string;
}

const GameHeaderImage: React.FC<GameHeaderImageProps> = ({ 
  appid, isNSFW, blurNSFW, className, alt = "Game" 
}) => {
  const [src, setSrc] = useState<string>(`https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`);
  const [retryCount, setRetryCount] = useState(0);
  const [failedAll, setFailedAll] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setSrc(`https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`);
    setRetryCount(0);
    setFailedAll(false);
    setIsLoaded(false);
  }, [appid]);

  const handleError = () => {
    setRetryCount(prev => {
      const nextCount = prev + 1;
      
      const fallbacks = [
        `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`,
        `https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`,
        `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/library_capsule.jpg`,
        `https://cdn.cloudflare.steamstatic.com/steam/apps/${appid}/header.jpg`
      ];

      if (nextCount <= fallbacks.length) {
        console.warn(`[IMAGE_RETRY] AppID ${appid} attempt ${nextCount}: ${fallbacks[nextCount-1]}`);
        setSrc(fallbacks[nextCount - 1]);
      } else {
        console.error(`[IMAGE_FAILED] AppID ${appid} - all fallbacks failed.`);
        setFailedAll(true);
      }
      return nextCount;
    });
  };

  const placeholder = (
    <div className={`${className} bg-secondary flex flex-col items-center justify-center p-2 text-center border border-border/50 relative overflow-hidden`}>
      <div className="absolute inset-0 opacity-10 flex items-center justify-center font-black text-4xl select-none uppercase tracking-tighter italic">
        {appid % 1000}
      </div>
      <span className="relative z-10 text-[8px] font-black text-muted-foreground/60 uppercase tracking-widest leading-tight line-clamp-3 px-1">
        {alt}
      </span>
    </div>
  );

  if (failedAll) return placeholder;

  return (
    <div className={`${className} relative bg-secondary/20 overflow-hidden`}>
      {!isLoaded && (
        <div className="absolute inset-0 z-0">
          {placeholder}
        </div>
      )}
      <img
        src={src}
        className={`w-full h-full object-cover transition-opacity duration-500 z-10 relative ${isLoaded ? 'opacity-100' : 'opacity-0'} ${isNSFW && blurNSFW ? 'blur-2xl scale-110' : ''}`}
        onLoad={() => setIsLoaded(true)}
        onError={handleError}
        alt={alt}
        loading="lazy"
      />
    </div>
  );
};

export default GameHeaderImage;
