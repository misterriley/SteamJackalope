import React, { useState, useEffect, useRef } from 'react';
import { ImageOff } from 'lucide-react';

interface GameHeaderImageProps {
  appid: number;
  header_image?: string;
  isNSFW?: boolean;
  blurNSFW?: boolean;
  className?: string;
  alt?: string;
}

const GameHeaderImage: React.FC<GameHeaderImageProps> = ({ 
  appid, header_image, isNSFW, blurNSFW, className, alt = "Game" 
}) => {
  const getInitialSrc = () => {
    if (header_image && header_image.startsWith('http')) return header_image;
    return `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`;
  };

  const [src, setSrc] = useState<string>(getInitialSrc());
  const [failedAll, setFailedAll] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const newSrc = getInitialSrc();
    setSrc(newSrc);
    setFailedAll(false);
    setIsLoaded(false);
  }, [appid, header_image]);

  useEffect(() => {
    if (imgRef.current?.complete) {
      setIsLoaded(true);
    }
  }, [src]);

  const handleError = () => {
    const fallbacks = [
      `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`,
      `https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`,
      `https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/${appid}/library_capsule.jpg`,
      `https://cdn.cloudflare.steamstatic.com/steam/apps/${appid}/header.jpg`
    ];

    const currentIdx = fallbacks.indexOf(src);
    const nextIdx = currentIdx + 1;

    if (nextIdx < fallbacks.length) {
      console.warn(`[IMAGE_RETRY] AppID ${appid} attempting fallback: ${fallbacks[nextIdx]}`);
      setSrc(fallbacks[nextIdx]);
    } else {
      console.error(`[IMAGE_FAILED] AppID ${appid} - all fallbacks failed.`);
      setFailedAll(true);
    }
  };

  const placeholder = (
    <div className={`${className} bg-slate-800 flex flex-col items-center justify-center p-4 text-center border border-slate-700 relative overflow-hidden group-hover:bg-slate-700 transition-colors`}>
      <div className="absolute inset-0 opacity-5 flex items-center justify-center font-black text-6xl select-none uppercase tracking-tighter italic">
        {appid % 10000}
      </div>
      <ImageOff size={24} className="text-slate-600 mb-2 opacity-50" />
      <span className="relative z-10 text-[10px] font-bold text-slate-400 uppercase tracking-widest leading-tight line-clamp-2 px-2">
        {alt}
      </span>
      <div className="mt-2 text-[8px] font-bold text-slate-500/50 uppercase tracking-tighter">
        AppID: {appid}
      </div>
    </div>
  );

  if (failedAll) return placeholder;

  return (
    <div className={`${className} relative bg-slate-900 overflow-hidden group`}>
      {!isLoaded && (
        <div className="absolute inset-0 z-0">
          {placeholder}
        </div>
      )}
      <img
        ref={imgRef}
        src={src}
        className={`w-full h-full object-cover transition-all duration-700 z-10 relative ${isLoaded ? 'opacity-100 scale-100' : 'opacity-0 scale-110'} ${isNSFW && blurNSFW ? 'blur-2xl scale-125' : ''}`}
        onLoad={() => setIsLoaded(true)}
        onError={handleError}
        alt={alt}
        loading="lazy"
      />
    </div>
  );
};

export default GameHeaderImage;
