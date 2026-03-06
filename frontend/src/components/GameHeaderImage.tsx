import React, { useState, useEffect } from 'react';

interface GameHeaderImageProps {
  appid: number;
  isNSFW?: boolean;
  blurNSFW?: boolean;
  className?: string;
  alt?: string;
}

const GameHeaderImage: React.FC<GameHeaderImageProps> = ({ 
  appid, isNSFW, blurNSFW, className, alt = "Game Header" 
}) => {
  const [src, setSrc] = useState(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setSrc(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`);
    setRetryCount(0);
  }, [appid]);

  const handleError = () => {
    if (retryCount === 0) {
      // Try Fastly CDN (often more reliable for newer games)
      setSrc(`https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`);
      setRetryCount(1);
    } else if (retryCount === 1) {
      // Try capsule image
      setSrc(`https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appid}/capsule_616x353.jpg`);
      setRetryCount(2);
    } else if (retryCount === 2) {
      setSrc(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/capsule_231x87.jpg`);
      setRetryCount(3);
    } else if (retryCount === 3) {
      setSrc(`https://cdn.akamai.steamstatic.com/steam/apps/${appid}/capsule_184x69.jpg`);
      setRetryCount(4);
    } else {
      // Final fallback to placeholder
      setSrc('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23262626"/%3E%3C/svg%3E');
    }
  };

  return (
    <img
      src={src}
      className={`${className} ${isNSFW && blurNSFW ? 'blur-2xl scale-110' : ''}`}
      onError={handleError}
      alt={alt}
    />
  );
};

export default GameHeaderImage;
