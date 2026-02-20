import React from 'react';

interface ViolinPlotProps {
  ratingsWith: number[];
  ratingsWithout: number[];
  tagName: string;
}

const ViolinPlot: React.FC<ViolinPlotProps> = ({ ratingsWith, ratingsWithout, tagName }) => {
  // Simple Kernel Density Estimation
  const getDensity = (data: number[], points: number[]) => {
    if (data.length === 0) return points.map(() => 0);
    // Adaptive bandwidth based on data size
    const bandwidth = Math.max(0.5, 1.06 * Math.sqrt(data.length === 1 ? 1 : data.length) ** -0.2); 
    
    return points.map(x => {
      return data.reduce((acc, val) => {
        const diff = (x - val) / bandwidth;
        return acc + Math.exp(-0.5 * diff * diff);
      }, 0) / (data.length * bandwidth * Math.sqrt(2 * Math.PI));
    });
  };

  const xPoints = Array.from({ length: 40 }, (_, i) => i * 10 / 39);
  const densityWith = getDensity(ratingsWith, xPoints);
  const densityWithout = getDensity(ratingsWithout, xPoints);

  const maxDensity = Math.max(...densityWith, ...densityWithout, 0.01);

  // SVG dimensions
  const width = 300;
  const height = 180;
  const margin = { top: 30, right: 20, bottom: 30, left: 40 };

  const scaleX = (d: number, violinIndex: number) => {
    const violinWidth = (width - margin.left - margin.right) / 2;
    const centerX = margin.left + violinWidth / 2 + violinIndex * violinWidth;
    const offset = (d / maxDensity) * (violinWidth / 2 - 10);
    return { centerX, left: centerX - offset, right: centerX + offset };
  };

  const scaleY = (x: number) => {
    return height - margin.bottom - (x / 10) * (height - margin.top - margin.bottom);
  };

  const generatePath = (densities: number[], violinIndex: number) => {
    const { centerX } = scaleX(0, violinIndex);
    const leftPoints: string[] = [];
    const rightPoints: string[] = [];

    xPoints.forEach((x, i) => {
      const { left, right } = scaleX(densities[i], violinIndex);
      const y = scaleY(x);
      leftPoints.push(`${left},${y}`);
      rightPoints.unshift(`${right},${y}`);
    });

    return `M ${centerX},${scaleY(xPoints[0])} ` + leftPoints.join(' L ') + ' L ' + rightPoints.join(' L ') + ' Z';
  };

  const meanWith = ratingsWith.length > 0 ? ratingsWith.reduce((a, b) => a + b, 0) / ratingsWith.length : 0;
  const meanWithout = ratingsWithout.length > 0 ? ratingsWithout.reduce((a, b) => a + b, 0) / ratingsWithout.length : 0;

  return (
    <div className="bg-card border border-primary/30 rounded-2xl shadow-2xl p-4 w-[340px] backdrop-blur-xl">
      <div className="flex justify-between items-center mb-4 px-2">
        <div className="text-xs font-bold uppercase tracking-widest text-primary truncate max-w-[200px]">
          {tagName}
        </div>
        <div className="text-[10px] font-mono text-muted-foreground">
          N={ratingsWith.length} vs {ratingsWithout.length}
        </div>
      </div>

      <svg width={width} height={height} className="overflow-visible">
        {/* Y Axis */}
        <line x1={margin.left - 5} y1={scaleY(0)} x2={margin.left - 5} y2={scaleY(10)} stroke="#444" strokeWidth="1" />
        {[0, 2.5, 5, 7.5, 10].map(tick => (
          <g key={tick}>
            <line x1={margin.left - 8} y1={scaleY(tick)} x2={margin.left - 5} y2={scaleY(tick)} stroke="#444" strokeWidth="1" />
            <text x={margin.left - 12} y={scaleY(tick) + 4} textAnchor="end" fontSize="9" fill="#888" fontWeight="bold">{tick}</text>
          </g>
        ))}
        <text x={10} y={height / 2} transform={`rotate(-90, 10, ${height / 2})`} textAnchor="middle" fontSize="9" fill="#666" fontWeight="bold" letterSpacing="0.1em">RATING</text>

        {/* Labels */}
        <text x={margin.left + (width - margin.left - margin.right) / 4} y={height - 5} textAnchor="middle" fontSize="10" fill="#10b981" fontWeight="bold">WITH</text>
        <text x={margin.left + 3 * (width - margin.left - margin.right) / 4} y={height - 5} textAnchor="middle" fontSize="10" fill="#666" fontWeight="bold">WITHOUT</text>

        {/* Violins */}
        <path 
          d={generatePath(densityWith, 0)} 
          fill="rgba(16, 185, 129, 0.2)" 
          stroke="#10b981" 
          strokeWidth="1.5" 
        />
        <path 
          d={generatePath(densityWithout, 1)} 
          fill="rgba(255, 255, 255, 0.05)" 
          stroke="#444" 
          strokeWidth="1.5" 
        />

        {/* Mean Lines */}
        <line 
          x1={scaleX(0, 0).left - 5} y1={scaleY(meanWith)} 
          x2={scaleX(0, 0).right + 5} y2={scaleY(meanWith)} 
          stroke="#10b981" strokeWidth="2" strokeDasharray="2 2"
        />
        <line 
          x1={scaleX(0, 1).left - 5} y1={scaleY(meanWithout)} 
          x2={scaleX(0, 1).right + 5} y2={scaleY(meanWithout)} 
          stroke="#ffffff" strokeWidth="2" strokeDasharray="2 2" opacity="0.3"
        />
      </svg>

      <div className="mt-4 grid grid-cols-2 gap-4 text-center border-t border-border/30 pt-4">
        <div className="space-y-1">
          <div className="text-[9px] uppercase tracking-tighter text-muted-foreground font-bold">Mean (With)</div>
          <div className="text-sm font-mono font-bold text-green-500">{meanWith.toFixed(2)}</div>
        </div>
        <div className="space-y-1">
          <div className="text-[9px] uppercase tracking-tighter text-muted-foreground font-bold">Mean (Without)</div>
          <div className="text-sm font-mono font-bold text-muted-foreground">{meanWithout.toFixed(2)}</div>
        </div>
      </div>
      
      <div className="mt-3 text-center">
        <div className="inline-block px-3 py-1 bg-primary/10 rounded-full text-[10px] font-bold text-primary border border-primary/20">
          Δ { (meanWith - meanWithout).toFixed(2) } Difference
        </div>
      </div>
    </div>
  );
};

export default ViolinPlot;
