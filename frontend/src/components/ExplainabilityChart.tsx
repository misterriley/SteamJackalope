import React from 'react';
import { ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Label } from 'recharts';

interface DataPoint {
  x: number;
  y: number;
}

interface ExplainabilityChartProps {
  data: DataPoint[];
  title: string;
  xLabel: string;
  isLog?: boolean;
  type?: 'scatter' | 'bar';
  showTrendline?: boolean;
}

const ExplainabilityChart: React.FC<ExplainabilityChartProps> = ({ data, title, xLabel, isLog = false, type = 'scatter', showTrendline = false }) => {
  const currentYear = new Date().getFullYear();

  // Filter out any anomalous 0 values for Year/Age to prevent bunching
  // Also filter out future dates for Age chart
  const filteredData = (title === 'age' || title === 'age_pref' || title === 'Release Date') 
    ? data.filter(p => p.x > 1900 && p.x <= currentYear) 
    : data;

  // Sort data by x for the left-to-right animation effect
  // For log scale, ensure all x values are > 0 to prevent Recharts from crashing/not-rendering
  const sortedData = [...filteredData]
    .map(p => ({ ...p, x: isLog ? Math.max(0.1, p.x) : p.x }))
    .sort((a, b) => a.x - b.x);

  // Generate clean ticks for log scale to prevent "blurry mass"
  const getLogTicks = (data: DataPoint[]) => {
    if (data.length === 0) return [];
    const min = Math.min(...data.map(p => p.x));
    const max = Math.max(...data.map(p => p.x));
    const ticks = [];
    let current = Math.pow(10, Math.floor(Math.log10(min)));
    if (current < 0.1) current = 0.1;
    
    while (current <= max * 10) {
      if (current >= min / 10) {
        ticks.push(current);
      }
      current *= 10;
    }
    return ticks;
  };

  // Generate ticks for Age chart to stop exactly at current year
  const getAgeTicks = (data: DataPoint[]) => {
    if (data.length === 0) return [];
    const min = Math.min(...data.map(p => p.x));
    const start = Math.floor(min / 5) * 5;
    const ticks = [];
    for (let y = start; y <= currentYear; y += 5) {
      if (y >= min - 5) ticks.push(y);
    }
    if (ticks[ticks.length - 1] !== currentYear) ticks.push(currentYear);
    return ticks;
  };

  const isAgeTitle = title === 'age' || title === 'age_pref' || title === 'Release Date';
  const ticks = isLog 
    ? getLogTicks(sortedData) 
    : (isAgeTitle 
        ? getAgeTicks(sortedData) 
        : (title === 'difficulty' ? [0, 5, 10] : undefined));

  // Simple Linear Regression for Trendline
  const getTrendlineData = (data: DataPoint[]) => {
    if (data.length < 2) return [];
    
    // Use log(x) for log-scaled features
    const points = isLog 
      ? data.map(p => ({ x: Math.log10(p.x), y: p.y }))
      : data;

    const n = points.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const p of points) {
      sumX += p.x;
      sumY += p.y;
      sumXY += p.x * p.y;
      sumX2 += p.x * p.x;
    }

    const denominator = (n * sumX2 - sumX * sumX);
    if (Math.abs(denominator) < 1e-10) return [];

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(...points.map(p => p.x));

    return [
      { 
        x: isLog ? Math.pow(10, minX) : minX, 
        y: slope * minX + intercept 
      },
      { 
        x: isLog ? Math.pow(10, maxX) : maxX, 
        y: slope * maxX + intercept 
      }
    ];
  };

  const trendlineData = showTrendline ? getTrendlineData(sortedData) : [];

  // Domain handling
  const minYear = sortedData.length > 0 ? Math.min(...sortedData.map(p => p.x)) : 1990;
  let xDomain: any = ['auto', 'auto'];
  if (isAgeTitle) xDomain = [minYear, currentYear];
  if (title === 'difficulty') xDomain = [0, 10];

  return (
    <div className="w-full h-52 space-y-2 pb-4">
      <div className="flex justify-between items-center px-1">
        <span className="text-[10px] font-bold uppercase tracking-widest text-primary">{title}</span>
        <span className="text-[8px] text-muted-foreground italic">Rating vs {xLabel}{isLog ? ' (Log Scale)' : ''}</span>
      </div>
      <div className="w-full h-full bg-secondary/20 rounded-xl p-2 border border-border/50">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'bar' ? (
            <BarChart data={sortedData} margin={{ top: 10, right: 10, bottom: 30, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis 
                dataKey="x" 
                stroke="#666" 
                fontSize={10} 
                tickFormatter={(val) => val.toFixed(1)}
                ticks={[-1, -0.5, 0, 0.5, 1]}
              >
                <Label value="Discovery Setting" offset={-15} position="insideBottom" style={{ fontSize: '10px', fill: '#888', fontWeight: 'bold' }} />
              </XAxis>
              <YAxis 
                stroke="#666" 
                fontSize={10} 
                name="R2"
                domain={[0, 'auto']}
              >
                <Label value="R² Correlation" angle={-90} position="insideLeft" style={{ fontSize: '10px', fill: '#888', fontWeight: 'bold', textAnchor: 'middle' }} />
              </YAxis>
              <Tooltip 
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', fontSize: '10px' }}
                formatter={(val: any) => [parseFloat(val || 0).toFixed(4), 'R²']}
                labelFormatter={(label: any) => `Discovery: ${parseFloat(label || 0).toFixed(1)}`}
              />
              <Bar 
                dataKey="y" 
                fill="#3b82f6" 
                animationDuration={500}
              >
                {sortedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.y === Math.max(...sortedData.map(d => d.y)) ? '#10b981' : '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <ScatterChart margin={{ top: 10, right: 35, bottom: 30, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
              <XAxis 
                type="number" 
                dataKey="x" 
                name={xLabel} 
                stroke="#666" 
                fontSize={10} 
                scale={isLog ? 'log' : 'auto'}
                domain={xDomain as any}
                allowDataOverflow={true}
                ticks={ticks}
                tickFormatter={(val) => {
                  if (val === 0.1) return '0';
                  if (isAgeTitle && val >= 1800 && val <= 2200) return val.toFixed(0);
                  if (title === 'price') return `$${val.toFixed(0)}`;
                  if (title === 'length') return `${val.toFixed(0)}m`;
                  if (val >= 1000000) return `${(val/1000000).toFixed(1)}M`;
                  if (val >= 1000) return `${(val/1000).toFixed(0)}k`;
                  return val.toFixed(0);
                }}
              >
                <Label value={xLabel.charAt(0).toUpperCase() + xLabel.slice(1)} offset={-15} position="insideBottom" style={{ fontSize: '10px', fill: '#888', fontWeight: 'bold' }} />
              </XAxis>
              <YAxis 
                type="number" 
                dataKey="y" 
                name="Rating" 
                domain={[0, 10]} 
                stroke="#666" 
                fontSize={10} 
                ticks={[0, 5, 10]}
              >
                <Label value="Rating" angle={-90} position="insideLeft" style={{ fontSize: '10px', fill: '#888', fontWeight: 'bold', textAnchor: 'middle' }} />
              </YAxis>
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px', fontSize: '10px' }}
              />
              <Scatter 
                name="Ratings" 
                data={sortedData} 
                fill="#3b82f6" 
                animationDuration={500}
                animationBegin={0}
                animationEasing="ease-out"
              />
              {showTrendline && trendlineData.length > 0 && (
                <Scatter 
                  data={trendlineData} 
                  line={{ stroke: '#ffffff', strokeWidth: 1.5, opacity: 0.4 }} 
                  shape={() => null} 
                  tooltipType="none"
                />
              )}
            </ScatterChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ExplainabilityChart;
