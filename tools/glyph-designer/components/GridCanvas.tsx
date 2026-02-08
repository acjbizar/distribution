
import React, { useState } from 'react';
import { 
  R, DX, DY, X0, Y0, 
  STROKE, PADDING
} from '../constants';
import { GridState, ToolType } from '../types';
import { getCircleCenter } from '../utils';

interface GridCanvasProps {
  state: GridState;
  tool: ToolType;
  onChange: (newState: GridState) => void;
  rows: number;
  cols: number;
}

const GridCanvas: React.FC<GridCanvasProps> = ({ state, tool, onChange, rows, cols }) => {
  const [hovered, setHovered] = useState<string | null>(null);

  const viewW = (cols - 1) * DX + (R * 2) + (PADDING * 2);
  const viewH = (rows - 1) * DY + (R * 2) + (PADDING * 2);

  const toggleItem = (type: ToolType, key: string) => {
    const newState = { ...state };
    let targetSet: Set<string>;
    let targetKey: string = key;

    if (type === ToolType.Circle) targetSet = newState.circles;
    else if (type === ToolType.Arc) targetSet = newState.arcs;
    else if (type === ToolType.Segment && key.startsWith('v')) {
      targetSet = newState.vSegments;
      targetKey = key.split('-').slice(1).join('-');
    } else if (type === ToolType.Segment && key.startsWith('h')) {
      targetSet = newState.hSegments;
      targetKey = key.split('-').slice(1).join('-');
    } else targetSet = newState.dots;

    // CRITICAL: Clone the set to ensure React detects the state change
    const newSet = new Set(targetSet);
    if (newSet.has(targetKey)) {
      newSet.delete(targetKey);
    } else {
      newSet.add(targetKey);
    }

    // Assign back to the correct property
    if (type === ToolType.Circle) newState.circles = newSet;
    else if (type === ToolType.Arc) newState.arcs = newSet;
    else if (type === ToolType.Segment && key.startsWith('v')) newState.vSegments = newSet;
    else if (type === ToolType.Segment && key.startsWith('h')) newState.hSegments = newSet;
    else newState.dots = newSet;

    onChange(newState);
  };

  const renderCircles = () => {
    const circles = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        const key = `${c}-${r}`;
        const isActive = state.circles.has(key);

        circles.push(
          <circle
            key={`grid-c-${key}`}
            cx={cx}
            cy={cy}
            r={R}
            fill="none"
            stroke={isActive ? "black" : "#f1f5f9"}
            strokeWidth={isActive ? STROKE : 1}
            className="transition-all duration-200 cursor-pointer"
            onMouseEnter={() => setHovered(`circle-${key}`)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => toggleItem(ToolType.Circle, key)}
          />
        );
      }
    }
    return circles;
  };

  const renderArcs = () => {
    const arcs = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        const paths = [
          `M ${cx + R} ${cy} A ${R} ${R} 0 0 0 ${cx} ${cy - R}`, // TR (0)
          `M ${cx} ${cy + R} A ${R} ${R} 0 0 0 ${cx + R} ${cy}`, // BR (1)
          `M ${cx - R} ${cy} A ${R} ${R} 0 0 0 ${cx} ${cy + R}`, // BL (2)
          `M ${cx} ${cy - R} A ${R} ${R} 0 0 0 ${cx - R} ${cy}`, // TL (3)
        ];

        paths.forEach((d, q) => {
          const key = `${c}-${r}-${q}`;
          const isActive = state.arcs.has(key);
          const isHovered = hovered === `arc-${key}` && tool === ToolType.Arc;

          arcs.push(
            <path
              key={`arc-${key}`}
              d={d}
              fill="none"
              stroke={isActive ? "black" : "transparent"}
              strokeWidth={STROKE}
              strokeLinecap="round"
              className="transition-all duration-200 pointer-events-none"
            />
          );

          arcs.push(
             <path
              key={`arc-hit-${key}`}
              d={d}
              fill="none"
              stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
              strokeWidth={STROKE * 3}
              strokeLinecap="round"
              className="cursor-pointer"
              onMouseEnter={() => setHovered(`arc-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Arc, key)}
            />
          );
        });
      }
    }
    return arcs;
  };

  const renderSegments = () => {
    const segments = [];
    
    // Vertical segments
    for (let c = 0; c < (cols - 1) * 2 + 1; c++) {
      const x = c * 0.5;
      for (let r = 0; r < (rows - 1) * 2; r++) {
        const yStart = r * 0.5;
        const cx = X0 + x * DX;
        const cyStart = Y0 + yStart * DY;
        const cyEnd = cyStart + 0.5 * DY;

        const key = `${x.toFixed(1)}-${yStart.toFixed(1)}`;
        const isActive = state.vSegments.has(key);
        const isHovered = hovered === `vseg-${key}` && tool === ToolType.Segment;

        segments.push(
          <line
            key={`vseg-${key}`}
            x1={cx}
            y1={cyStart}
            x2={cx}
            y2={cyEnd}
            stroke={isActive ? "black" : "transparent"}
            strokeWidth={STROKE}
            strokeLinecap="round"
            className="pointer-events-none transition-all"
          />
        );

        segments.push(
          <line
            key={`vseg-hit-${key}`}
            x1={cx}
            y1={cyStart}
            x2={cx}
            y2={cyEnd}
            stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
            strokeWidth={STROKE * 3.5}
            className="cursor-pointer"
            onMouseEnter={() => setHovered(`vseg-${key}`)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => toggleItem(ToolType.Segment, `v-${key}`)}
          />
        );
      }
    }

    // Horizontal segments
    for (let r = 0; r < (rows - 1) * 2 + 1; r++) {
      const y = r * 0.5;
      for (let c = 0; c < (cols - 1) * 2; c++) {
        const xStart = c * 0.5;
        const cxStart = X0 + xStart * DX;
        const cxEnd = cxStart + 0.5 * DX;
        const cy = Y0 + y * DY;

        const key = `${xStart.toFixed(1)}-${y.toFixed(1)}`;
        const isActive = state.hSegments.has(key);
        const isHovered = hovered === `hseg-${key}` && tool === ToolType.Segment;

        segments.push(
          <line
            key={`hseg-${key}`}
            x1={cxStart}
            y1={cy}
            x2={cxEnd}
            y2={cy}
            stroke={isActive ? "black" : "transparent"}
            strokeWidth={STROKE}
            strokeLinecap="round"
            className="pointer-events-none transition-all"
          />
        );

         segments.push(
          <line
            key={`hseg-hit-${key}`}
            x1={cxStart}
            y1={cy}
            x2={cxEnd}
            y2={cy}
            stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
            strokeWidth={STROKE * 3.5}
            className="cursor-pointer"
            onMouseEnter={() => setHovered(`hseg-${key}`)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => toggleItem(ToolType.Segment, `h-${key}`)}
          />
        );
      }
    }

    return segments;
  };

  const renderDots = () => {
     const dots = [];
     for (let r = 0; r <= (rows - 1) * 2; r++) {
       for (let c = 0; c <= (cols - 1) * 2; c++) {
         const cxIdx = c / 2;
         const cyIdx = r / 2;
         const cx = X0 + cxIdx * DX;
         const cy = Y0 + cyIdx * DY;
         const key = `${cxIdx.toFixed(1)}-${cyIdx.toFixed(1)}`;
         const isActive = state.dots.has(key);
         const isHovered = hovered === `dot-${key}` && tool === ToolType.Dot;

         const isCenter = Number.isInteger(cxIdx) && Number.isInteger(cyIdx);

         dots.push(
           <circle
             key={`dot-p-${key}`}
             cx={cx}
             cy={cy}
             r={isActive ? 5 : (isCenter ? 3 : 2)}
             fill={isActive ? "black" : (isHovered ? "#3b82f6" : (isCenter ? "#cbd5e1" : "#f1f5f9"))}
             className="transition-all duration-200 cursor-pointer"
             onMouseEnter={() => setHovered(`dot-${key}`)}
             onMouseLeave={() => setHovered(null)}
             onClick={() => toggleItem(ToolType.Dot, key)}
           />
         );
       }
     }
     return dots;
  };

  return (
    <div className="relative bg-white shadow-2xl rounded-3xl p-12 border border-slate-200 flex items-center justify-center">
      <svg 
        width={viewW} 
        height={viewH} 
        viewBox={`0 0 ${viewW} ${viewH}`}
        className="select-none"
      >
        <rect width="100%" height="100%" fill="white" />
        
        <g stroke="#f8fafc" strokeWidth="1" fill="none">
          {Array.from({length: rows}).map((_, r) => 
            Array.from({length: cols}).map((_, c) => {
               const [cx, cy] = getCircleCenter(c, r);
               return <circle key={`guide-${c}-${r}`} cx={cx} cy={cy} r={R} opacity="0.3" />;
            })
          )}
        </g>

        {renderCircles()}
        {renderArcs()}
        {renderSegments()}
        {renderDots()}
      </svg>
    </div>
  );
};

export default GridCanvas;
