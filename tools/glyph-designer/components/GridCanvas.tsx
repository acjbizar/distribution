
import React, { useState } from 'react';
import { 
  R, DX, DY, X0, Y0, 
  GRID_COLS, GRID_ROWS, 
  VIEW_W, VIEW_H, STROKE 
} from '../constants';
import { GridState, ToolType } from '../types';
import { getCircleCenter } from '../utils';

interface GridCanvasProps {
  state: GridState;
  tool: ToolType;
  onChange: (newState: GridState) => void;
}

const GridCanvas: React.FC<GridCanvasProps> = ({ state, tool, onChange }) => {
  const [hovered, setHovered] = useState<string | null>(null);

  const toggleItem = (type: ToolType, key: string) => {
    const newState = { ...state };
    const targetSet = 
      type === ToolType.Circle ? newState.circles :
      type === ToolType.Arc ? newState.arcs :
      type === ToolType.Segment && key.startsWith('v') ? newState.vSegments :
      type === ToolType.Segment && key.startsWith('h') ? newState.hSegments :
      newState.dots;

    const actualKey = key.includes('-') && (key.startsWith('v') || key.startsWith('h')) ? key.split('-').slice(1).join('-') : key;

    if (targetSet.has(actualKey)) {
      targetSet.delete(actualKey);
    } else {
      targetSet.add(actualKey);
    }
    onChange(newState);
  };

  const renderCircles = () => {
    const circles = [];
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        const key = `${c}-${r}`;
        const isActive = state.circles.has(key);
        const isHovered = hovered === `circle-${key}` && tool === ToolType.Circle;

        circles.push(
          <circle
            key={`grid-c-${key}`}
            cx={cx}
            cy={cy}
            r={R}
            fill="none"
            stroke={isActive ? "black" : "#e5e7eb"}
            strokeWidth={isActive ? STROKE : 1}
            className="transition-all duration-200 cursor-pointer"
            onMouseEnter={() => setHovered(`circle-${key}`)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => toggleItem(ToolType.Circle, key)}
            opacity={isHovered && !isActive ? 0.3 : 1}
          />
        );

        // Center point for tool selection
        circles.push(
          <circle
            key={`center-${key}`}
            cx={cx}
            cy={cy}
            r={6}
            fill={isActive ? "black" : "transparent"}
            className="cursor-pointer"
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
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        
        // 4 quadrants: 0:TR, 1:BR, 2:BL, 3:TL
        // Paths: 
        // TR: M(cx+r, cy) A(r, r, 0, 0, 0, cx, cy-r) -> sweep 0
        const paths = [
          `M ${cx + R} ${cy} A ${R} ${R} 0 0 0 ${cx} ${cy - R}`, // TR
          `M ${cx} ${cy + R} A ${R} ${R} 0 0 0 ${cx + R} ${cy}`, // BR
          `M ${cx - R} ${cy} A ${R} ${R} 0 0 0 ${cx} ${cy + R}`, // BL
          `M ${cx} ${cy - R} A ${R} ${R} 0 0 0 ${cx - R} ${cy}`, // TL
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
              className="transition-all duration-200 cursor-pointer"
              onMouseEnter={() => setHovered(`arc-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Arc, key)}
              pointerEvents="stroke"
            />
          );

          // Invisible wider path for better clicking
          arcs.push(
             <path
              key={`arc-hit-${key}`}
              d={d}
              fill="none"
              stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
              strokeWidth={STROKE * 2}
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
    for (let r = 0; r < GRID_ROWS - 1; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        const [_, cyNext] = getCircleCenter(c, r + 1);

        // Two sides: Left (0) and Right (1)
        [0, 1].forEach(s => {
          const xOffset = s === 0 ? -R : R;
          const key = `${c}-${r}-${s}`;
          const isActive = state.vSegments.has(key);
          const isHovered = hovered === `vseg-${key}` && tool === ToolType.Segment;

          segments.push(
            <line
              key={`vseg-${key}`}
              x1={cx + xOffset}
              y1={cy}
              x2={cx + xOffset}
              y2={cyNext}
              stroke={isActive ? "black" : "transparent"}
              strokeWidth={STROKE}
              strokeLinecap="butt"
              className="transition-all duration-200 cursor-pointer"
              onMouseEnter={() => setHovered(`vseg-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Segment, `v-${key}`)}
              pointerEvents="stroke"
            />
          );

          // Hit area
          segments.push(
            <line
              key={`vseg-hit-${key}`}
              x1={cx + xOffset}
              y1={cy}
              x2={cx + xOffset}
              y2={cyNext}
              stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
              strokeWidth={STROKE * 2}
              className="cursor-pointer"
              onMouseEnter={() => setHovered(`vseg-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Segment, `v-${key}`)}
            />
          );
        });
      }
    }

    // Horizontal segments
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS - 1; c++) {
        const [cx, cy] = getCircleCenter(c, r);
        const [cxNext, _] = getCircleCenter(c + 1, r);

        // Two sides: Top (0) and Bottom (1)
        [0, 1].forEach(s => {
          const yOffset = s === 0 ? -R : R;
          const key = `${c}-${r}-${s}`;
          const isActive = state.hSegments.has(key);
          const isHovered = hovered === `hseg-${key}` && tool === ToolType.Segment;

          segments.push(
            <line
              key={`hseg-${key}`}
              x1={cx}
              y1={cy + yOffset}
              x2={cxNext}
              y2={cy + yOffset}
              stroke={isActive ? "black" : "transparent"}
              strokeWidth={STROKE}
              strokeLinecap="butt"
              className="transition-all duration-200 cursor-pointer"
              onMouseEnter={() => setHovered(`hseg-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Segment, `h-${key}`)}
              pointerEvents="stroke"
            />
          );

           // Hit area
           segments.push(
            <line
              key={`hseg-hit-${key}`}
              x1={cx}
              y1={cy + yOffset}
              x2={cxNext}
              y2={cy + yOffset}
              stroke={isHovered ? "rgba(59, 130, 246, 0.2)" : "transparent"}
              strokeWidth={STROKE * 2}
              className="cursor-pointer"
              onMouseEnter={() => setHovered(`hseg-${key}`)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => toggleItem(ToolType.Segment, `h-${key}`)}
            />
          );
        });
      }
    }

    return segments;
  };

  const renderDots = () => {
     const dots = [];
     for (let r = 0; r < GRID_ROWS; r++) {
       for (let c = 0; c < GRID_COLS; c++) {
         const [cx, cy] = getCircleCenter(c, r);
         const key = `${c}-${r}`;
         const isActive = state.dots.has(key);
         const isHovered = hovered === `dot-${key}` && tool === ToolType.Dot;

         dots.push(
           <circle
             key={`dot-p-${key}`}
             cx={cx}
             cy={cy}
             r={isActive ? 4 : 2}
             fill={isActive ? "black" : "#d1d5db"}
             stroke={isActive ? "black" : "transparent"}
             strokeWidth={isActive ? STROKE : 0}
             className="transition-all duration-200 cursor-pointer"
             onMouseEnter={() => setHovered(`dot-${key}`)}
             onMouseLeave={() => setHovered(null)}
             onClick={() => toggleItem(ToolType.Dot, key)}
             opacity={isHovered && !isActive ? 0.5 : 1}
           />
         );
       }
     }
     return dots;
  };

  return (
    <div className="relative bg-white shadow-xl rounded-lg p-8 border border-gray-200 flex items-center justify-center">
      <svg 
        width={VIEW_W} 
        height={VIEW_H} 
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="select-none"
      >
        <rect width="100%" height="100%" fill="white" />
        
        {/* Helper grid lines */}
        <g stroke="#f3f4f6" strokeWidth="1">
           {Array.from({length: GRID_ROWS}).map((_, i) => (
             <line key={`gl-r-${i}`} x1={0} y1={Y0 + i*DY} x2={VIEW_W} y2={Y0 + i*DY} />
           ))}
           {Array.from({length: GRID_COLS}).map((_, i) => (
             <line key={`gl-c-${i}`} x1={X0 + i*DX} y1={0} x2={X0 + i*DX} y2={VIEW_H} />
           ))}
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
