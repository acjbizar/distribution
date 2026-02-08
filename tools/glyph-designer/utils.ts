
import { GridState } from './types';
import { R, DX, DY, X0, Y0 } from './constants';

export const getCircleCenter = (col: number, row: number): [number, number] => {
  return [X0 + col * DX, Y0 + row * DY];
};

export const formatPythonCode = (state: GridState): string => {
  const lines: string[] = [];
  lines.push(`def glyph_custom(grid: Grid) -> GlyphRes3:`);
  lines.push(`    paths = []`);
  lines.push(`    anchors = []`);
  lines.push(`    dots = []`);
  lines.push(``);

  // Circles
  const sortedCircles = Array.from(state.circles).sort();
  sortedCircles.forEach(key => {
    const [c, r] = key.split('-').map(Number);
    lines.push(`    paths.append(circle_full(grid.circle(${c}, ${r})))`);
  });

  // Vertical Segments (x, yStart to yStart + 0.5)
  const sortedVSegs = Array.from(state.vSegments).sort();
  sortedVSegs.forEach(key => {
    const [x, y] = key.split('-').map(Number);
    lines.push(`    paths.append(SvgPath().M(grid.X(${x}), grid.Y(${y})).L(grid.X(${x}), grid.Y(${y + 0.5})).d())`);
  });

  // Horizontal Segments (xStart to xStart + 0.5, y)
  const sortedHSegs = Array.from(state.hSegments).sort();
  sortedHSegs.forEach(key => {
    const [x, y] = key.split('-').map(Number);
    lines.push(`    paths.append(SvgPath().M(grid.X(${x}), grid.Y(${y})).L(grid.X(${x + 0.5}), grid.Y(${y})).d())`);
  });

  // Arcs (Quadrants)
  const sortedArcs = Array.from(state.arcs).sort();
  sortedArcs.forEach(key => {
    const [c, r, q] = key.split('-').map(Number);
    const cStr = `grid.circle(${c}, ${r})`;
    let move = '';
    let end = '';

    switch(q) {
      case 0: move = 'T()'; end = 'R()'; break; // TR
      case 1: move = 'R()'; end = 'B()'; break; // BR
      case 2: move = 'B()'; end = 'L()'; break; // BL
      case 3: move = 'L()'; end = 'T()'; break; // TL
    }
    
    lines.push(`    paths.append(SvgPath().M(${cStr}.${move}).A(${cStr}.r, ${cStr}.r, 0, 0, 0, ${cStr}.${end}).d())`);
  });

  // Dots
  const sortedDots = Array.from(state.dots).sort();
  sortedDots.forEach(key => {
    const [cxIdx, cyIdx] = key.split('-').map(Number);
    lines.push(`    dots.append((grid.X(${cxIdx}), grid.Y(${cyIdx}), DOT_R))`);
  });

  lines.push(``);
  lines.push(`    return paths, anchors, dots`);
  
  return lines.join('\n');
};
