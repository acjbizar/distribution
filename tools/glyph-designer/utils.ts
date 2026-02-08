
import { GridState } from './types';
import { R, DX, DY, X0, Y0, GRID_COLS, GRID_ROWS } from './constants';

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
  state.circles.forEach(key => {
    const [c, r] = key.split('-').map(Number);
    lines.push(`    # Full circle at ${c},${r}`);
    lines.push(`    paths.append(circle_full(grid.circle(${c}, ${r})))`);
  });

  // Vertical Segments (Tangents)
  state.vSegments.forEach(key => {
    const [c, r, s] = key.split('-').map(Number);
    const sideName = s === 0 ? 'L' : 'R';
    lines.push(`    # Vertical segment col ${c}, row ${r}, side ${sideName}`);
    lines.push(`    paths.append(SvgPath().M(grid.circle(${c}, ${r}).${sideName}()).L(grid.circle(${c}, ${r+1}).${sideName}()).d())`);
  });

  // Horizontal Segments (Tangents)
  state.hSegments.forEach(key => {
    const [c, r, s] = key.split('-').map(Number);
    const sideName = s === 0 ? 'T' : 'B';
    lines.push(`    # Horizontal segment col ${c}, row ${r}, side ${sideName}`);
    lines.push(`    paths.append(SvgPath().M(grid.circle(${c}, ${r}).${sideName}()).L(grid.circle(${c+1}, ${r}).${sideName}()).d())`);
  });

  // Arcs (Quadrants)
  // Mapping quadrants: 0:TR, 1:BR, 2:BL, 3:TL
  state.arcs.forEach(key => {
    const [c, r, q] = key.split('-').map(Number);
    const cStr = `grid.circle(${c}, ${r})`;
    let move = '';
    let end = '';
    let sweep = 0; // standard for most of our glyph logic if opening up

    switch(q) {
      case 0: move = 'R()'; end = 'T()'; break; // TR
      case 1: move = 'B()'; end = 'R()'; break; // BR
      case 2: move = 'L()'; end = 'B()'; break; // BL
      case 3: move = 'T()'; end = 'L()'; break; // TL
    }
    
    lines.push(`    # Arc at ${c},${r} quadrant ${q}`);
    lines.push(`    paths.append(SvgPath().M(${cStr}.${move}).A(${cStr}.r, 0, 0, ${cStr}.${end}).d())`);
  });

  // Dots
  state.dots.forEach(key => {
    const [c, r, pos] = key.split('-').map(Number);
    // In our simplified interaction, dots are at tangency points or centers
    // For now, let's just use the center as a placeholder for dot logic
    lines.push(`    # Dot at ${c},${r}`);
    lines.push(`    dots.append((grid.circle(${c}, ${r}).cx, grid.circle(${c}, ${r}).cy, DOT_R))`);
  });

  lines.push(``);
  lines.push(`    return paths, anchors, dots`);
  
  return lines.join('\n');
};
