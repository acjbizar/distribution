
export type Point = [number, number];

export interface GridState {
  circles: Set<string>; // "col-row"
  vSegments: Set<string>; // "x-yStart" (e.g., "0.5-1.0")
  hSegments: Set<string>; // "xStart-y" (e.g., "1.0-2.5")
  arcs: Set<string>; // "col-row-quadrant" (0:TR, 1:BR, 2:BL, 3:TL)
  dots: Set<string>; // "x-y" (e.g., "0.5-0.5")
}

export enum ToolType {
  Circle = 'CIRCLE',
  Arc = 'ARC',
  Segment = 'SEGMENT',
  Dot = 'DOT'
}
