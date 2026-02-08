
export type Point = [number, number];

export interface GridState {
  circles: Set<string>; // "col-row"
  vSegments: Set<string>; // "col-row-side" (side: 0 for Left, 1 for Right)
  hSegments: Set<string>; // "col-row-side" (side: 0 for Top, 1 for Bottom)
  arcs: Set<string>; // "col-row-quadrant" (0:TR, 1:BR, 2:BL, 3:TL)
  dots: Set<string>; // "col-row-pos"
}

export enum ToolType {
  Circle = 'CIRCLE',
  Arc = 'ARC',
  Segment = 'SEGMENT',
  Dot = 'DOT'
}
