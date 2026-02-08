
import { GridState } from './types';

export interface Preset {
  name: string;
  data: {
    circles?: string[];
    vSegments?: string[];
    hSegments?: string[];
    arcs?: string[];
    dots?: string[];
  };
}

/**
 * Geometric Construction Plan (Unified Coordinates):
 * 
 * - "0": Verticals at x=0.5 and x=1.5 (y=1.0-2.0). Arcs 1-1-0, 1-1-3, 1-2-1, 1-2-2.
 * - "1": Right side stem (x=1.5) from y=1.0 to 2.5. TR hook at 1-1-0.
 * - "2": Matches snippet: h-segment y=2.5 (x=1.0 to 1.5). 
 *        Arcs circle(1,1): TR(0), BR(1), TL(3). circle(1,2): BL(2), TL(3).
 * - "C": Arcs TL, TR, BR, BL. Vertical line on the LEFT side (x=0.5).
 * - "i": Matches snippet: x=0.5 verticals at y=1.5, 2.0, and 3.0. Dot at 0.5-0.5.
 * - ".": Placed half a unit higher (y=2.5).
 * - "a": Matches snippet: x=1.5 verticals at y=1.0, 1.5. 
 *        Arcs circle(1,1): TR(0), BR(1), TL(3). circle(1,2): BR(1), BL(2), TL(3).
 */
export const CHARACTER_PRESETS: Preset[] = [
  {
    name: '0',
    data: {
      vSegments: ['0.5-1.0', '0.5-1.5', '1.5-1.0', '1.5-1.5'],
      arcs: ['1-1-0', '1-1-3', '1-2-1', '1-2-2']
    }
  },
  {
    name: '1',
    data: {
      arcs: ['1-1-0'], 
      vSegments: ['1.5-1.0', '1.5-1.5', '1.5-2.0'] 
    }
  },
  {
    name: '2',
    data: {
      arcs: ['1-1-0', '1-1-1', '1-1-3', '1-2-2', '1-2-3'],
      hSegments: ['1.0-2.5']
    }
  },
  {
    name: '3',
    data: {
      arcs: ['1-1-3', '1-1-0', '1-1-1', '1-2-0', '1-2-1', '1-2-2']
    }
  },
  {
    name: '8',
    data: {
      circles: ['1-1', '1-2']
    }
  },
  {
    name: 'a',
    data: {
      vSegments: ['1.5-1.0', '1.5-1.5'],
      arcs: ['1-1-0', '1-1-1', '1-1-3', '1-2-1', '1-2-2', '1-2-3']
    }
  },
  {
    name: 'C',
    data: {
      arcs: ['1-1-3', '1-1-0', '1-2-1', '1-2-2'],
      vSegments: ['0.5-1.0', '0.5-1.5'] // Vertical bar on the left side
    }
  },
  {
    name: 'i',
    data: {
      dots: ['0.5-0.5'], 
      vSegments: ['0.5-1.5', '0.5-2.0', '0.5-3.0']
    }
  },
  {
    name: '.',
    data: {
      dots: ['0.5-2.5'] 
    }
  }
];

export const applyPreset = (preset: Preset): GridState => {
  return {
    circles: new Set(preset.data.circles || []),
    vSegments: new Set(preset.data.vSegments || []),
    hSegments: new Set(preset.data.hSegments || []),
    arcs: new Set(preset.data.arcs || []),
    dots: new Set(preset.data.dots || [])
  };
};
