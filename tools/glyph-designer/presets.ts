
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
 * Character construction rules:
 * - Columns: 0, 1, 2
 * - Main Height: Rows 1, 2, 3
 * - Arcs: 0:TR, 1:BR, 2:BL, 3:TL
 * - vSegments side: 0:Left, 1:Right
 * - hSegments side: 0:Top, 1:Bottom
 */
export const CHARACTER_PRESETS: Preset[] = [
  {
    name: 'O',
    data: {
      vSegments: ['0-1-0', '0-2-0', '2-1-1', '2-2-1'],
      hSegments: ['0-1-0', '1-1-0', '0-3-1', '1-3-1'],
      arcs: ['0-1-3', '2-1-0', '2-3-1', '0-3-2']
    }
  },
  {
    name: 'H',
    data: {
      vSegments: ['0-1-0', '0-2-0', '2-1-1', '2-2-1'],
      hSegments: ['0-2-1', '1-2-1']
    }
  },
  {
    name: 'A',
    data: {
      vSegments: ['0-1-0', '0-2-0', '2-1-1', '2-2-1'],
      hSegments: ['0-1-0', '1-1-0', '0-2-1', '1-2-1'],
      arcs: ['0-1-3', '2-1-0']
    }
  },
  {
    name: 'U',
    data: {
      vSegments: ['0-1-0', '0-2-0', '2-1-1', '2-2-1'],
      hSegments: ['0-3-1', '1-3-1'],
      arcs: ['0-3-2', '2-3-1']
    }
  },
  {
    name: 'E',
    data: {
      vSegments: ['0-1-0', '0-2-0'],
      hSegments: ['0-1-0', '1-1-0', '0-2-1', '1-2-1', '0-3-1', '1-3-1']
    }
  },
  {
    name: 'L',
    data: {
      vSegments: ['0-1-0', '0-2-0'],
      hSegments: ['0-3-1', '1-3-1']
    }
  },
  {
    name: 'C',
    data: {
      vSegments: ['0-1-0', '0-2-0'],
      hSegments: ['0-1-0', '1-1-0', '0-3-1', '1-3-1'],
      arcs: ['0-1-3', '0-3-2']
    }
  },
  {
    name: 'I',
    data: {
      vSegments: ['1-1-0', '1-2-0'],
      dots: ['1-1', '1-3']
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
