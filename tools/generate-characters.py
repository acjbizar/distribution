#!/usr/bin/env python3
"""
generate_glyph_svgs_v2.py
-------------------------
Reconstructs smooth, grid-fitted glyphs based on the Flower-of-Life style grid.
Produces one SVG per glyph into ../src/.

Principles:
- Underlying grid: hexagonal lattice of circles (radius R)
- Strokes follow intersections of those circles
- Curves join smoothly using cubic Bézier segments
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple, Union, Dict, Iterable

# ============================================================
# Grid definition
# ============================================================
GRID_R = 30.0  # circle radius in the underlying hex grid
DX = GRID_R          # horizontal spacing between circle centers
DY = GRID_R * math.sqrt(3) / 2  # vertical spacing between rows

# Typical glyph box (scaled to fit inside 240×240 like before)
EM_W = 240
EM_H = 240
STROKE_W = 10

ASC  = 24
XH   = 60
BASE = 180
DESC = 228

# ============================================================
# Helper functions
# ============================================================
def point_on_circle(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    a = math.radians(deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a))

def cubic_path(x0, y0, cx1, cy1, cx2, cy2, x1, y1) -> str:
    return f"M {x0:.2f} {y0:.2f} C {cx1:.2f} {cy1:.2f}, {cx2:.2f} {cy2:.2f}, {x1:.2f} {y1:.2f}"

def arc_path(cx, cy, r, a0, a1) -> str:
    x0, y0 = point_on_circle(cx, cy, r, a0)
    x1, y1 = point_on_circle(cx, cy, r, a1)
    large = 1 if abs(a1 - a0) > 180 else 0
    sweep = 1 if a1 > a0 else 0
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {large} {sweep} {x1:.2f} {y1:.2f}"

def line_path(x0, y0, x1, y1) -> str:
    return f"M {x0:.2f} {y0:.2f} L {x1:.2f} {y1:.2f}"

def svg_doc(paths: Iterable[str], label: str) -> str:
    body = "\n".join([f'  <path d="{p}" />' for p in paths])
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {EM_W} {EM_H}" width="{EM_W}" height="{EM_H}">
  <title>{label}</title>
  <g fill="none" stroke="#000" stroke-width="{STROKE_W}" stroke-linecap="round" stroke-linejoin="round">
{body}
  </g>
</svg>
'''

def write_text_lf(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# ============================================================
# Glyph definitions (smoothed)
# ============================================================
def bowl(cx: float, cy: float, r: float, open_side="right") -> List[str]:
    if open_side == "right":
        return [arc_path(cx, cy, r, 45, 315)]
    elif open_side == "left":
        return [arc_path(cx, cy, r, 135, -135)]
    else:
        return [arc_path(cx, cy, r, 0, 360)]

def glyph_o():  # full closed loop
    return bowl(120, 120, 60, open_side=None)

def glyph_c():
    return bowl(120, 120, 60, open_side="right")

def glyph_e():
    return bowl(120, 120, 60, open_side="right") + [line_path(60, 120, 120, 120)]

def glyph_a():
    return bowl(120, 120, 60, open_side="right") + [
        line_path(180, 120, 180, 180),
        cubic_path(150, 120, 165, 115, 175, 115, 180, 120)
    ]

def glyph_b():
    return [
        line_path(60, 40, 60, 180),
        arc_path(90, 120, 30, 90, 270)
    ]

def glyph_d():
    return [
        line_path(180, 40, 180, 180),
        arc_path(150, 120, 30, 270, 90)
    ]

def glyph_n():
    return [
        line_path(60, 120, 60, 180),
        cubic_path(60, 120, 90, 90, 150, 90, 180, 120),
        line_path(180, 120, 180, 180),
    ]

def glyph_m():
    return [
        line_path(50, 120, 50, 180),
        cubic_path(50, 120, 80, 90, 120, 90, 150, 120),
        line_path(150, 120, 150, 180),
        cubic_path(150, 120, 180, 90, 220, 90, 250, 120),
        line_path(250, 120, 250, 180),
    ]

def glyph_u():
    return [
        line_path(60, 120, 60, 160),
        cubic_path(60, 160, 80, 190, 160, 190, 180, 160),
        line_path(180, 160, 180, 120),
    ]

def glyph_h():
    return [
        line_path(60, 40, 60, 180),
        cubic_path(60, 120, 90, 90, 150, 90, 180, 120),
        line_path(180, 120, 180, 180),
    ]

def glyph_y():
    return [
        cubic_path(60, 120, 100, 180, 140, 180, 180, 120),
        line_path(120, 180, 120, 220),
    ]

def glyph_x():
    return [
        cubic_path(60, 120, 90, 90, 150, 150, 180, 120),
        cubic_path(60, 180, 90, 150, 150, 90, 180, 180),
    ]

def glyph_s():
    return [
        cubic_path(60, 110, 80, 80, 160, 80, 180, 110),
        cubic_path(180, 110, 160, 140, 80, 140, 60, 170),
    ]

def glyph_0():
    return [arc_path(120, 120, 60, 0, 360)]

def glyph_1():
    return [line_path(120, 60, 120, 180)]

def glyph_2():
    return [
        cubic_path(60, 90, 100, 60, 160, 60, 180, 100),
        line_path(180, 100, 60, 180),
        line_path(60, 180, 180, 180),
    ]

def glyph_3():
    return [
        cubic_path(60, 90, 100, 60, 160, 60, 180, 100),
        cubic_path(60, 140, 100, 180, 160, 180, 180, 140),
    ]

def glyph_4():
    return [
        line_path(60, 120, 180, 120),
        line_path(180, 60, 180, 180),
    ]

def glyph_5():
    return [
        line_path(180, 60, 60, 60),
        line_path(60, 60, 60, 120),
        cubic_path(60, 120, 90, 150, 150, 150, 180, 120),
    ]

def glyph_6():
    return [
        arc_path(120, 120, 60, 45, 315),
        cubic_path(90, 120, 100, 160, 160, 160, 180, 120),
    ]

def glyph_7():
    return [line_path(60, 60, 180, 60), line_path(180, 60, 60, 180)]

def glyph_8():
    return [
        arc_path(120, 90, 30, 0, 360),
        arc_path(120, 150, 30, 0, 360),
    ]

def glyph_9():
    return [
        cubic_path(60, 120, 100, 60, 160, 60, 180, 120),
        line_path(180, 120, 180, 180),
    ]

def glyph_period():
    return [arc_path(120, 180, 5, 0, 360)]

def glyph_comma():
    return [
        arc_path(120, 180, 5, 0, 360),
        cubic_path(120, 185, 115, 200, 125, 210, 130, 215),
    ]

def glyph_question():
    return [
        cubic_path(60, 80, 100, 50, 160, 50, 180, 90),
        line_path(120, 180, 120, 190),
        arc_path(120, 205, 5, 0, 360),
    ]

# ============================================================
# Glyph map
# ============================================================
GLYPHS: Dict[str, List[str]] = {
    "a": glyph_a(),
    "b": glyph_b(),
    "c": glyph_c(),
    "d": glyph_d(),
    "e": glyph_e(),
    "h": glyph_h(),
    "m": glyph_m(),
    "n": glyph_n(),
    "o": glyph_o(),
    "s": glyph_s(),
    "u": glyph_u(),
    "x": glyph_x(),
    "y": glyph_y(),
    "0": glyph_0(),
    "1": glyph_1(),
    "2": glyph_2(),
    "3": glyph_3(),
    "4": glyph_4(),
    "5": glyph_5(),
    "6": glyph_6(),
    "7": glyph_7(),
    "8": glyph_8(),
    "9": glyph_9(),
    ".": glyph_period(),
    ",": glyph_comma(),
    "?": glyph_question(),
}

# ============================================================
# Writer
# ============================================================
def slug_for_filename(ch: str) -> str:
    special = {".": "period", ",": "comma", "?": "question"}
    return special.get(ch, ch)

def main():
    out_dir = (Path(__file__).resolve().parent / "../src").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for ch, paths in GLYPHS.items():
        svg = svg_doc(paths, f"glyph {ch}")
        fname = f"character-{slug_for_filename(ch)}.svg"
        write_text_lf(out_dir / fname, svg)
        manifest.append(fname)

    print(f"Generated {len(manifest)} glyphs in {out_dir}")

if __name__ == "__main__":
    main()
