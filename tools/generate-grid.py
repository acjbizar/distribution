#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_grid_svg.py

Renders a hex/Flower-of-Life style circle grid patch:
- 2 circles wide (2 columns)
- 3 circles high (3 rows)
using the standard hex packing:
  dx = R
  dy = R*sqrt(3)/2
  odd rows offset by dx/2

Outputs: src/grid.svg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


@dataclass(frozen=True)
class HexCircleGrid:
    R: float  # circle radius

    @property
    def dx(self) -> float:
        return self.R

    @property
    def dy(self) -> float:
        return self.R * math.sqrt(3) / 2.0

    def center(self, col: int, row: int) -> Tuple[float, float]:
        x = col * self.dx + (self.dx / 2.0 if (row & 1) else 0.0)
        y = row * self.dy
        return (x, y)


def build_svg(
    grid: HexCircleGrid,
    cols: int,
    rows: int,
    padding: float,
    stroke: str = "#777",
    stroke_width: float = 1.0,
    stroke_opacity: float = 0.35,
    show_centers: bool = True,
    center_radius: float = 1.8,
    show_border: bool = False,
) -> str:
    centers: List[Tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            centers.append(grid.center(c, r))

    # Compute bounding box that fully contains all circles
    xs = [x for x, _ in centers]
    ys = [y for _, y in centers]
    min_x = min(xs) - grid.R - padding
    max_x = max(xs) + grid.R + padding
    min_y = min(ys) - grid.R - padding
    max_y = max(ys) + grid.R + padding

    W = max_x - min_x
    H = max_y - min_y

    def f(v: float) -> str:
        return f"{v:.3f}".rstrip("0").rstrip(".")

    # Translate everything so min_x/min_y becomes (0,0)
    tx = -min_x
    ty = -min_y

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{f(W)}" height="{f(H)}" viewBox="0 0 {f(W)} {f(H)}">'
    )

    if show_border:
        parts.append(f'  <rect x="0" y="0" width="{f(W)}" height="{f(H)}" fill="none" stroke="#00f" stroke-width="1"/>')

    # Circle outlines
    parts.append(
        f'  <g fill="none" stroke="{stroke}" stroke-width="{f(stroke_width)}" opacity="{f(stroke_opacity)}">'
    )
    for (cx, cy) in centers:
        parts.append(f'    <circle cx="{f(cx + tx)}" cy="{f(cy + ty)}" r="{f(grid.R)}" />')
    parts.append("  </g>")

    # Center dots (optional)
    if show_centers:
        parts.append('  <g fill="#00f" opacity="0.65">')
        for (cx, cy) in centers:
            parts.append(f'    <circle cx="{f(cx + tx)}" cy="{f(cy + ty)}" r="{f(center_radius)}" />')
        parts.append("  </g>")

    parts.append("</svg>")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    # ---- CONFIG ----
    R = 30.0          # circle radius (adjust to match your grid)
    COLS = 2
    ROWS = 3
    PADDING = 8.0

    out_path = Path("src/grid.svg")

    grid = HexCircleGrid(R=R)

    svg = build_svg(
        grid=grid,
        cols=COLS,
        rows=ROWS,
        padding=PADDING,
        stroke="#777",
        stroke_width=1.0,
        stroke_opacity=0.35,
        show_centers=True,     # helps you verify the offset pattern
        center_radius=1.8,
        show_border=False,
    )

    write_text_lf(out_path, svg)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
