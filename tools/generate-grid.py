#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_grid_svg_cropped_diameters.py

Renders a Flower-of-Life / hex-packed circle grid (circle radius R),
but crops the output to an exact rectangle measuring:

  width  = 2 circles wide  = 2 * (2R) = 4R
  height = 3 circles high  = 3 * (2R) = 6R

It still renders extra circles around the crop so the overlap "petals"
inside the crop are fully formed.

Output: src/grid.svg
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
        # odd rows offset by dx/2
        x = col * self.dx + (self.dx / 2.0 if (row & 1) else 0.0)
        y = row * self.dy
        return (x, y)


def build_svg(
    grid: HexCircleGrid,
    crop_w: float,
    crop_h: float,
    # where the crop sits in the infinite grid
    crop_origin_x: float,
    crop_origin_y: float,
    # extra circles to draw around the crop so petals form
    pad_cols: int,
    pad_rows: int,
    stroke: str = "#777",
    stroke_width: float = 1.0,
    stroke_opacity: float = 0.35,
    show_border: bool = False,
) -> str:
    # Determine which grid centers to render: choose a neighborhood that definitely covers the crop + petals.
    # We'll compute a conservative col/row range by projecting crop bounds into grid coords.

    # Expand the crop region by 2R to ensure petal contributors are included
    expand = 2.0 * grid.R
    min_x = crop_origin_x - expand
    max_x = crop_origin_x + crop_w + expand
    min_y = crop_origin_y - expand
    max_y = crop_origin_y + crop_h + expand

    # Convert bounds to rough row range (using dy)
    r0 = int(math.floor(min_y / grid.dy)) - pad_rows - 2
    r1 = int(math.ceil(max_y / grid.dy)) + pad_rows + 2

    # For columns, use dx but remember odd-row offset; just overshoot a bit
    c0 = int(math.floor(min_x / grid.dx)) - pad_cols - 4
    c1 = int(math.ceil(max_x / grid.dx)) + pad_cols + 4

    centers: List[Tuple[float, float]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            centers.append(grid.center(c, r))

    # Translate so crop origin becomes (0,0)
    tx = -crop_origin_x
    ty = -crop_origin_y

    def f(v: float) -> str:
        return f"{v:.3f}".rstrip("0").rstrip(".")

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{f(crop_w)}" height="{f(crop_h)}" '
        f'viewBox="0 0 {f(crop_w)} {f(crop_h)}">'
    )

    if show_border:
        parts.append(
            f'  <rect x="0" y="0" width="{f(crop_w)}" height="{f(crop_h)}" '
            f'fill="none" stroke="#00f" stroke-width="1"/>'
        )

    # Draw all circles; viewBox does the cropping
    parts.append(
        f'  <g fill="none" stroke="{stroke}" stroke-width="{f(stroke_width)}" opacity="{f(stroke_opacity)}">'
    )
    for (cx, cy) in centers:
        parts.append(f'    <circle cx="{f(cx + tx)}" cy="{f(cy + ty)}" r="{f(grid.R)}" />')
    parts.append("  </g>")

    parts.append("</svg>")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    # --- CONFIG ---
    R = 30.0

    # Crop is EXACTLY 2 diameters wide by 3 diameters high:
    crop_w = 4.0 * R
    crop_h = 6.0 * R

    # Where to place that crop in the infinite grid.
    # Choosing origin (0,0) means the crop starts at the center of the top-left circle
    # minus nothing — i.e. you're cropping "through" the lattice rather than centering
    # on a circle. If you'd rather center a circle in the crop, shift by R.
    crop_origin_x = -R
    crop_origin_y = -R

    # Extra neighborhood render padding (usually enough for petals)
    PAD_COLS = 2
    PAD_ROWS = 3

    out_path = Path("src/grid.svg")
    grid = HexCircleGrid(R=R)

    svg = build_svg(
        grid=grid,
        crop_w=crop_w,
        crop_h=crop_h,
        crop_origin_x=crop_origin_x,
        crop_origin_y=crop_origin_y,
        pad_cols=PAD_COLS,
        pad_rows=PAD_ROWS,
        stroke="#777",
        stroke_width=1.0,
        stroke_opacity=0.35,
        show_border=True,
    )

    write_text_lf(out_path, svg)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
