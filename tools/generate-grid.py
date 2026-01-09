#!/usr/bin/env python3
"""
flower_grid.py — generate a "flower of life" style circle grid as SVG.

Usage:
  python flower_grid.py --width 2000 --height 1200 --radius 36 --out grid.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def make_svg(
    width: float,
    height: float,
    r: float,
    stroke: str,
    stroke_width: float,
    opacity: float,
    background: str,
) -> str:
    dx = r
    dy = r * math.sqrt(3) / 2.0

    # Start a bit outside the canvas so the crop looks like your screenshot edges
    y = -r
    row = 0

    centers: list[tuple[float, float]] = []
    while y <= height + r:
        x_offset = (row % 2) * (dx / 2.0)
        x = -r + x_offset
        while x <= width + r:
            centers.append((x, y))
            x += dx
        y += dy
        row += 1

    def fnum(v: float) -> str:
        # compact but stable formatting
        return f"{v:.3f}".rstrip("0").rstrip(".")

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{fnum(width)}" height="{fnum(height)}" '
        f'viewBox="0 0 {fnum(width)} {fnum(height)}">',
        f'  <rect width="100%" height="100%" fill="{background}"/>',
        f'  <g fill="none" stroke="{stroke}" stroke-width="{fnum(stroke_width)}" opacity="{fnum(opacity)}">',
        "    <defs>",
        f'      <circle id="c" cx="0" cy="0" r="{fnum(r)}" />',
        "    </defs>",
    ]

    # Use <use> to keep SVG smaller than emitting thousands of <circle> elements
    for cx, cy in centers:
        svg_parts.append(
            f'    <use xlink:href="#c" href="#c" transform="translate({fnum(cx)} {fnum(cy)})" />'
        )

    svg_parts += [
        "  </g>",
        "</svg>",
        "",
    ]
    return "\n".join(svg_parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1200.0, help="SVG width in px")
    ap.add_argument("--height", type=float, default=700.0, help="SVG height in px")
    ap.add_argument("--radius", type=float, default=36.0, help="Circle radius in px")
    ap.add_argument("--stroke", type=str, default="#9aa0a6", help="Stroke color (CSS)")
    ap.add_argument("--stroke-width", type=float, default=1.0, help="Stroke width in px")
    ap.add_argument("--opacity", type=float, default=0.45, help="Stroke opacity 0..1")
    ap.add_argument("--background", type=str, default="#ffffff", help="Background color")
    ap.add_argument("--out", type=Path, default=Path("grid.svg"), help="Output SVG path")
    args = ap.parse_args()

    svg = make_svg(
        width=args.width,
        height=args.height,
        r=args.radius,
        stroke=args.stroke,
        stroke_width=args.stroke_width,
        opacity=args.opacity,
        background=args.background,
    )
    args.out.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.out}  ({args.width}×{args.height}, r={args.radius})")


if __name__ == "__main__":
    main()
