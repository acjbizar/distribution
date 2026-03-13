#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-instagram-sheet.py

Generate a 1080×1080 PNG “sheet” showing all glyph SVG designs in a grid.

Input:
  src/character-uXXXX.svg

Output (default):
  dist/images/instagram/sheet.png

Deps:
  pip install pillow cairosvg
  (On Windows you may also need Cairo/GTK runtimes for cairosvg; see cairosvg docs.)

Usage:
  python tools/generate-instagram-sheet.py
  python tools/generate-instagram-sheet.py --src-dir src --out dist/images/instagram/sheet.png
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

try:
    import cairosvg
except Exception as e:
    raise SystemExit(
        "Missing dependency: cairosvg\n"
        "Install with: pip install cairosvg\n"
        "If you’re on Windows, ensure Cairo is available (cairosvg docs).\n"
        f"Original error: {e}"
    )

SVG_FILE_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")


def list_glyph_svgs(src_dir: Path) -> List[Path]:
    files = [p for p in src_dir.iterdir() if p.is_file() and SVG_FILE_RE.match(p.name)]
    files.sort(key=lambda p: int(SVG_FILE_RE.match(p.name).group(1), 16))  # by codepoint
    return files


def choose_grid(n: int) -> Tuple[int, int]:
    # Near-square grid
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def svg_to_rgba_image(svg_path: Path, target_w: int, target_h: int) -> Image.Image:
    # Render SVG to PNG bytes at the target size
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=target_w,
        output_height=target_h,
        background_color="white",
    )
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return im


def paste_center(dst: Image.Image, src: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    sw, sh = src.size
    ox = x0 + (bw - sw) // 2
    oy = y0 + (bh - sh) // 2
    dst.alpha_composite(src, (ox, oy))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="src", help="Folder containing character-uXXXX.svg")
    ap.add_argument("--out", default="dist/images/instagram/sheet.png", help="Output PNG path")
    ap.add_argument("--size", type=int, default=1080, help="Canvas size (square), default 1080")
    ap.add_argument("--padding", type=int, default=36, help="Outer padding on canvas")
    ap.add_argument("--gap", type=int, default=18, help="Gap between cells")
    ap.add_argument("--draw-grid", action="store_true", help="Draw faint grid lines (debug)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_path = Path(args.out)

    svgs = list_glyph_svgs(src_dir)
    if not svgs:
        raise SystemExit(f"No glyph SVGs found in {src_dir} (expected character-uXXXX.svg).")

    n = len(svgs)
    cols, rows = choose_grid(n)

    size = int(args.size)
    pad = int(args.padding)
    gap = int(args.gap)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Compute cell rects within inner area
    inner_w = size - 2 * pad
    inner_h = size - 2 * pad

    # total gaps between cells
    total_gap_x = gap * (cols - 1)
    total_gap_y = gap * (rows - 1)

    cell_w = (inner_w - total_gap_x) // cols
    cell_h = (inner_h - total_gap_y) // rows

    # Render each SVG into an "inner box" within the cell
    # (leave some breathing room so strokes don’t kiss edges)
    inset = max(6, min(cell_w, cell_h) // 10)
    target_w = max(32, cell_w - 2 * inset)
    target_h = max(32, cell_h - 2 * inset)

    # Import io here (kept late to keep top tidy)
    global io
    import io  # noqa: E402

    for idx, svg_path in enumerate(svgs):
        r = idx // cols
        c = idx % cols

        x0 = pad + c * (cell_w + gap)
        y0 = pad + r * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        if args.draw_grid:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 25), width=1)

        # Render SVG to target size
        try:
            png_bytes = cairosvg.svg2png(
                url=str(svg_path),
                output_width=target_w,
                output_height=target_h,
                background_color="white",
            )
            glyph_im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        except Exception as e:
            raise SystemExit(f"Failed to render {svg_path.name}: {e}")

        # Center inside inner cell box (with inset)
        box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
        paste_center(canvas, glyph_im, box)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, "PNG")
    print(f"✓ Wrote {out_path} ({size}×{size}), {n} glyph(s), grid {cols}×{rows}")


if __name__ == "__main__":
    main()