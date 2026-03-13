#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-instagram-sheet.py

Generate 1080×1080 PNG “sheets” showing all glyphs in a grid:

1) sheet.png
   - renders the SVG designs from src/character-uXXXX.svg (as before)

2) sheet-wgth100.png / sheet-wgth900.png
   - renders using the built master fonts:
       build/fonts/{basename}-master-100.ttf
       build/fonts/{basename}-master-900.ttf

Deps:
  pip install pillow cairosvg

Usage:
  python tools/generate-instagram-sheet.py
  python tools/generate-instagram-sheet.py --basename distribution --src-dir src --build-dir build/fonts
"""

from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    import cairosvg
except Exception as e:
    raise SystemExit(
        "Missing dependency: cairosvg\n"
        "Install with: pip install cairosvg\n"
        f"Original error: {e}"
    )

SVG_FILE_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")


def list_glyph_svgs(src_dir: Path) -> List[Path]:
    files = [p for p in src_dir.iterdir() if p.is_file() and SVG_FILE_RE.match(p.name)]
    files.sort(key=lambda p: int(SVG_FILE_RE.match(p.name).group(1), 16))  # by codepoint
    return files


def svg_codepoint(svg_path: Path) -> int:
    m = SVG_FILE_RE.match(svg_path.name)
    if not m:
        raise ValueError(svg_path.name)
    return int(m.group(1), 16)


def choose_grid(n: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def paste_center(dst: Image.Image, src: Image.Image, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    sw, sh = src.size
    ox = x0 + (bw - sw) // 2
    oy = y0 + (bh - sh) // 2
    dst.alpha_composite(src, (ox, oy))


def render_svg_glyph(svg_path: Path, target_w: int, target_h: int) -> Image.Image:
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=target_w,
        output_height=target_h,
        background_color="white",
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def pick_font_size_to_fit(font_path: Path, ch: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    # Binary search a font size that fits inside max_w/max_h.
    # Use a representative bbox for the character.
    lo, hi = 4, 1024
    best: Optional[ImageFont.FreeTypeFont] = None

    # Use a dummy draw context to measure
    tmp = Image.new("L", (10, 10), 0)
    d = ImageDraw.Draw(tmp)

    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(str(font_path), size=mid)
        bbox = d.textbbox((0, 0), ch, font=f)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_w and h <= max_h:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return ImageFont.truetype(str(font_path), size=max(4, min(max_w, max_h)))
    return best


def draw_font_glyph(canvas: Image.Image, box: Tuple[int, int, int, int], font_path: Path, ch: str) -> None:
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0

    # Pick size that fits (leave a little margin inside the box)
    margin = max(4, min(bw, bh) // 12)
    max_w = max(8, bw - 2 * margin)
    max_h = max(8, bh - 2 * margin)

    font = pick_font_size_to_fit(font_path, ch, max_w, max_h)

    d = ImageDraw.Draw(canvas)
    bbox = d.textbbox((0, 0), ch, font=font)
    gw = bbox[2] - bbox[0]
    gh = bbox[3] - bbox[1]

    # Center, correcting for bbox origin offsets
    cx = x0 + bw // 2
    cy = y0 + bh // 2
    tx = cx - gw // 2 - bbox[0]
    ty = cy - gh // 2 - bbox[1]

    d.text((tx, ty), ch, font=font, fill=(0, 0, 0, 255))


def build_sheet_from_svgs(
    svgs: List[Path],
    out_path: Path,
    size: int,
    padding: int,
    gap: int,
    draw_grid: bool,
) -> None:
    n = len(svgs)
    cols, rows = choose_grid(n)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    inner_w = size - 2 * padding
    inner_h = size - 2 * padding
    total_gap_x = gap * (cols - 1)
    total_gap_y = gap * (rows - 1)
    cell_w = (inner_w - total_gap_x) // cols
    cell_h = (inner_h - total_gap_y) // rows

    inset = max(6, min(cell_w, cell_h) // 10)
    target_w = max(32, cell_w - 2 * inset)
    target_h = max(32, cell_h - 2 * inset)

    for idx, svg_path in enumerate(svgs):
        r = idx // cols
        c = idx % cols

        x0 = padding + c * (cell_w + gap)
        y0 = padding + r * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        if draw_grid:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 25), width=1)

        glyph_im = render_svg_glyph(svg_path, target_w, target_h)
        box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
        paste_center(canvas, glyph_im, box)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, "PNG")
    print(f"✓ Wrote {out_path} ({size}×{size})")


def build_sheet_from_font(
    svgs: List[Path],
    out_path: Path,
    size: int,
    padding: int,
    gap: int,
    draw_grid: bool,
    font_path: Path,
) -> None:
    if not font_path.exists():
        raise SystemExit(f"Font not found: {font_path}")

    cps = [svg_codepoint(p) for p in svgs]
    chars = [chr(cp) for cp in cps]

    n = len(chars)
    cols, rows = choose_grid(n)

    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    inner_w = size - 2 * padding
    inner_h = size - 2 * padding
    total_gap_x = gap * (cols - 1)
    total_gap_y = gap * (rows - 1)
    cell_w = (inner_w - total_gap_x) // cols
    cell_h = (inner_h - total_gap_y) // rows

    inset = max(6, min(cell_w, cell_h) // 12)

    for idx, ch in enumerate(chars):
        r = idx // cols
        c = idx % cols

        x0 = padding + c * (cell_w + gap)
        y0 = padding + r * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        if draw_grid:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 25), width=1)

        box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
        draw_font_glyph(canvas, box, font_path, ch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, "PNG")
    print(f"✓ Wrote {out_path} ({size}×{size}) using {font_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="src", help="Folder containing character-uXXXX.svg")
    ap.add_argument("--build-dir", default="build/fonts", help="Where master TTFs are built")
    ap.add_argument("--basename", default="distribution", help="Font basename (default: distribution)")
    ap.add_argument("--out-dir", default="dist/images/instagram", help="Output folder")
    ap.add_argument("--size", type=int, default=1080, help="Canvas size (square)")
    ap.add_argument("--padding", type=int, default=36, help="Outer padding")
    ap.add_argument("--gap", type=int, default=18, help="Gap between cells")
    ap.add_argument("--draw-grid", action="store_true", help="Draw faint grid lines (debug)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    build_dir = Path(args.build_dir)
    out_dir = Path(args.out_dir)

    svgs = list_glyph_svgs(src_dir)
    if not svgs:
        raise SystemExit(f"No glyph SVGs found in {src_dir} (expected character-uXXXX.svg).")

    # 1) SVG design sheet
    build_sheet_from_svgs(
        svgs=svgs,
        out_path=out_dir / "sheet.png",
        size=args.size,
        padding=args.padding,
        gap=args.gap,
        draw_grid=args.draw_grid,
    )

    # 2) Weight sheets from master fonts
    font_100 = build_dir / f"{args.basename}-master-100.ttf"
    font_900 = build_dir / f"{args.basename}-master-900.ttf"

    build_sheet_from_font(
        svgs=svgs,
        out_path=out_dir / "sheet-wgth100.png",
        size=args.size,
        padding=args.padding,
        gap=args.gap,
        draw_grid=args.draw_grid,
        font_path=font_100,
    )

    build_sheet_from_font(
        svgs=svgs,
        out_path=out_dir / "sheet-wgth900.png",
        size=args.size,
        padding=args.padding,
        gap=args.gap,
        draw_grid=args.draw_grid,
        font_path=font_900,
    )


if __name__ == "__main__":
    main()