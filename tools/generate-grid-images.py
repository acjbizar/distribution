#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


def write_png(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def compute_radius_and_offset(w: int, h: int, cols: int, rows: int) -> Tuple[float, float, float]:
    """
    Circles touch horizontally/vertically:
      centers spaced by 2R
      content size = (2*cols*R, 2*rows*R)

    Choose R to fit inside (w,h) and center the grid.
    Returns (R, x0, y0) where first center is at (x0+R, y0+R).
    """
    if cols <= 0 or rows <= 0:
        raise ValueError("cols/rows must be >= 1")

    r_w = w / (2.0 * cols)
    r_h = h / (2.0 * rows)
    r = min(r_w, r_h)

    content_w = 2.0 * cols * r
    content_h = 2.0 * rows * r

    x0 = (w - content_w) / 2.0
    y0 = (h - content_h) / 2.0
    return r, x0, y0


def hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    s = hexstr.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        raise ValueError(f"Invalid color: {hexstr}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def get_lanczos_resample(Image) -> int:
    if hasattr(Image, "Resampling") and hasattr(Image.Resampling, "LANCZOS"):
        return Image.Resampling.LANCZOS
    if hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.ANTIALIAS  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a transparent PNG with a touching-circle grid: dist/images/grid-{w}x{h}.png"
    )
    ap.add_argument("--w", type=int, default=1080, help="Output width (default: 1080)")
    ap.add_argument("--h", type=int, default=1080, help="Output height (default: 1080)")
    ap.add_argument("--cols", type=int, default=6, help="Number of circle columns (default: 6)")
    ap.add_argument("--rows", type=int, default=6, help="Number of circle rows (default: 6)")

    # ✅ transparent bg by default, semi-transparent gray stroke by default
    ap.add_argument("--stroke", default="#9e9e9e", help="Circle stroke color (default: #9e9e9e)")
    ap.add_argument("--stroke-width", type=float, default=2.0, help="Circle stroke width in px (default: 2.0)")
    ap.add_argument("--opacity", type=float, default=0.35, help="Stroke opacity 0..1 (default: 0.35)")

    ap.add_argument("--aa", type=int, default=4, help="Supersampling factor for antialiasing (default: 4)")
    ap.add_argument("--out-dir", default="dist/images", help="Output directory (default: dist/images)")
    args = ap.parse_args()

    if args.w <= 0 or args.h <= 0:
        raise SystemExit("w and h must be positive")
    if args.aa < 1:
        raise SystemExit("aa must be >= 1")
    if not (0.0 <= args.opacity <= 1.0):
        raise SystemExit("opacity must be between 0 and 1")

    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: Pillow.\n"
            "Install with:\n"
            "  pip install pillow\n"
        ) from e

    resample = get_lanczos_resample(Image)

    # Supersample for cleaner lines, then downscale.
    W = args.w * args.aa
    H = args.h * args.aa
    stroke_w = max(1, int(round(args.stroke_width * args.aa)))

    # ✅ transparent background
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r, x0, y0 = compute_radius_and_offset(W, H, args.cols, args.rows)
    dx = 2.0 * r
    dy = 2.0 * r

    sr, sg, sb = hex_to_rgb(args.stroke)
    sa = int(round(args.opacity * 255))
    stroke_rgba = (sr, sg, sb, sa)

    for row in range(args.rows):
        cy = y0 + r + row * dy
        for col in range(args.cols):
            cx = x0 + r + col * dx
            bbox = (cx - r, cy - r, cx + r, cy + r)
            draw.ellipse(bbox, outline=stroke_rgba, width=stroke_w)

    if args.aa != 1:
        img = img.resize((args.w, args.h), resample=resample)

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"grid-{args.w}x{args.h}.png"
    write_png(out_path, img)
    print(f"✓ Wrote {out_path.as_posix()}")


if __name__ == "__main__":
    main()
