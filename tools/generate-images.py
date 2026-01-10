#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# -----------------------------
# Character set (must match glyph generator)
# -----------------------------
SHEET_ROWS = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]
REQUESTED = "".join(SHEET_ROWS)


def uniq(s: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ch in s:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render SVG glyphs (character-uXXXX.svg) to PNGs (character-uXXXX.png)."
    )
    ap.add_argument("--svg-dir", default="src", help="Input SVG directory (default: src)")
    ap.add_argument("--out-dir", default="dist/images", help="Output PNG directory (default: dist/images)")
    ap.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Scale factor for output PNGs (default: 4.0). 1.0 keeps SVG pixel size.",
    )
    ap.add_argument(
        "--transparent",
        action="store_true",
        help="Keep transparency (do not force white background).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written, but don't write files")
    args = ap.parse_args()

    try:
        import cairosvg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: cairosvg.\n"
            "Install with:\n"
            "  pip install cairosvg\n"
            "Note: On some systems you may also need Cairo/Pango system packages.\n"
        ) from e

    svg_dir = Path(args.svg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    written = 0

    for ch in uniq(REQUESTED):
        cp = ord(ch)
        cp_hex = f"{cp:04x}"

        in_svg = svg_dir / f"character-u{cp_hex}.svg"
        out_png = out_dir / f"character-u{cp_hex}.png"

        if not in_svg.exists():
            missing.append(f"{ch} (expected {in_svg.as_posix()})")
            continue

        if args.dry_run:
            print(f"[DRY] {ch} U+{cp:04X} -> {out_png.as_posix()}")
            continue

        svg_bytes = in_svg.read_bytes()

        # Background:
        # - your SVG generator already includes a white <rect ... fill="#fff"/>.
        # - if you want *true* transparency, you can remove that rect in the SVG generator
        #   or pass --transparent and we *attempt* to neutralize it by rendering with no background,
        #   but if the rect is present it will still be white.
        # So: we keep default behavior (white), and --transparent only helps if you remove the rect.
        background_color = None if args.transparent else "white"

        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(out_png),
            scale=args.scale,
            background_color=background_color,
        )

        written += 1
        print(f"✓ {ch} U+{cp:04X} -> {out_png.as_posix()}")

    print(f"\nDone. Wrote {written} PNG(s) into {out_dir.resolve()}")
    if missing:
        print("\nMissing source SVGs for:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
