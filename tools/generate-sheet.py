#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


# -----------------------------
# Character sheet layout (must match your glyph set)
# -----------------------------
SHEET_ROWS = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]

# Your glyphs share height; width varies for m/M
DEFAULT_GLYPH_H = 320  # fallback if viewBox parsing fails


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def strip_xml_prolog(svg: str) -> str:
    lines = svg.splitlines()
    if lines and lines[0].lstrip().startswith("<?xml"):
        lines = lines[1:]
        while lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines)


SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE | re.DOTALL)
VIEWBOX_RE = re.compile(r'viewBox\s*=\s*"([^"]+)"', re.IGNORECASE)
WIDTH_RE = re.compile(r'width\s*=\s*"([^"]+)"', re.IGNORECASE)
HEIGHT_RE = re.compile(r'height\s*=\s*"([^"]+)"', re.IGNORECASE)


def _parse_number(s: str) -> float:
    # Handles "240", "240px", "240.0"
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Could not parse number from {s!r}")
    return float(m.group(0))


def read_glyph_svg(svg_path: Path) -> Tuple[str, int, int]:
    """
    Returns:
      inner_svg: content INSIDE the <svg> ... </svg> wrapper (no outer tag)
      w, h: extracted from viewBox (preferred) or width/height fallback
    """
    raw = strip_xml_prolog(svg_path.read_text(encoding="utf-8"))

    # Find the opening <svg ...>
    m_open = SVG_OPEN_RE.search(raw)
    if not m_open:
        raise ValueError(f"No <svg> root found in {svg_path}")

    # Determine width/height (prefer viewBox)
    attrs = m_open.group(1) or ""
    w = h = None

    m_vb = VIEWBOX_RE.search(attrs)
    if m_vb:
        parts = m_vb.group(1).strip().split()
        if len(parts) == 4:
            w = int(round(float(parts[2])))
            h = int(round(float(parts[3])))

    if w is None or h is None:
        m_w = WIDTH_RE.search(attrs)
        m_h = HEIGHT_RE.search(attrs)
        if m_w and m_h:
            w = int(round(_parse_number(m_w.group(1))))
            h = int(round(_parse_number(m_h.group(1))))
        else:
            # last resort fallback (your generator uses 240/320 or 320/320)
            # try to infer from filename for m/M if desired, but not necessary.
            w = 240
            h = DEFAULT_GLYPH_H

    # Extract inner content between first '>' of <svg ...> and the closing </svg>
    start = raw.find(">", m_open.start())
    end = raw.rfind("</svg>")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract inner SVG content from {svg_path}")

    inner = raw[start + 1 : end].strip()
    return inner + "\n", w, h


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate src/sheet.svg by composing generated glyph SVGs."
    )
    ap.add_argument("--svg-dir", default="src", help="Directory containing character-uXXXX.svg (default: src)")
    ap.add_argument("--out", default="src/sheet.svg", help="Output sheet svg path (default: src/sheet.svg)")
    ap.add_argument("--gap-x", type=int, default=30, help="Horizontal gap between glyphs (default: 30)")
    ap.add_argument("--gap-y", type=int, default=30, help="Vertical gap between rows (default: 30)")
    ap.add_argument("--padding", type=int, default=30, help="Outer padding around sheet (default: 30)")
    ap.add_argument("--bg", default="#fff", help="Background fill (default: #fff). Use 'none' for transparent.")
    ap.add_argument("--labels", action="store_true", help="Draw small labels under each glyph (debug)")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    out_path = Path(args.out)

    # Preload glyph inners + sizes so we can compute sheet dimensions
    rows_data: List[List[Tuple[str, str, int, int]]] = []
    # tuple: (char, inner_svg, w, h)

    missing: List[str] = []

    max_row_w = 0
    total_h = 0

    for row in SHEET_ROWS:
        row_glyphs: List[Tuple[str, str, int, int]] = []
        row_w = 0
        row_h = 0

        for idx, ch in enumerate(row):
            cp_hex = f"{ord(ch):04x}"
            glyph_path = svg_dir / f"character-u{cp_hex}.svg"
            if not glyph_path.exists():
                missing.append(f"{ch} (missing {glyph_path.as_posix()})")
                continue

            inner, w, h = read_glyph_svg(glyph_path)
            row_glyphs.append((ch, inner, w, h))

            if idx > 0 and row_glyphs:
                row_w += args.gap_x
            row_w += w
            row_h = max(row_h, h)

        rows_data.append(row_glyphs)
        max_row_w = max(max_row_w, row_w)
        total_h += row_h

    # Add row gaps
    row_count = len(rows_data)
    if row_count > 1:
        total_h += args.gap_y * (row_count - 1)

    sheet_w = args.padding * 2 + max_row_w
    sheet_h = args.padding * 2 + total_h

    bg = args.bg
    bg_rect = ""
    if bg.lower() != "none":
        bg_rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{bg}"/>\n'

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {sheet_w} {sheet_h}" width="{sheet_w}" height="{sheet_h}">'
    )
    out.append("<desc>Glyph sheet composed from character-uXXXX.svg glyphs</desc>")
    if bg_rect:
        out.append(bg_rect.rstrip())

    cur_y = args.padding

    # Simple label style (optional)
    if args.labels:
        out.append(
            '<style>'
            '.lbl{font:12px monospace; fill:#333;}'
            '</style>'
        )

    for row_glyphs in rows_data:
        # Determine row height for y increment
        row_h = 0
        for _, _, _, h in row_glyphs:
            row_h = max(row_h, h)
        cur_x = args.padding

        for i, (ch, inner, w, h) in enumerate(row_glyphs):
            out.append(f'<g transform="translate({cur_x} {cur_y})">')
            out.append(inner.rstrip())
            if args.labels:
                cp = ord(ch)
                # place label just below the glyph box, with a little margin
                out.append(f'<text class="lbl" x="0" y="{h + 16}">{ch} U+{cp:04X}</text>')
            out.append("</g>")
            cur_x += w + args.gap_x

        cur_y += row_h + args.gap_y

    out.append("</svg>")
    out.append("")

    write_text_lf(out_path, "\n".join(out))

    print(f"✓ Wrote {out_path.as_posix()} ({sheet_w}×{sheet_h})")
    if missing:
        print("\nMissing glyph SVGs:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
