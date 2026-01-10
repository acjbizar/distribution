#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


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
CODEPOINT_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")


def _parse_number(s: str) -> float:
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

    m_open = SVG_OPEN_RE.search(raw)
    if not m_open:
        raise ValueError(f"No <svg> root found in {svg_path}")

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
            # Fallback (your generator is typically 240×320 or 320×320)
            w = 240
            h = 320

    start = raw.find(">", m_open.start())
    end = raw.rfind("</svg>")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract inner SVG content from {svg_path}")

    inner = raw[start + 1 : end].strip()
    return inner + "\n", w, h


def list_glyph_files(svg_dir: Path) -> List[Tuple[int, Path]]:
    """
    Find src/character-u{hex}.svg files and return sorted list of (codepoint_int, path).
    """
    items: List[Tuple[int, Path]] = []
    for p in svg_dir.iterdir():
        if not p.is_file():
            continue
        m = CODEPOINT_RE.match(p.name)
        if not m:
            continue
        cp = int(m.group(1), 16)
        items.append((cp, p))
    items.sort(key=lambda t: t[0])
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate src/sheet.svg by composing src/character-u{codepoint}.svg glyphs found on disk."
    )
    ap.add_argument("--svg-dir", default="src", help="Directory containing character-u*.svg (default: src)")
    ap.add_argument("--out", default="src/sheet.svg", help="Output sheet svg path (default: src/sheet.svg)")
    ap.add_argument("--cols", type=int, default=10, help="Number of columns in sheet (default: 10)")
    ap.add_argument("--gap-x", type=int, default=30, help="Horizontal gap between glyphs (default: 30)")
    ap.add_argument("--gap-y", type=int, default=30, help="Vertical gap between glyphs (default: 30)")
    ap.add_argument("--padding", type=int, default=30, help="Outer padding around sheet (default: 30)")
    ap.add_argument("--bg", default="#fff", help="Background fill (default: #fff). Use 'none' for transparent.")
    ap.add_argument("--labels", action="store_true", help="Draw small labels under each glyph (debug)")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    out_path = Path(args.out)

    glyph_files = list_glyph_files(svg_dir)
    if not glyph_files:
        raise SystemExit(f"No glyph files found in {svg_dir.resolve()} matching character-uXXXX.svg")

    # Preload glyphs
    glyphs: List[Dict[str, object]] = []
    for cp, path in glyph_files:
        inner, w, h = read_glyph_svg(path)
        glyphs.append(
            {
                "cp": cp,
                "path": path,
                "inner": inner,
                "w": w,
                "h": h,
                "ch": chr(cp) if 0 <= cp <= 0x10FFFF else "?",
            }
        )

    cols = max(1, args.cols)

    # Compute row heights (variable width/height safe)
    rows: List[List[Dict[str, object]]] = []
    for i in range(0, len(glyphs), cols):
        rows.append(glyphs[i : i + cols])

    row_heights: List[int] = []
    row_widths: List[int] = []

    for row in rows:
        rh = 0
        rw = 0
        for j, g in enumerate(row):
            rh = max(rh, int(g["h"]))  # type: ignore[arg-type]
            if j > 0:
                rw += args.gap_x
            rw += int(g["w"])  # type: ignore[arg-type]
        row_heights.append(rh)
        row_widths.append(rw)

    sheet_w = args.padding * 2 + max(row_widths)
    sheet_h = args.padding * 2 + sum(row_heights) + args.gap_y * (len(rows) - 1)

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
    out.append("<desc>Glyph sheet composed from src/character-u{codepoint}.svg files</desc>")
    if bg_rect:
        out.append(bg_rect.rstrip())

    if args.labels:
        out.append(
            "<style>"
            ".lbl{font:12px monospace; fill:#333;}"
            "</style>"
        )

    cur_y = args.padding
    for row_idx, row in enumerate(rows):
        cur_x = args.padding
        row_h = row_heights[row_idx]

        for g in row:
            cp = int(g["cp"])  # type: ignore[arg-type]
            ch = str(g["ch"])
            inner = str(g["inner"])
            w = int(g["w"])  # type: ignore[arg-type]
            h = int(g["h"])  # type: ignore[arg-type]

            out.append(f'<g transform="translate({cur_x} {cur_y})">')
            out.append(inner.rstrip())

            if args.labels:
                out.append(f'<text class="lbl" x="0" y="{h + 16}">{ch} U+{cp:04X}</text>')

            out.append("</g>")

            cur_x += w + args.gap_x

        cur_y += row_h + args.gap_y

    out.append("</svg>")
    out.append("")

    write_text_lf(out_path, "\n".join(out))
    print(f"✓ Wrote {out_path.as_posix()} ({sheet_w}×{sheet_h}), glyphs: {len(glyphs)}")


if __name__ == "__main__":
    main()
