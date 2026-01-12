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
            w, h = 240, 320  # fallback for your generator

    start = raw.find(">", m_open.start())
    end = raw.rfind("</svg>")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract inner SVG content from {svg_path}")

    inner = raw[start + 1 : end].strip()
    return inner + "\n", w, h


def list_glyph_files(svg_dir: Path) -> List[Tuple[int, Path]]:
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
        description="Generate src/snelweg/sheet.svg by composing src/snelweg/character-u{codepoint}.svg files found on disk."
    )
    ap.add_argument("--svg-dir", default="src/snelweg", help="Directory containing character-u*.svg (default: src/snelweg)")
    ap.add_argument("--out", default="src/snelweg/sheet.svg", help="Output sheet svg path (default: src/snelweg/sheet.svg)")
    ap.add_argument("--cols", type=int, default=10, help="Number of columns in sheet (default: 10)")
    ap.add_argument("--gap-x", type=int, default=0, help="Horizontal gap between glyphs (default: 0)")
    ap.add_argument("--gap-y", type=int, default=0, help="Vertical gap between glyph rows (default: 0)")
    ap.add_argument("--padding", type=int, default=0, help="Outer padding around sheet (default: 0)")
    ap.add_argument("--bg", default="#fff", help="Background fill (default: #fff). Use 'none' for transparent.")
    ap.add_argument("--labels", action="store_true", help="Draw small labels under each glyph (debug)")
    ap.add_argument("--label-pad-y", type=int, default=22, help="Extra vertical space reserved for labels (default: 22)")
    ap.add_argument("--label-offset-y", type=int, default=16, help="Label baseline below glyph box (default: 16)")
    ap.add_argument("--border", action="store_true", help="Draw a thin border around each glyph viewBox")
    ap.add_argument("--border-width", type=float, default=1.0, help="Border stroke width (default: 1.0)")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    out_path = Path(args.out)

    glyph_files = list_glyph_files(svg_dir)
    if not glyph_files:
        raise SystemExit(f"No glyph files found in {svg_dir.resolve()} matching character-uXXXX.svg")

    glyphs: List[Dict[str, object]] = []
    for cp, path in glyph_files:
        inner, w, h = read_glyph_svg(path)
        glyphs.append(
            {
                "cp": cp,
                "inner": inner,
                "w": w,
                "h": h,
                "ch": chr(cp) if 0 <= cp <= 0x10FFFF else "?",
            }
        )

    cols = max(1, args.cols)

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
        if args.labels:
            rh += args.label_pad_y
        row_heights.append(rh)
        row_widths.append(rw)

    sheet_w = args.padding * 2 + (max(row_widths) if row_widths else 0)
    sheet_h = args.padding * 2 + sum(row_heights) + args.gap_y * max(0, (len(rows) - 1))

    bg_rect = ""
    if args.bg.lower() != "none":
        bg_rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{args.bg}"/>\n'

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sheet_w} {sheet_h}">')
    out.append("<desc>Glyph sheet composed from src/snelweg/character-u{codepoint}.svg files</desc>")
    if bg_rect:
        out.append(bg_rect.rstrip())

    label_nodes: List[str] = []

    # Minimal styles
    styles: List[str] = []
    if args.labels:
        styles.append(".lbl{font:12px monospace; fill:#333;}")
    if args.border:
        styles.append(f".gbox{{fill:none; stroke:#888; stroke-width:{args.border_width};}}")
        # crisper border in many renderers (optional but helps)
        styles.append(".gbox{shape-rendering:crispEdges;}")
    if styles:
        out.append("<style>" + "".join(styles) + "</style>")

    cur_y = args.padding
    for row_idx, row in enumerate(rows):
        cur_x = args.padding

        base_row_glyph_h = 0
        for g in row:
            base_row_glyph_h = max(base_row_glyph_h, int(g["h"]))  # type: ignore[arg-type]

        for g in row:
            cp = int(g["cp"])  # type: ignore[arg-type]
            ch = str(g["ch"])
            inner = str(g["inner"])
            w = int(g["w"])  # type: ignore[arg-type]
            h = int(g["h"])  # type: ignore[arg-type]

            out.append(f'<g transform="translate({cur_x} {cur_y})">')

            # ✅ border around the glyph's own viewBox (0..w, 0..h)
            if args.border:
                # Put it behind content: draw first
                out.append(f'<rect class="gbox" x="0" y="0" width="{w}" height="{h}"/>')

            out.append(inner.rstrip())
            out.append("</g>")

            if args.labels:
                lx = cur_x
                ly = cur_y + base_row_glyph_h + args.label_offset_y
                label_nodes.append(f'<text class="lbl" x="{lx}" y="{ly}">{ch} U+{cp:04X}</text>')

            cur_x += w + args.gap_x

        cur_y += row_heights[row_idx] + args.gap_y

    if args.labels and label_nodes:
        out.append('<g id="labels">')
        out.extend(label_nodes)
        out.append("</g>")

    out.append("</svg>")
    out.append("")

    write_text_lf(out_path, "\n".join(out))
    print(f"✓ Wrote {out_path.as_posix()} (viewBox {sheet_w}×{sheet_h}), glyphs: {len(glyphs)}")


if __name__ == "__main__":
    main()
