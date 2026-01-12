#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


CODEPOINT_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")

SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE | re.DOTALL)
VIEWBOX_RE = re.compile(r'viewBox\s*=\s*"([^"]+)"', re.IGNORECASE)
WIDTH_RE = re.compile(r'width\s*=\s*"([^"]+)"', re.IGNORECASE)
HEIGHT_RE = re.compile(r'height\s*=\s*"([^"]+)"', re.IGNORECASE)


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


def _parse_number(s: str) -> float:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Could not parse number from {s!r}")
    return float(m.group(0))


def read_glyph_svg(svg_path: Path) -> Tuple[str, int, int]:
    """
    Returns:
      inner_svg: content INSIDE <svg> ... </svg> (no outer tag)
      w, h: from viewBox (preferred) or width/height fallback
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
            # Fallback for your generator (typically 240×320 or 320×320)
            w, h = 240, 320

    start = raw.find(">", m_open.start())
    end = raw.rfind("</svg>")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not extract inner SVG content from {svg_path}")

    inner = raw[start + 1 : end].strip()
    return inner + "\n", w, h


def list_glyph_files(svg_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    if not svg_dir.exists():
        return items

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


def generate_sheet_for_dir(
    svg_dir: Path,
    out_path: Path,
    *,
    cols: int,
    gap_x: int,
    gap_y: int,
    padding: int,
    bg: str,
    labels: bool,
    label_pad_y: int,
    label_offset_y: int,
    border: bool,
    border_width: float,
) -> int:
    glyph_files = list_glyph_files(svg_dir)
    if not glyph_files:
        return 0

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

    cols = max(1, cols)
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
                rw += gap_x
            rw += int(g["w"])  # type: ignore[arg-type]
        if labels:
            rh += label_pad_y
        row_heights.append(rh)
        row_widths.append(rw)

    sheet_w = padding * 2 + (max(row_widths) if row_widths else 0)
    sheet_h = padding * 2 + sum(row_heights) + gap_y * max(0, len(rows) - 1)

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    # No width/height; viewBox only
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sheet_w} {sheet_h}">')
    out.append(f"<desc>Glyph sheet composed from {svg_dir.as_posix()}/character-u{{codepoint}}.svg</desc>")

    if bg.lower() != "none":
        out.append(f'<rect x="0" y="0" width="100%" height="100%" fill="{bg}"/>')

    label_nodes: List[str] = []
    styles: List[str] = []
    if labels:
        styles.append(".lbl{font:12px monospace; fill:#333;}")
    if border:
        styles.append(f".gbox{{fill:none; stroke:#888; stroke-width:{border_width}; shape-rendering:crispEdges;}}")
    if styles:
        out.append("<style>" + "".join(styles) + "</style>")

    cur_y = padding
    for row_idx, row in enumerate(rows):
        cur_x = padding

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

            if border:
                out.append(f'<rect class="gbox" x="0" y="0" width="{w}" height="{h}"/>')

            out.append(inner.rstrip())
            out.append("</g>")

            if labels:
                lx = cur_x
                ly = cur_y + base_row_glyph_h + label_offset_y
                label_nodes.append(f'<text class="lbl" x="{lx}" y="{ly}">{ch} U+{cp:04X}</text>')

            cur_x += w + gap_x

        cur_y += row_heights[row_idx] + gap_y

    if labels and label_nodes:
        out.append('<g id="labels">')
        out.extend(label_nodes)
        out.append("</g>")

    out.append("</svg>")
    out.append("")

    write_text_lf(out_path, "\n".join(out))
    return len(glyphs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate sheet.svg for multiple styles (snelweg, waterweg, spoorweg)."
    )
    ap.add_argument("--src-root", default="src", help="Root folder containing style directories (default: src)")
    ap.add_argument("--out-root", default="src", help="Root folder to write sheets into (default: src)")
    ap.add_argument(
        "--styles",
        default="snelweg,waterweg,spoorweg",
        help="Comma-separated style folder names (default: snelweg,waterweg,spoorweg)",
    )

    ap.add_argument("--cols", type=int, default=10, help="Number of columns in sheet (default: 10)")
    ap.add_argument("--gap-x", type=int, default=0, help="Horizontal gap between glyphs (default: 0)")
    ap.add_argument("--gap-y", type=int, default=0, help="Vertical gap between rows (default: 0)")
    ap.add_argument("--padding", type=int, default=0, help="Outer padding around sheet (default: 0)")
    ap.add_argument("--bg", default="#fff", help="Background fill (default: #fff). Use 'none' for transparent.")

    ap.add_argument("--labels", action="store_true", help="Draw labels (debug)")
    ap.add_argument("--label-pad-y", type=int, default=22, help="Extra vertical space reserved for labels (default: 22)")
    ap.add_argument("--label-offset-y", type=int, default=16, help="Label baseline below glyph box (default: 16)")

    ap.add_argument("--border", action="store_true", help="Draw a thin border around each glyph viewBox")
    ap.add_argument("--border-width", type=float, default=1.0, help="Border stroke width (default: 1.0)")

    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    total = 0
    for style in styles:
        style_dir = src_root / style
        out_path = out_root / style / "sheet.svg"

        count = generate_sheet_for_dir(
            style_dir,
            out_path,
            cols=args.cols,
            gap_x=args.gap_x,
            gap_y=args.gap_y,
            padding=args.padding,
            bg=args.bg,
            labels=args.labels,
            label_pad_y=args.label_pad_y,
            label_offset_y=args.label_offset_y,
            border=args.border,
            border_width=args.border_width,
        )

        if count == 0:
            print(f"⚠ No glyphs found for style '{style}' in {style_dir.as_posix()} (skipped)")
            continue

        total += count
        print(f"✓ {style}: wrote {out_path.as_posix()} ({count} glyphs)")

    print(f"\nDone. Total glyphs across sheets: {total}")


if __name__ == "__main__":
    main()
