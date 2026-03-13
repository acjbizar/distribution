#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path as FSPath
from typing import Any, Dict, List, Tuple


# -----------------------------
# Grid / rendering constants
# -----------------------------
R = 40.0
DX = 2.0 * R
DY = 2.0 * R

X0 = 40.0
Y0 = 40.0

VIEW_H_DEFAULT = 320

STROKE = 9.0

# Debug overlay
GRID_OPACITY = 0.35
GRID_STROKE = 1.0
ANCHOR_R = 3.5

DOT_R = 1.0


# -----------------------------
# Utils
# -----------------------------
def write_text_lf(path: FSPath, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def load_glyph_data_module(data_file: FSPath):
    spec = importlib.util.spec_from_file_location("distribution_glyphs_data", data_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {data_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------
# Shape helpers
# -----------------------------
def line_path(shape: Dict[str, Any]) -> str:
    p1 = shape["p1"]
    p2 = shape["p2"]
    return f"M {fmt(float(p1['x']))} {fmt(float(p1['y']))} L {fmt(float(p2['x']))} {fmt(float(p2['y']))}"


def arc_sweep_for_variant(p1: Dict[str, Any], p2: Dict[str, Any], variant: int) -> int:
    # Must match the editor/export logic
    import math

    center = (
        {"x": p1["x"], "y": p2["y"]}
        if variant == 0
        else {"x": p2["x"], "y": p1["y"]}
    )

    a1 = math.atan2(float(p1["y"]) - float(center["y"]), float(p1["x"]) - float(center["x"]))
    a2 = math.atan2(float(p2["y"]) - float(center["y"]), float(p2["x"]) - float(center["x"]))
    delta = a2 - a1

    while delta <= -math.pi:
        delta += math.pi * 2
    while delta > math.pi:
        delta -= math.pi * 2

    return 1 if delta > 0 else 0


def arc_path(shape: Dict[str, Any]) -> str:
    p1 = shape["p1"]
    p2 = shape["p2"]
    variant = int(shape.get("variant", 0))
    sweep = arc_sweep_for_variant(p1, p2, variant)

    return (
        f"M {fmt(float(p1['x']))} {fmt(float(p1['y']))} "
        f"A {fmt(R)} {fmt(R)} 0 0 {sweep} {fmt(float(p2['x']))} {fmt(float(p2['y']))}"
    )


def collect_anchor_points(shapes: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []

    def add_point(x: float, y: float) -> None:
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            out.append((x, y))

    for shape in shapes:
        st = shape["type"]
        if st == "dot":
            p = shape["p"]
            add_point(float(p["x"]), float(p["y"]))
        elif st in ("line", "arc"):
            p1 = shape["p1"]
            p2 = shape["p2"]
            add_point(float(p1["x"]), float(p1["y"]))
            add_point(float(p2["x"]), float(p2["y"]))

    return out


# -----------------------------
# SVG rendering
# -----------------------------
def svg_glyph_doc(
    glyph_name: str,
    codepoint: int,
    shapes: List[Dict[str, Any]],
    debug: bool,
    view_w: int,
    view_h: int,
    guide_cols: int,
) -> str:
    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {view_h}" width="{view_w}" height="{view_h}">'
    )
    out.append(f'<desc>glyph: {glyph_name} U+{codepoint:04X}</desc>')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')

    if debug:
        out.append(
            f'<g fill="none" stroke="#bdbdbd" stroke-width="{fmt(GRID_STROKE)}" opacity="{fmt(GRID_OPACITY)}">'
        )
        rows = int(view_h // DY)
        for row in range(rows):
            for col in range(guide_cols):
                cx = X0 + col * DX
                cy = Y0 + row * DY
                out.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(R)}" />')
        out.append("</g>")

    path_ds: List[str] = []
    dots: List[Tuple[float, float, float]] = []

    for shape in shapes:
        st = shape["type"]
        if st == "line":
            path_ds.append(line_path(shape))
        elif st == "arc":
            path_ds.append(arc_path(shape))
        elif st == "dot":
            p = shape["p"]
            dots.append((float(p["x"]), float(p["y"]), DOT_R))
        else:
            raise ValueError(f"Unsupported shape type: {st!r}")

    out.append(
        f'<g fill="none" stroke="#000" stroke-width="{fmt(STROKE)}" '
        f'stroke-linecap="butt" stroke-linejoin="round">'
    )
    for d in path_ds:
        out.append(f'<path d="{d}" />')
    out.append("</g>")

    if dots:
        out.append(f'<g fill="#000" stroke="#000" stroke-width="{fmt(STROKE)}">')
        for (cx, cy, rr) in dots:
            out.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(rr)}" />')
        out.append("</g>")

    if debug:
        anchors = collect_anchor_points(shapes)
        if anchors:
            out.append('<g fill="#1e6bff" stroke="none">')
            for (x, y) in anchors:
                out.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(ANCHOR_R)}" />')
            out.append("</g>")

    out.append("</svg>")
    out.append("")
    return "\n".join(out)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="data/glyphs.py", help="Path to glyph data file (default: data/glyphs.py)")
    ap.add_argument("--out-dir", default="src", help="Output directory for normal SVGs (default: src)")
    ap.add_argument("--only", default="", help="Only render these characters")
    args = ap.parse_args()

    data_file = FSPath(args.data_file)
    out_dir = FSPath(args.out_dir)
    debug_dir = out_dir / "debug"

    if not data_file.exists():
        raise SystemExit(f"Data file not found: {data_file}")

    module = load_glyph_data_module(data_file)

    if not hasattr(module, "GLYPHS"):
        raise SystemExit(f"{data_file} does not define GLYPHS")

    glyphs: List[Dict[str, Any]] = list(module.GLYPHS)

    if args.only:
        only = set(args.only)
        glyphs = [g for g in glyphs if g["char"] in only]

    glyphs.sort(key=lambda g: int(g["codepoint"]))

    written_normal = 0
    written_debug = 0

    for glyph in glyphs:
        ch = glyph["char"]
        codepoint = int(glyph["codepoint"])
        shapes = glyph["shapes"]

        view_w = int(glyph.get("width", 240))
        view_h = int(glyph.get("height", VIEW_H_DEFAULT))
        guide_cols = int(glyph.get("guide_cols", max(1, round(view_w / DX))))

        out_name = glyph.get("filename") or f"character-u{codepoint:04x}.svg"

        svg_normal = svg_glyph_doc(
            glyph_name=ch,
            codepoint=codepoint,
            shapes=shapes,
            debug=False,
            view_w=view_w,
            view_h=view_h,
            guide_cols=guide_cols,
        )

        svg_debug = svg_glyph_doc(
            glyph_name=ch,
            codepoint=codepoint,
            shapes=shapes,
            debug=True,
            view_w=view_w,
            view_h=view_h,
            guide_cols=guide_cols,
        )

        write_text_lf(out_dir / out_name, svg_normal)
        write_text_lf(debug_dir / out_name, svg_debug)

        written_normal += 1
        written_debug += 1
        print(f"✓ {ch} -> {out_name}")
        print(f"  ├─ normal: {(out_dir / out_name).as_posix()}")
        print(f"  └─ debug : {(debug_dir / out_name).as_posix()}")

    print(f"\nDone.")
    print(f"Wrote {written_normal} normal glyph(s) into {out_dir.resolve()}")
    print(f"Wrote {written_debug} debug glyph(s) into {debug_dir.resolve()}")


if __name__ == "__main__":
    main()