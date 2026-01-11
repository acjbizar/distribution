#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-characters-waterweg.py

Takes existing glyph SVGs (character-uXXXX.svg) from src/ and writes a "waterway"
styled version into src/waterweg/:

- Replaces the main black stroke group (<g fill="none" stroke="#000"...><path/></g>)
  with:
    1) a solid blue "water body" stroke
    2) light-blue squiggly foam lines that FOLLOW the stroke direction
       (by drawing along the same paths, then applying displacement filters),
       masked so the foam stays inside the stroke.

- Recolors dot-only/circle groups (like '.') from black to the same blue.

Usage:
  python tools/generate-characters-waterweg.py
  python tools/generate-characters-waterweg.py --in-dir src --out-dir src/waterweg
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

RE_GLYPH = re.compile(r"^character-u[0-9a-fA-F]{4,}\.svg$")

# Style constants (tweak freely)
WATER_BLUE = "#1f6fe5"
FOAM_BLUE = "#bfe8ff"


def q(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def fmt(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def parse_float(s: Optional[str], default: float) -> float:
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def is_black(s: Optional[str]) -> bool:
    if not s:
        return False
    v = s.strip().lower()
    return v in ("#000", "#000000", "black")


def clone_el(el: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(el, encoding="utf-8"))


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def ensure_xml_decl(xml_body: str) -> str:
    body = xml_body.strip()
    if not body.startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body + "\n"


def find_first_path_stroke_group(root: ET.Element) -> Optional[ET.Element]:
    """
    Find a <g> with fill="none", black stroke, and at least one <path>.
    Matches your generator output.
    """
    for el in list(root):
        if el.tag != q("g"):
            continue
        if (el.get("fill") or "").strip().lower() != "none":
            continue
        if not is_black(el.get("stroke")):
            continue
        if any(child.tag == q("path") for child in list(el)):
            return el
    return None


def find_circle_groups(root: ET.Element) -> List[ET.Element]:
    """
    Find <g> groups that contain <circle> children and look like your dot groups.
    """
    out: List[ET.Element] = []
    for el in list(root):
        if el.tag != q("g"):
            continue
        if not any(c.tag == q("circle") for c in list(el)):
            continue
        fill = el.get("fill")
        stroke = el.get("stroke")
        if is_black(fill) or is_black(stroke):
            out.append(el)
    return out


def get_or_create_defs(root: ET.Element) -> ET.Element:
    for el in list(root):
        if el.tag == q("defs"):
            return el
    defs = ET.Element(q("defs"))
    # Put <defs> early (after <desc> if present, else at top)
    insert_at = 0
    kids = list(root)
    for i, kid in enumerate(kids):
        if kid.tag == q("desc"):
            insert_at = i + 1
            break
    root.insert(insert_at, defs)
    return defs


def build_waterweg_defs(
    defs: ET.Element,
    view_w: float,
    view_h: float,
    paths: List[ET.Element],
    stroke_w: float,
    cap: str,
    join: str,
) -> None:
    """
    Adds mask + filters into <defs>. Safe to add repeatedly in separate files
    (IDs are per-document).
    """
    # Mask: the thick stroke area
    mask = ET.SubElement(defs, q("mask"), {"id": "glyphStrokeMask"})
    ET.SubElement(mask, q("rect"), {
        "x": "0", "y": "0",
        "width": fmt(view_w), "height": fmt(view_h),
        "fill": "#000"
    })
    mg = ET.SubElement(mask, q("g"), {
        "fill": "none",
        "stroke": "#fff",
        "stroke-width": fmt(stroke_w),
        "stroke-linecap": cap,
        "stroke-linejoin": join,
    })
    for p in paths:
        mg.append(clone_el(p))

    # Strong scribble filter (line wiggle)
    f_scrib = ET.SubElement(defs, q("filter"), {
        "id": "scribbleStrong",
        "x": "-25%", "y": "-25%",
        "width": "150%", "height": "150%",
    })
    ET.SubElement(f_scrib, q("feTurbulence"), {
        "type": "fractalNoise",
        "baseFrequency": "0.030",
        "numOctaves": "3",
        "seed": "11",
        "result": "n1",
    })
    ET.SubElement(f_scrib, q("feTurbulence"), {
        "type": "fractalNoise",
        "baseFrequency": "0.070",
        "numOctaves": "1",
        "seed": "23",
        "result": "n2",
    })
    ET.SubElement(f_scrib, q("feComposite"), {
        "in": "n1",
        "in2": "n2",
        "operator": "arithmetic",
        "k1": "0",
        "k2": "0.7",
        "k3": "0.6",
        "k4": "0",
        "result": "n",
    })
    ET.SubElement(f_scrib, q("feDisplacementMap"), {
        "in": "SourceGraphic",
        "in2": "n",
        "scale": "7.8",
        "xChannelSelector": "R",
        "yChannelSelector": "G",
    })

    # Random-ish offset fields (push foam away from center a bit, differently per thread)
    def make_offset(fid: str, base_freq: str, seed: str, scale: str) -> None:
        f = ET.SubElement(defs, q("filter"), {
            "id": fid,
            "x": "-30%", "y": "-30%",
            "width": "160%", "height": "160%",
        })
        ET.SubElement(f, q("feTurbulence"), {
            "type": "fractalNoise",
            "baseFrequency": base_freq,
            "numOctaves": "2",
            "seed": seed,
            "result": "o",
        })
        ET.SubElement(f, q("feDisplacementMap"), {
            "in": "SourceGraphic",
            "in2": "o",
            "scale": scale,
            "xChannelSelector": "R",
            "yChannelSelector": "G",
        })

    make_offset("offsetA", "0.012", "101", "6.0")
    make_offset("offsetB", "0.010", "202", "9.0")
    make_offset("offsetC", "0.014", "303", "12.0")

    # Slight softening
    f_soft = ET.SubElement(defs, q("filter"), {
        "id": "foamSoft",
        "x": "-20%", "y": "-20%",
        "width": "140%", "height": "140%",
    })
    ET.SubElement(f_soft, q("feGaussianBlur"), {"stdDeviation": "0.35"})


def build_waterweg_groups(
    paths: List[ET.Element],
    stroke_w: float,
    cap: str,
    join: str,
) -> List[ET.Element]:
    """
    Returns elements to insert where the original black stroke group was.
    """
    # 1) Water body stroke
    water = ET.Element(q("g"), {
        "fill": "none",
        "stroke": WATER_BLUE,
        "stroke-width": fmt(stroke_w),
        "stroke-linecap": cap,
        "stroke-linejoin": join,
    })
    for p in paths:
        water.append(clone_el(p))

    # 2) Foam lines: along the same paths, squiggled + masked into the stroke
    foam_wrap = ET.Element(q("g"), {
        "mask": "url(#glyphStrokeMask)",
        "fill": "none",
        "stroke-linecap": cap,
        "stroke-linejoin": join,
    })

    # Thread helper
    def add_thread(
        dash: str,
        dashoffset: str,
        width: float,
        opacity: str,
        offset_filter: str,
        use_soft: bool,
    ) -> None:
        g = ET.SubElement(foam_wrap, q("g"), {
            "stroke": FOAM_BLUE,
            "stroke-width": fmt(width),
            "opacity": opacity,
            "stroke-dasharray": dash,
            "stroke-dashoffset": dashoffset,
            "filter": f"url(#{offset_filter})",
        })
        inner = ET.SubElement(g, q("g"), {"filter": "url(#scribbleStrong)"})
        if use_soft:
            inner2 = ET.SubElement(inner, q("g"), {"filter": "url(#foamSoft)"})
        else:
            inner2 = inner

        for p in paths:
            inner2.append(clone_el(p))

    # Spaced further apart + varied offsets from center
    add_thread(dash="6 24", dashoffset="3",  width=2.6, opacity="0.75", offset_filter="offsetA", use_soft=True)
    add_thread(dash="5 30", dashoffset="17", width=2.1, opacity="0.60", offset_filter="offsetB", use_soft=True)
    add_thread(dash="3 36", dashoffset="9",  width=1.7, opacity="0.40", offset_filter="offsetC", use_soft=False)

    return [water, foam_wrap]


def recolor_dot_groups_to_water(circle_groups: List[ET.Element]) -> None:
    for g in circle_groups:
        g.set("fill", WATER_BLUE)
        g.set("stroke", WATER_BLUE)
        # keep original stroke-width; dots remain consistent with your generator


def transform_svg(root: ET.Element) -> bool:
    """
    Returns True if anything changed.
    """
    changed = False

    # Viewbox size for mask rect
    vb = root.get("viewBox") or ""
    parts = vb.replace(",", " ").split()
    view_w = float(parts[2]) if len(parts) == 4 else 240.0
    view_h = float(parts[3]) if len(parts) == 4 else 320.0

    main_g = find_first_path_stroke_group(root)
    circle_groups = find_circle_groups(root)

    # Always recolor dots if present ('.', 'i', '?', etc.)
    if circle_groups:
        recolor_dot_groups_to_water(circle_groups)
        changed = True

    if main_g is None:
        # dot-only glyphs are still valid
        return changed

    # Pull original stroke style from main group
    orig_sw = parse_float(main_g.get("stroke-width"), default=9.0)
    cap = main_g.get("stroke-linecap") or "round"
    join = main_g.get("stroke-linejoin") or "round"

    # Give room for interior foam (like your highway: 9 -> 12)
    water_sw = orig_sw * (4.0 / 3.0)

    paths = [c for c in list(main_g) if c.tag == q("path")]
    if not paths:
        return changed

    defs = get_or_create_defs(root)

    # Add our defs (mask + filters). If you re-run, duplicates don’t matter per file,
    # but we try to avoid adding if already present.
    existing_ids = {el.get("id") for el in defs.iter() if el.get("id")}
    if "glyphStrokeMask" not in existing_ids:
        build_waterweg_defs(defs, view_w, view_h, paths, water_sw, cap, join)

    # Replace the original black stroke group with (water + foam)
    kids = list(root)
    idx = kids.index(main_g)
    root.remove(main_g)

    new_groups = build_waterweg_groups(paths, water_sw, cap, join)
    for j, ng in enumerate(new_groups):
        root.insert(idx + j, ng)

    changed = True
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="src", help="Input directory containing character-uXXXX.svg")
    ap.add_argument("--out-dir", default="src/waterweg", help="Output directory for waterway-style glyphs")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and RE_GLYPH.match(p.name)],
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise SystemExit(f"No glyph SVGs found in {in_dir.resolve()} (expected character-uXXXX.svg).")

    written = 0
    skipped = 0

    for p in files:
        data = p.read_text(encoding="utf-8")
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            print(f"✗ {p.name}: XML parse error: {e}")
            skipped += 1
            continue

        if root.tag != q("svg"):
            print(f"✗ {p.name}: not an <svg> root")
            skipped += 1
            continue

        changed = transform_svg(root)
        if not changed:
            print(f"… {p.name}: no matching content to waterweg-ify (skipped)")
            skipped += 1
            continue

        xml = ET.tostring(root, encoding="unicode")
        xml = ensure_xml_decl(xml)

        write_text_lf(out_dir / p.name, xml)
        print(f"✓ {p.name} -> {out_dir / p.name}")
        written += 1

    print(f"\nDone. Wrote {written} waterweg glyph(s) into {out_dir.resolve()}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
