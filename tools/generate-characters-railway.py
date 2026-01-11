#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-characters-railway.py

Reads existing glyph SVGs (character-uXXXX.svg) from src/ and writes a "railway"
styled version into src/railway/.

What it does:
- Replaces the main black stroke group (<g fill="none" stroke="#000"...><path/></g>)
  with a layered railway look:
    1) Track bed (dark ballast) as a wide stroke
    2) Rails (steel) as two strokes (one stronger, one softer highlight)
    3) Sleepers/ties as stamped dashes along the same paths, clipped to the bed
- Recolors dot/circle-only glyphs (like '.') to the ballast color.

Usage:
  python tools/generate-characters-railway.py
  python tools/generate-characters-railway.py --in-dir src --out-dir src/railway
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

# Palette (tweak as you like)
BALLAST = "#2b2b2b"
STEEL = "#b8b8b8"
TIE1 = "#7a4f2a"
TIE2 = "#6b4121"


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
    Find <g> groups that contain <circle> children and are styled as black.
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
    # Put <defs> early (after <desc> if present)
    insert_at = 0
    kids = list(root)
    for i, kid in enumerate(kids):
        if kid.tag == q("desc"):
            insert_at = i + 1
            break
    root.insert(insert_at, defs)
    return defs


def add_railway_defs(
    defs: ET.Element,
    view_w: float,
    view_h: float,
    paths: List[ET.Element],
    mask_stroke_w: float,
    cap: str,
    join: str,
) -> None:
    """
    Adds:
    - mask id="trackMask" that matches the track bed
    - filter id="tieJitter" for slight sleeper irregularity
    """
    # track mask
    mask = ET.SubElement(defs, q("mask"), {"id": "trackMask"})
    ET.SubElement(mask, q("rect"), {
        "x": "0", "y": "0",
        "width": fmt(view_w), "height": fmt(view_h),
        "fill": "#000"
    })
    mg = ET.SubElement(mask, q("g"), {
        "fill": "none",
        "stroke": "#fff",
        "stroke-width": fmt(mask_stroke_w),
        "stroke-linecap": cap,
        "stroke-linejoin": join,
    })
    for p in paths:
        mg.append(clone_el(p))

    # jitter filter for ties
    f = ET.SubElement(defs, q("filter"), {
        "id": "tieJitter",
        "x": "-20%", "y": "-20%",
        "width": "140%", "height": "140%",
    })
    ET.SubElement(f, q("feTurbulence"), {
        "type": "fractalNoise",
        "baseFrequency": "0.020",
        "numOctaves": "2",
        "seed": "9",
        "result": "n",
    })
    ET.SubElement(f, q("feDisplacementMap"), {
        "in": "SourceGraphic",
        "in2": "n",
        "scale": "1.6",
        "xChannelSelector": "R",
        "yChannelSelector": "G",
    })


def build_railway_groups(
    paths: List[ET.Element],
    orig_sw: float,
    cap: str,
    join: str,
) -> List[ET.Element]:
    """
    Creates the replacement group stack:
      - ballast bed
      - rails (steel)
      - ties (masked)
    """
    # Proportions: original 9 -> bed 14-ish
    bed_w = orig_sw * (14.0 / 9.0)
    rail_w = max(1.0, orig_sw * (3.2 / 9.0))
    rail_w2 = rail_w  # second rail layer (visual highlight)
    tie_w1 = max(1.0, orig_sw * (6.0 / 9.0))
    tie_w2 = max(1.0, orig_sw * (5.0 / 9.0))

    # 1) Ballast / bed
    bed = ET.Element(q("g"), {
        "fill": "none",
        "stroke": BALLAST,
        "stroke-width": fmt(bed_w),
        "stroke-linecap": cap,
        "stroke-linejoin": join,
    })
    for p in paths:
        bed.append(clone_el(p))

    # 2) Rails (steel)
    rails = ET.Element(q("g"), {
        "fill": "none",
        "stroke-linecap": cap,
        "stroke-linejoin": join,
        "opacity": "0.95",
    })

    r1 = ET.SubElement(rails, q("g"), {
        "stroke": STEEL,
        "stroke-width": fmt(rail_w),
    })
    for p in paths:
        r1.append(clone_el(p))

    r2 = ET.SubElement(rails, q("g"), {
        "stroke": STEEL,
        "stroke-width": fmt(rail_w2),
        "opacity": "0.75",
    })
    for p in paths:
        r2.append(clone_el(p))

    # 3) Sleepers / ties (masked inside bed)
    ties = ET.Element(q("g"), {
        "mask": "url(#trackMask)",
        "filter": "url(#tieJitter)",
        "opacity": "0.9",
    })

    # NOTE: SVG dashes follow the path direction automatically (good enough visually).
    # These read as repeated ties along the track.
    t1 = ET.SubElement(ties, q("g"), {
        "fill": "none",
        "stroke": TIE1,
        "stroke-width": fmt(tie_w1),
        "stroke-linecap": "square",
        "stroke-dasharray": "1 18",
    })
    for p in paths:
        t1.append(clone_el(p))

    t2 = ET.SubElement(ties, q("g"), {
        "fill": "none",
        "stroke": TIE2,
        "stroke-width": fmt(tie_w2),
        "stroke-linecap": "square",
        "stroke-dasharray": "1 26",
        "stroke-dashoffset": "9",
        "opacity": "0.65",
    })
    for p in paths:
        t2.append(clone_el(p))

    return [bed, rails, ties]


def recolor_dot_groups(circle_groups: List[ET.Element]) -> None:
    # For dot-only glyphs we just make them ballast-y (you can change this to STEEL if you prefer).
    for g in circle_groups:
        g.set("fill", BALLAST)
        g.set("stroke", BALLAST)


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

    # Dots first ('.', 'i', '?', etc.)
    circle_groups = find_circle_groups(root)
    if circle_groups:
        recolor_dot_groups(circle_groups)
        changed = True

    main_g = find_first_path_stroke_group(root)
    if main_g is None:
        return changed  # dot-only file still ok

    paths = [c for c in list(main_g) if c.tag == q("path")]
    if not paths:
        return changed

    cap = main_g.get("stroke-linecap") or "round"
    join = main_g.get("stroke-linejoin") or "round"
    orig_sw = parse_float(main_g.get("stroke-width"), default=9.0)

    bed_w = orig_sw * (14.0 / 9.0)
    mask_w = max(bed_w, orig_sw * (14.0 / 9.0))

    defs = get_or_create_defs(root)
    existing_ids = {el.get("id") for el in defs.iter() if el.get("id")}
    if "trackMask" not in existing_ids:
        add_railway_defs(defs, view_w, view_h, paths, mask_w, cap, join)

    # Replace original black stroke group
    kids = list(root)
    idx = kids.index(main_g)
    root.remove(main_g)

    new_groups = build_railway_groups(paths, orig_sw, cap, join)
    for j, ng in enumerate(new_groups):
        root.insert(idx + j, ng)

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="src", help="Input directory containing character-uXXXX.svg")
    ap.add_argument("--out-dir", default="src/treinspoor", help="Output directory for railway-style glyphs")
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
            print(f"… {p.name}: no matching content to railway-ify (skipped)")
            skipped += 1
            continue

        xml = ET.tostring(root, encoding="unicode")
        xml = ensure_xml_decl(xml)

        write_text_lf(out_dir / p.name, xml)
        print(f"✓ {p.name} -> {out_dir / p.name}")
        written += 1

    print(f"\nDone. Wrote {written} railway glyph(s) into {out_dir.resolve()}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
