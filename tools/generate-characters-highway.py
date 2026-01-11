#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-characters-highway.py

Reads existing glyph SVGs from src/ (character-uXXXX.svg),
replaces simple black strokes with a "highway style":
- dark grey road stroke (wider)
- dashed white center line (narrower)
and writes the result into src/snelweg/.

Also recolors dot/circle-only glyphs (like '.') to dark grey instead of skipping.

Usage:
  python tools/generate-characters-highway.py
  python tools/generate-characters-highway.py --in-dir src --out-dir src/snelweg
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

RE_GLYPH = re.compile(r"^character-u[0-9a-fA-F]{4,}\.svg$")


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
    # Clone via serialize/parse (safe for our tiny SVGs, keeps attributes)
    return ET.fromstring(ET.tostring(el, encoding="utf-8"))


def nice_dasharray(center_w: float) -> str:
    # Tuned to match your earlier example at center_w=3 => "10 12"
    dash = center_w * (10.0 / 3.0)
    gap = center_w * (12.0 / 3.0)

    def n(v: float) -> str:
        iv = int(round(v))
        return str(iv) if abs(v - iv) < 0.05 else fmt(v)

    return f"{n(dash)} {n(gap)}"


def find_first_path_stroke_group(root: ET.Element) -> Optional[ET.Element]:
    """
    Find a <g> with fill="none", black stroke, and at least one <path>.
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
    These are your dot groups.
    """
    out: List[ET.Element] = []
    for el in list(root):
        if el.tag != q("g"):
            continue
        if not any(child.tag == q("circle") for child in list(el)):
            continue

        fill = el.get("fill")
        stroke = el.get("stroke")

        # Your generator uses: fill="#000" stroke="#000" for dots.
        # We accept either.
        if is_black(fill) or is_black(stroke):
            out.append(el)
    return out


def highwayify_path_group(g: ET.Element) -> ET.Element:
    """
    Replace a simple stroke group with highway style:
    - road: #333 stroke, wider
    - center: #fff dashed stroke, narrower
    """
    orig_sw = parse_float(g.get("stroke-width"), default=9.0)
    orig_cap = g.get("stroke-linecap") or "round"
    orig_join = g.get("stroke-linejoin") or "round"

    # Same ratios as we used before: STROKE 9 -> road 12, center 3
    road_w = orig_sw * (4.0 / 3.0)
    center_w = max(1.0, orig_sw / 3.0)

    paths = [child for child in list(g) if child.tag == q("path")]

    highway = ET.Element(q("g"), {
        "fill": "none",
        "stroke-linecap": orig_cap,
        "stroke-linejoin": orig_join,
    })

    road = ET.SubElement(highway, q("g"), {
        "stroke": "#333",
        "stroke-width": fmt(road_w),
    })
    for p in paths:
        road.append(clone_el(p))

    center = ET.SubElement(highway, q("g"), {
        "stroke": "#fff",
        "stroke-width": fmt(center_w),
        "stroke-dasharray": nice_dasharray(center_w),
    })
    for p in paths:
        center.append(clone_el(p))

    return highway


def recolor_circle_group_to_road(g: ET.Element) -> None:
    """
    Dots aren't strokes, so we just make them asphalt colored.
    """
    g.set("fill", "#333")
    g.set("stroke", "#333")
    # keep original stroke-width if present; you can also scale it, but
    # this keeps dots looking like they did before (just recolored).


def ensure_xml_decl(xml_body: str) -> str:
    body = xml_body.strip()
    if not body.startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body + "\n"


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="src", help="Input directory containing character-uXXXX.svg")
    ap.add_argument("--out-dir", default="src/snelweg", help="Output directory for highway-style glyphs")
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

        changed = False

        # 1) Path-based stroke group -> highway style
        main_g = find_first_path_stroke_group(root)
        if main_g is not None:
            kids = list(root)
            idx = kids.index(main_g)
            replacement = highwayify_path_group(main_g)
            root.remove(main_g)
            root.insert(idx, replacement)
            changed = True

        # 2) Dot/circle groups -> recolor to road
        circle_groups = find_circle_groups(root)
        if circle_groups:
            for cg in circle_groups:
                recolor_circle_group_to_road(cg)
                changed = True

        if not changed:
            print(f"… {p.name}: no matching black stroke/dot groups found (skipped)")
            skipped += 1
            continue

        xml = ET.tostring(root, encoding="unicode")
        xml = ensure_xml_decl(xml)

        write_text_lf(out_dir / p.name, xml)
        print(f"✓ {p.name} -> {out_dir / p.name}")
        written += 1

    print(f"\nDone. Wrote {written} snelweg glyph(s) into {out_dir.resolve()}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
