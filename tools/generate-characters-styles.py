#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-characters-styles.py

Generate 3 styled variants of all existing glyph SVGs:

- Highway style  -> src/snelweg/
- Waterway style -> src/waterweg/
- Railway style  -> src/spoorweg/

AND enforce: stroke-linecap="butt" on all stroked elements/groups in the outputs.

Input SVGs are expected at: src/character-uXXXX.svg

Usage:
  python tools/generate-characters-styles.py
  python tools/generate-characters-styles.py --in-dir src
  python tools/generate-characters-styles.py --out-root src
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

# ---- palette ----
HWY_ROAD = "#333"
HWY_CENTER = "#fff"

WATER_BLUE = "#1f6fe5"
FOAM_BLUE = "#bfe8ff"

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


def deep_clone_root(root: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(root, encoding="utf-8"))


def ensure_xml_decl(xml_body: str) -> str:
    body = xml_body.strip()
    if not body.startswith("<?xml"):
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + body
    return body + "\n"


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def get_viewbox_wh(root: ET.Element) -> Tuple[float, float]:
    vb = root.get("viewBox") or ""
    parts = vb.replace(",", " ").split()
    if len(parts) == 4:
        try:
            return float(parts[2]), float(parts[3])
        except ValueError:
            pass
    return 240.0, 320.0


def find_first_path_stroke_group(root: ET.Element) -> Optional[ET.Element]:
    """
    Find <g fill="none" stroke="#000"...> that contains <path>.
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


def find_black_circle_groups(root: ET.Element) -> List[ET.Element]:
    out: List[ET.Element] = []
    for el in list(root):
        if el.tag != q("g"):
            continue
        if not any(c.tag == q("circle") for c in list(el)):
            continue
        if is_black(el.get("fill")) or is_black(el.get("stroke")):
            out.append(el)
    return out


def get_or_create_defs(root: ET.Element) -> ET.Element:
    for el in list(root):
        if el.tag == q("defs"):
            return el
    defs = ET.Element(q("defs"))
    # insert after <desc> if present
    insert_at = 0
    kids = list(root)
    for i, kid in enumerate(kids):
        if kid.tag == q("desc"):
            insert_at = i + 1
            break
    root.insert(insert_at, defs)
    return defs


def enforce_linecap_butt(root: ET.Element) -> None:
    """
    Set stroke-linecap="butt" on any element that looks like it participates in stroking.
    (Groups or shapes with stroke/stroke-width/dash props.)
    """
    candidates = {
        q("g"),
        q("path"),
        q("line"),
        q("polyline"),
        q("polygon"),
        q("rect"),
        q("circle"),
        q("ellipse"),
    }
    for el in root.iter():
        if el.tag not in candidates:
            continue
        if (
            el.get("stroke") is not None
            or el.get("stroke-width") is not None
            or el.get("stroke-dasharray") is not None
            or el.get("stroke-linecap") is not None
        ):
            el.set("stroke-linecap", "butt")


# -----------------------------
# STYLE 1: SNELWEG (highway)
# -----------------------------
def nice_dasharray(center_w: float) -> str:
    # tuned so center_w=3 -> "10 12"
    dash = center_w * (10.0 / 3.0)
    gap = center_w * (12.0 / 3.0)

    def n(v: float) -> str:
        iv = int(round(v))
        return str(iv) if abs(v - iv) < 0.05 else fmt(v)

    return f"{n(dash)} {n(gap)}"


def apply_snelweg(root: ET.Element) -> bool:
    changed = False

    main_g = find_first_path_stroke_group(root)
    circle_groups = find_black_circle_groups(root)

    # recolor dots to asphalt
    for g in circle_groups:
        g.set("fill", HWY_ROAD)
        g.set("stroke", HWY_ROAD)
        changed = True

    if main_g is None:
        return changed

    orig_sw = parse_float(main_g.get("stroke-width"), 9.0)
    orig_join = main_g.get("stroke-linejoin") or "round"

    road_w = orig_sw * (4.0 / 3.0)
    center_w = max(1.0, orig_sw / 3.0)

    paths = [c for c in list(main_g) if c.tag == q("path")]
    if not paths:
        return changed

    highway = ET.Element(q("g"), {
        "fill": "none",
        "stroke-linejoin": orig_join,
        "stroke-linecap": "butt",  # enforced anyway, but explicit
    })

    road = ET.SubElement(highway, q("g"), {
        "stroke": HWY_ROAD,
        "stroke-width": fmt(road_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": orig_join,
    })
    for p in paths:
        road.append(clone_el(p))

    center = ET.SubElement(highway, q("g"), {
        "stroke": HWY_CENTER,
        "stroke-width": fmt(center_w),
        "stroke-dasharray": nice_dasharray(center_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": orig_join,
    })
    for p in paths:
        center.append(clone_el(p))

    kids = list(root)
    idx = kids.index(main_g)
    root.remove(main_g)
    root.insert(idx, highway)

    return True


# -----------------------------
# STYLE 2: WATERWEG (waterway)
# -----------------------------
def build_waterweg_defs(
    defs: ET.Element,
    view_w: float,
    view_h: float,
    paths: List[ET.Element],
    stroke_w: float,
    join: str,
) -> None:
    # Mask for stroke area
    mask = ET.SubElement(defs, q("mask"), {"id": "glyphStrokeMask"})
    ET.SubElement(mask, q("rect"), {
        "x": "0", "y": "0", "width": fmt(view_w), "height": fmt(view_h), "fill": "#000"
    })
    mg = ET.SubElement(mask, q("g"), {
        "fill": "none",
        "stroke": "#fff",
        "stroke-width": fmt(stroke_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        mg.append(clone_el(p))

    # Strong scribble
    f_scrib = ET.SubElement(defs, q("filter"), {
        "id": "scribbleStrong", "x": "-25%", "y": "-25%", "width": "150%", "height": "150%"
    })
    ET.SubElement(f_scrib, q("feTurbulence"), {
        "type": "fractalNoise", "baseFrequency": "0.030", "numOctaves": "3", "seed": "11", "result": "n1"
    })
    ET.SubElement(f_scrib, q("feTurbulence"), {
        "type": "fractalNoise", "baseFrequency": "0.070", "numOctaves": "1", "seed": "23", "result": "n2"
    })
    ET.SubElement(f_scrib, q("feComposite"), {
        "in": "n1", "in2": "n2", "operator": "arithmetic",
        "k1": "0", "k2": "0.7", "k3": "0.6", "k4": "0", "result": "n"
    })
    ET.SubElement(f_scrib, q("feDisplacementMap"), {
        "in": "SourceGraphic", "in2": "n", "scale": "7.8", "xChannelSelector": "R", "yChannelSelector": "G"
    })

    # Offset filters
    def make_offset(fid: str, base_freq: str, seed: str, scale: str) -> None:
        f = ET.SubElement(defs, q("filter"), {
            "id": fid, "x": "-30%", "y": "-30%", "width": "160%", "height": "160%"
        })
        ET.SubElement(f, q("feTurbulence"), {
            "type": "fractalNoise", "baseFrequency": base_freq, "numOctaves": "2", "seed": seed, "result": "o"
        })
        ET.SubElement(f, q("feDisplacementMap"), {
            "in": "SourceGraphic", "in2": "o", "scale": scale, "xChannelSelector": "R", "yChannelSelector": "G"
        })

    make_offset("offsetA", "0.012", "101", "6.0")
    make_offset("offsetB", "0.010", "202", "9.0")
    make_offset("offsetC", "0.014", "303", "12.0")

    # softener
    f_soft = ET.SubElement(defs, q("filter"), {
        "id": "foamSoft", "x": "-20%", "y": "-20%", "width": "140%", "height": "140%"
    })
    ET.SubElement(f_soft, q("feGaussianBlur"), {"stdDeviation": "0.35"})


def apply_waterweg(root: ET.Element) -> bool:
    changed = False
    view_w, view_h = get_viewbox_wh(root)

    # recolor dot groups to water
    for g in find_black_circle_groups(root):
        g.set("fill", WATER_BLUE)
        g.set("stroke", WATER_BLUE)
        changed = True

    main_g = find_first_path_stroke_group(root)
    if main_g is None:
        return changed

    orig_sw = parse_float(main_g.get("stroke-width"), 9.0)
    join = main_g.get("stroke-linejoin") or "round"

    # match earlier look: 9 -> 12
    water_sw = orig_sw * (4.0 / 3.0)

    paths = [c for c in list(main_g) if c.tag == q("path")]
    if not paths:
        return changed

    defs = get_or_create_defs(root)
    existing_ids = {el.get("id") for el in defs.iter() if el.get("id")}
    if "glyphStrokeMask" not in existing_ids:
        build_waterweg_defs(defs, view_w, view_h, paths, water_sw, join)

    # water body
    water = ET.Element(q("g"), {
        "fill": "none",
        "stroke": WATER_BLUE,
        "stroke-width": fmt(water_sw),
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        water.append(clone_el(p))

    # foam wrapper
    foam_wrap = ET.Element(q("g"), {
        "mask": "url(#glyphStrokeMask)",
        "fill": "none",
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })

    def add_thread(dash: str, dashoffset: str, width: float, opacity: str, offset_filter: str, soft: bool) -> None:
        g = ET.SubElement(foam_wrap, q("g"), {
            "stroke": FOAM_BLUE,
            "stroke-width": fmt(width),
            "opacity": opacity,
            "stroke-dasharray": dash,
            "stroke-dashoffset": dashoffset,
            "filter": f"url(#{offset_filter})",
            "stroke-linecap": "butt",
            "stroke-linejoin": join,
        })
        inner = ET.SubElement(g, q("g"), {"filter": "url(#scribbleStrong)"})
        if soft:
            inner2 = ET.SubElement(inner, q("g"), {"filter": "url(#foamSoft)"})
        else:
            inner2 = inner
        for p in paths:
            inner2.append(clone_el(p))

    add_thread("6 24", "3", 2.6, "0.75", "offsetA", True)
    add_thread("5 30", "17", 2.1, "0.60", "offsetB", True)
    add_thread("3 36", "9", 1.7, "0.40", "offsetC", False)

    # replace main group
    kids = list(root)
    idx = kids.index(main_g)
    root.remove(main_g)
    root.insert(idx, water)
    root.insert(idx + 1, foam_wrap)

    return True


# -----------------------------
# STYLE 3: RAILWAY
# -----------------------------
def add_railway_defs(
    defs: ET.Element,
    view_w: float,
    view_h: float,
    paths: List[ET.Element],
    mask_stroke_w: float,
    join: str,
) -> None:
    mask = ET.SubElement(defs, q("mask"), {"id": "trackMask"})
    ET.SubElement(mask, q("rect"), {
        "x": "0", "y": "0", "width": fmt(view_w), "height": fmt(view_h), "fill": "#000"
    })
    mg = ET.SubElement(mask, q("g"), {
        "fill": "none",
        "stroke": "#fff",
        "stroke-width": fmt(mask_stroke_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        mg.append(clone_el(p))

    f = ET.SubElement(defs, q("filter"), {
        "id": "tieJitter", "x": "-20%", "y": "-20%", "width": "140%", "height": "140%"
    })
    ET.SubElement(f, q("feTurbulence"), {
        "type": "fractalNoise", "baseFrequency": "0.020", "numOctaves": "2", "seed": "9", "result": "n"
    })
    ET.SubElement(f, q("feDisplacementMap"), {
        "in": "SourceGraphic", "in2": "n", "scale": "1.6", "xChannelSelector": "R", "yChannelSelector": "G"
    })


def apply_railway(root: ET.Element) -> bool:
    changed = False
    view_w, view_h = get_viewbox_wh(root)

    # recolor dots to ballast
    for g in find_black_circle_groups(root):
        g.set("fill", BALLAST)
        g.set("stroke", BALLAST)
        changed = True

    main_g = find_first_path_stroke_group(root)
    if main_g is None:
        return changed

    paths = [c for c in list(main_g) if c.tag == q("path")]
    if not paths:
        return changed

    join = main_g.get("stroke-linejoin") or "round"
    orig_sw = parse_float(main_g.get("stroke-width"), 9.0)

    bed_w = orig_sw * (14.0 / 9.0)
    rail_w = max(1.0, orig_sw * (3.2 / 9.0))
    tie_w1 = max(1.0, orig_sw * (6.0 / 9.0))
    tie_w2 = max(1.0, orig_sw * (5.0 / 9.0))

    defs = get_or_create_defs(root)
    existing_ids = {el.get("id") for el in defs.iter() if el.get("id")}
    if "trackMask" not in existing_ids:
        add_railway_defs(defs, view_w, view_h, paths, bed_w, join)

    # bed
    bed = ET.Element(q("g"), {
        "fill": "none",
        "stroke": BALLAST,
        "stroke-width": fmt(bed_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        bed.append(clone_el(p))

    # rails (steel)
    rails = ET.Element(q("g"), {
        "fill": "none",
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
        "opacity": "0.95",
    })
    r1 = ET.SubElement(rails, q("g"), {
        "stroke": STEEL,
        "stroke-width": fmt(rail_w),
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        r1.append(clone_el(p))

    r2 = ET.SubElement(rails, q("g"), {
        "stroke": STEEL,
        "stroke-width": fmt(rail_w),
        "opacity": "0.75",
        "stroke-linecap": "butt",
        "stroke-linejoin": join,
    })
    for p in paths:
        r2.append(clone_el(p))

    # ties (masked, slightly jittered)
    ties = ET.Element(q("g"), {
        "mask": "url(#trackMask)",
        "filter": "url(#tieJitter)",
        "opacity": "0.9",
    })
    t1 = ET.SubElement(ties, q("g"), {
        "fill": "none",
        "stroke": TIE1,
        "stroke-width": fmt(tie_w1),
        "stroke-linecap": "butt",
        "stroke-dasharray": "1 18",
    })
    for p in paths:
        t1.append(clone_el(p))

    t2 = ET.SubElement(ties, q("g"), {
        "fill": "none",
        "stroke": TIE2,
        "stroke-width": fmt(tie_w2),
        "stroke-linecap": "butt",
        "stroke-dasharray": "1 26",
        "stroke-dashoffset": "9",
        "opacity": "0.65",
    })
    for p in paths:
        t2.append(clone_el(p))

    # replace
    kids = list(root)
    idx = kids.index(main_g)
    root.remove(main_g)
    root.insert(idx, bed)
    root.insert(idx + 1, rails)
    root.insert(idx + 2, ties)

    return True


# -----------------------------
# driver
# -----------------------------
def load_svg_root(path: Path) -> Optional[ET.Element]:
    try:
        data = path.read_text(encoding="utf-8")
        root = ET.fromstring(data)
    except Exception:
        return None
    if root.tag != q("svg"):
        return None
    return root


def render_svg(root: ET.Element) -> str:
    xml = ET.tostring(root, encoding="unicode")
    return ensure_xml_decl(xml)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="src", help="Input directory with character-uXXXX.svg files (default: src)")
    ap.add_argument("--out-root", default="src", help="Root output directory (default: src)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_root)

    out_snelweg = out_root / "snelweg"
    out_waterweg = out_root / "waterweg"
    out_spoorweg = out_root / "spoorweg"

    out_snelweg.mkdir(parents=True, exist_ok=True)
    out_waterweg.mkdir(parents=True, exist_ok=True)
    out_spoorweg.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and RE_GLYPH.match(p.name)],
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise SystemExit(f"No glyph SVGs found in {in_dir.resolve()} (expected character-uXXXX.svg).")

    written = {"snelweg": 0, "waterweg": 0, "spoorweg": 0}
    skipped = 0

    for p in files:
        base_root = load_svg_root(p)
        if base_root is None:
            print(f"✗ {p.name}: unreadable or not an <svg>")
            skipped += 1
            continue

        # --- snelweg ---
        r = deep_clone_root(base_root)
        if apply_snelweg(r):
            enforce_linecap_butt(r)
            write_text_lf(out_snelweg / p.name, render_svg(r))
            written["snelweg"] += 1
        else:
            # still enforce butt + write? if nothing changes, skip to avoid duplicates
            pass

        # --- waterweg ---
        r = deep_clone_root(base_root)
        if apply_waterweg(r):
            enforce_linecap_butt(r)
            write_text_lf(out_waterweg / p.name, render_svg(r))
            written["waterweg"] += 1
        else:
            pass

        # --- railway ---
        r = deep_clone_root(base_root)
        if apply_railway(r):
            enforce_linecap_butt(r)
            write_text_lf(out_spoorweg / p.name, render_svg(r))
            written["spoorweg"] += 1
        else:
            pass

        print(f"✓ {p.name} -> snelweg/waterweg/spoorweg")

    print(
        "\nDone."
        f"\n- snelweg : {written['snelweg']} file(s) -> {out_snelweg.resolve()}"
        f"\n- waterweg: {written['waterweg']} file(s) -> {out_waterweg.resolve()}"
        f"\n- spoorweg : {written['spoorweg']} file(s) -> {out_spoorweg.resolve()}"
        f"\nSkipped unreadable: {skipped}"
    )


if __name__ == "__main__":
    main()
