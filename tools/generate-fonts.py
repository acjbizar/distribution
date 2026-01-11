#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-fonts.py

Build a variable font from your generated SVG glyphs that are drawn with strokes.
Axis: wght (maps to stroke thickness).

Input:  src/character-uXXXX.svg
Output: dist/fonts/distribution.ttf
        dist/fonts/distribution.woff2

Design assumptions (from your glyph generator):
- SVG viewBox heights are 320 units.
- Baseline is at y=240 in SVG coordinates.
- y grows downward in SVG; in fonts y grows upward.

Dependencies:
  pip install shapely fonttools brotli
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Required deps
# -----------------------------
try:
    from shapely.geometry import LineString, LinearRing, Point, Polygon, MultiPolygon
except Exception:
    print("Missing dependency: shapely")
    print("Install with:  pip install shapely")
    raise

try:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont
    from fontTools.varLib import build as var_build
except Exception:
    print("Missing dependency: fonttools")
    print("Install with:  pip install fonttools")
    raise


# -----------------------------
# SVG -> font mapping config
# -----------------------------
SVG_BASELINE_Y = 240.0
SVG_VIEW_H = 320.0

DEFAULT_UPM = 1000
DEFAULT_EXTERIOR_PTS = 96
DEFAULT_HOLE_PTS = 48
ARC_SEGMENTS_PER_CIRCLE = 64

SVG_FILE_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class CenterlinePath:
    pts: List[Tuple[float, float]]  # SVG coords (y down)
    closed: bool


@dataclass(frozen=True)
class DotSpec:
    cx: float
    cy: float
    r: float  # dot radius in SVG (your DOT_R=1)


@dataclass
class GlyphSpec:
    codepoint: int
    glyph_name: str
    adv_w_svg: float
    centerlines: List[CenterlinePath]
    dots: List[DotSpec]


# -----------------------------
# Small helpers
# -----------------------------
def glyph_name_from_cp(cp: int) -> str:
    if cp == 0x20:
        return "space"
    return f"uni{cp:04X}"


def parse_float(s: str) -> float:
    return float(s)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_area(pts: List[Tuple[float, float]]) -> float:
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def rotate_to_min_xy(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not pts:
        return pts
    i0 = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    return pts[i0:] + pts[:i0]


def ensure_direction(pts: List[Tuple[int, int]], clockwise: bool) -> List[Tuple[int, int]]:
    # pts is a ring without duplicated last point
    a = signed_area([(float(x), float(y)) for (x, y) in pts])
    is_ccw = a > 0
    if clockwise and is_ccw:
        pts = list(reversed(pts))
    if (not clockwise) and (not is_ccw):
        pts = list(reversed(pts))
    return pts

def rotate_to_seed(pts: List[Tuple[float, float]], seed: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Rotate a ring point list so it starts at the point closest to seed (both in same coord space)."""
    if not pts:
        return pts
    sx, sy = seed
    i0 = min(range(len(pts)), key=lambda i: (pts[i][0] - sx) ** 2 + (pts[i][1] - sy) ** 2)
    return pts[i0:] + pts[:i0]


def ensure_direction_f(pts: List[Tuple[float, float]], clockwise: bool) -> List[Tuple[float, float]]:
    """Ensure ring winding direction (float version)."""
    a = signed_area(pts)
    is_ccw = a > 0
    if clockwise and is_ccw:
        pts = list(reversed(pts))
    if (not clockwise) and (not is_ccw):
        pts = list(reversed(pts))
    return pts


def svg_to_font_xy(x: float, y: float, scale: float) -> Tuple[float, float]:
    # flip y and set baseline
    return (x * scale, (SVG_BASELINE_Y - y) * scale)


# -----------------------------
# SVG path parsing (only M, L, A; matches your generator)
# -----------------------------
def tokenize_path(d: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(d):
        ch = d[i]
        if ch in "MLA":
            out.append(ch)
            i += 1
            continue
        if ch.isspace() or ch == ",":
            i += 1
            continue
        m = NUM_RE.match(d, i)
        if not m:
            raise ValueError(f"Unexpected path data at: {d[i:i+20]!r}")
        out.append(m.group(0))
        i = m.end()
    return out


def arc_center_parameterization(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    large: int,
    sweep: int,
) -> Tuple[float, float, float, float]:
    """
    SVG arc center conversion specialized for:
    - rx == ry == r
    - xAxisRotation == 0
    Returns: (cx, cy, theta1, delta_theta) in SVG coords (y down).
    """
    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p = dx2
    y1p = dy2

    lam = (x1p * x1p + y1p * y1p) / (r * r)
    if lam > 1.0:
        r *= math.sqrt(lam)

    sign = 1.0 if (large != sweep) else -1.0
    denom = (x1p * x1p + y1p * y1p)
    if denom == 0:
        return (x1, y1, 0.0, 0.0)

    num = max(0.0, (r * r - x1p * x1p - y1p * y1p))
    factor = sign * math.sqrt(num / denom)
    cxp = factor * y1p
    cyp = -factor * x1p

    cx = cxp + (x1 + x2) / 2.0
    cy = cyp + (y1 + y2) / 2.0

    def unit(vx: float, vy: float) -> Tuple[float, float]:
        n = math.hypot(vx, vy)
        if n == 0:
            return (0.0, 0.0)
        return (vx / n, vy / n)

    v1x, v1y = unit((x1p - cxp) / r, (y1p - cyp) / r)
    v2x, v2y = unit((-x1p - cxp) / r, (-y1p - cyp) / r)

    theta1 = math.atan2(v1y, v1x)
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    delta = math.atan2(cross, dot)

    if sweep == 0 and delta > 0:
        delta -= 2.0 * math.pi
    elif sweep == 1 and delta < 0:
        delta += 2.0 * math.pi

    return (cx, cy, theta1, delta)


def sample_arc(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r: float,
    large: int,
    sweep: int,
) -> List[Tuple[float, float]]:
    cx, cy, theta1, delta = arc_center_parameterization(x1, y1, x2, y2, r, large, sweep)
    if abs(delta) < 1e-9:
        return [(x2, y2)]

    step = 2.0 * math.pi / ARC_SEGMENTS_PER_CIRCLE
    n = max(2, int(math.ceil(abs(delta) / step)))

    pts: List[Tuple[float, float]] = []
    for i in range(1, n + 1):
        t = i / n
        ang = theta1 + delta * t
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def path_d_to_centerline(d: str) -> CenterlinePath:
    toks = tokenize_path(d)
    i = 0
    pts: List[Tuple[float, float]] = []
    cur: Optional[Tuple[float, float]] = None

    while i < len(toks):
        cmd = toks[i]
        i += 1

        if cmd == "M":
            x = parse_float(toks[i])
            y = parse_float(toks[i + 1])
            i += 2
            cur = (x, y)
            pts.append(cur)

        elif cmd == "L":
            if cur is None:
                raise ValueError("L without current point")
            x = parse_float(toks[i])
            y = parse_float(toks[i + 1])
            i += 2
            cur = (x, y)
            pts.append(cur)

        elif cmd == "A":
            if cur is None:
                raise ValueError("A without current point")
            rx = parse_float(toks[i])
            ry = parse_float(toks[i + 1])
            i += 2
            _rot = parse_float(toks[i])
            i += 1
            large = int(float(toks[i]))
            sweep = int(float(toks[i + 1]))
            i += 2
            x = parse_float(toks[i])
            y = parse_float(toks[i + 1])
            i += 2

            if abs(rx - ry) > 1e-6:
                raise ValueError("Only circular arcs (rx==ry) supported for this glyph set.")

            pts.extend(sample_arc(cur[0], cur[1], x, y, rx, large, sweep))
            cur = (x, y)

        else:
            raise ValueError(f"Unsupported SVG path command: {cmd}")

    closed = False
    if len(pts) >= 3 and dist(pts[0], pts[-1]) < 1e-6:
        closed = True
        pts[-1] = pts[0]

    return CenterlinePath(pts=pts, closed=closed)


# -----------------------------
# Resampling rings (stable interpolation topology)
# -----------------------------
def resample_closed_ring(coords: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if not coords:
        return []
    if dist(coords[0], coords[-1]) > 1e-9:
        coords = coords + [coords[0]]

    ls = LineString(coords)
    total = ls.length
    if total <= 1e-9:
        p = coords[0]
        return [p for _ in range(n)]

    out: List[Tuple[float, float]] = []
    for k in range(n):
        d = (total * k) / n
        p = ls.interpolate(d)
        out.append((p.x, p.y))
    return out


# -----------------------------
# Buffer centerlines -> polygons -> fixed contours
# -----------------------------
def buffer_centerline_to_polygon(center: CenterlinePath, radius: float) -> Polygon:
    # cap_style=1 (round), join_style=1 (round)
    if center.closed:
        ring = LinearRing(center.pts)
        geom = ring.buffer(radius, cap_style=1, join_style=1, resolution=16)
    else:
        ls = LineString(center.pts)
        geom = ls.buffer(radius, cap_style=1, join_style=1, resolution=16)

    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        raise RuntimeError(
            "Buffer produced a MultiPolygon (disconnected outline). "
            "Try reducing --stroke-max/--stroke-min."
        )
    raise RuntimeError(f"Unexpected buffered geometry type: {type(geom)}")


def polygon_to_contours_fixed(
    poly: Polygon,
    exterior_pts: int,
    hole_pts: int,
    scale: float,
    seed_font: Optional[Tuple[float, float]] = None,
) -> List[List[Tuple[int, int]]]:
    contours: List[List[Tuple[int, int]]] = []

    # Exterior (float points in FONT coords)
    ext_coords = list(poly.exterior.coords)
    ext_svg = resample_closed_ring(ext_coords, exterior_pts)  # returns N points (not closed)
    ext_font_f = [svg_to_font_xy(x, y, scale) for (x, y) in ext_svg]

    # Make the start point stable across masters
    if seed_font is not None:
        ext_font_f = rotate_to_seed(ext_font_f, seed_font)
    else:
        # fallback if you ever call without seed
        # (avoid min-xy as primary strategy)
        pass

    ext_font_f = ensure_direction_f(ext_font_f, clockwise=True)

    # Round to ints after rotation + direction fixing
    ext_int = [(int(round(x)), int(round(y))) for (x, y) in ext_font_f]
    contours.append(ext_int)

    # Holes: usually not the problem; keep deterministic but simple
    for interior in poly.interiors:
        hole_coords = list(interior.coords)
        hole_svg = resample_closed_ring(hole_coords, hole_pts)
        hole_font_f = [svg_to_font_xy(x, y, scale) for (x, y) in hole_svg]
        # If you want: rotate holes too, but with their own seed (centroid). Not necessary for your case.
        hole_font_f = ensure_direction_f(hole_font_f, clockwise=False)
        hole_int = [(int(round(x)), int(round(y))) for (x, y) in hole_font_f]
        contours.append(hole_int)

    return contours

def build_glyph_tt(
    spec: GlyphSpec,
    stroke_width_svg: float,
    scale: float,
    exterior_pts: int,
    hole_pts: int,
) -> Tuple[object, int]:
    pen = TTGlyphPen(None)
    radius = stroke_width_svg / 2.0

    # Deterministic contour order:
    # - centerlines in SVG order
    # - dots in SVG order
    for cl in spec.centerlines:
        poly = buffer_centerline_to_polygon(cl, radius=radius)

        # Seed: first point of the ORIGINAL centerline, in FONT coords
        seed_font = svg_to_font_xy(cl.pts[0][0], cl.pts[0][1], scale)

        contours = polygon_to_contours_fixed(poly, exterior_pts, hole_pts, scale, seed_font=seed_font)

        for ring in contours:
            if len(ring) < 3:
                continue
            pen.moveTo(ring[0])
            for p in ring[1:]:
                pen.lineTo(p)
            pen.closePath()

    # Dots: SVG dots are drawn as filled circles *and* stroked with the same stroke width,
    # so the effective filled disk radius is (dot_r + stroke/2).
    for d in spec.dots:
        rr = d.r + radius
        poly = Point(d.cx, d.cy).buffer(rr, resolution=32)
        contours = polygon_to_contours_fixed(poly, exterior_pts, hole_pts, scale)
        for ring in contours:
            if len(ring) < 3:
                continue
            pen.moveTo(ring[0])
            for p in ring[1:]:
                pen.lineTo(p)
            pen.closePath()

    glyph = pen.glyph()
    adv_w = int(round(spec.adv_w_svg * scale))
    return glyph, adv_w


# -----------------------------
# SVG loading
# -----------------------------
def parse_svg_glyph(svg_path: Path) -> GlyphSpec:
    m = SVG_FILE_RE.match(svg_path.name)
    if not m:
        raise ValueError(f"Not a glyph SVG: {svg_path}")
    cp = int(m.group(1), 16)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = root.attrib.get("viewBox", "").strip()
    if not vb:
        raise ValueError(f"Missing viewBox in {svg_path}")
    parts = vb.split()
    if len(parts) != 4:
        raise ValueError(f"Unexpected viewBox in {svg_path}: {vb!r}")
    adv_w = float(parts[2])
    h = float(parts[3])
    if abs(h - SVG_VIEW_H) > 1e-3:
        print(f"Warning: {svg_path} viewBox height={h}, expected {SVG_VIEW_H}")

    centerlines: List[CenterlinePath] = []
    for el in root.iter():
        if el.tag.endswith("path"):
            d = el.attrib.get("d", "").strip()
            if d:
                centerlines.append(path_d_to_centerline(d))

    dots: List[DotSpec] = []
    for el in root.iter():
        if el.tag.endswith("circle"):
            try:
                r = float(el.attrib.get("r", "0"))
                if r <= 0:
                    continue
                # ignore grid circles (r ~ 40) and only keep tiny dots (r <= 2)
                if r > 2.0:
                    continue
                cx = float(el.attrib["cx"])
                cy = float(el.attrib["cy"])
                fill = (el.attrib.get("fill") or "").lower()
                stroke = (el.attrib.get("stroke") or "").lower()
                # heuristic: your dots are black-ish
                if ("#000" in fill) or ("#000" in stroke) or (fill == ""):
                    dots.append(DotSpec(cx=cx, cy=cy, r=r))
            except Exception:
                continue

    return GlyphSpec(
        codepoint=cp,
        glyph_name=glyph_name_from_cp(cp),
        adv_w_svg=adv_w,
        centerlines=centerlines,
        dots=dots,
    )


def load_glyphs_from_src(src_dir: Path) -> List[GlyphSpec]:
    svgs = sorted([p for p in src_dir.iterdir() if p.is_file() and SVG_FILE_RE.match(p.name)])
    if not svgs:
        raise FileNotFoundError(f"No glyph SVGs found in {src_dir} (expected src/character-uXXXX.svg).")
    return [parse_svg_glyph(p) for p in svgs]


# -----------------------------
# Master TTF build
# -----------------------------
def build_master_ttf(
    specs: List[GlyphSpec],
    out_path: Path,
    family: str,
    style: str,
    upm: int,
    ascent: int,
    descent: int,
    stroke_width_svg: float,
    scale: float,
    exterior_pts: int,
    hole_pts: int,
) -> None:
    glyph_order = [".notdef", "space"] + [s.glyph_name for s in specs]
    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder(glyph_order)

    glyf: Dict[str, object] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}

    # .notdef
    pen = TTGlyphPen(None)
    x0, y0 = 50, descent + 50
    x1, y1 = 450, ascent - 50
    pen.moveTo((x0, y0))
    pen.lineTo((x1, y0))
    pen.lineTo((x1, y1))
    pen.lineTo((x0, y1))
    pen.closePath()
    glyf[".notdef"] = pen.glyph()
    hmtx[".notdef"] = (500, 0)

    # space
    glyf["space"] = TTGlyphPen(None).glyph()
    hmtx["space"] = (int(round(160.0 * scale)), 0)

    for s in specs:
        g, aw = build_glyph_tt(
            s,
            stroke_width_svg=stroke_width_svg,
            scale=scale,
            exterior_pts=exterior_pts,
            hole_pts=hole_pts,
        )
        glyf[s.glyph_name] = g
        hmtx[s.glyph_name] = (aw, 0)

    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(hmtx)

    cmap: Dict[int, str] = {0x20: "space"}
    for s in specs:
        cmap[s.codepoint] = s.glyph_name
    fb.setupCharacterMap(cmap)

    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupOS2(
        sTypoAscender=ascent,
        sTypoDescender=descent,
        usWinAscent=ascent,
        usWinDescent=-descent,
    )
    fb.setupNameTable(
        {
            "familyName": family,
            "styleName": style,
            "uniqueFontIdentifier": f"{family}-{style}",
            "fullName": f"{family} {style}",
            "psName": f"{family.replace(' ', '')}-{style.replace(' ', '')}",
            "version": "Version 1.000",
        }
    )
    fb.setupPost()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(out_path)


# -----------------------------
# Variable font build
# -----------------------------
def build_variable_font(
    master_min_path: Path,
    master_reg_path: Path,
    master_max_path: Path,
    out_var_path: Path,
    family: str,
) -> None:
    """
    Build variable TTF using an in-memory designspace.

    IMPORTANT:
    Designspace locations are keyed by AXIS *NAME* (e.g. "Weight"),
    not the OpenType axis tag (e.g. "wght").
    """
    from fontTools.designspaceLib import DesignSpaceDocument, AxisDescriptor, SourceDescriptor
    from fontTools.varLib import build as var_build

    doc = DesignSpaceDocument()

    axis = AxisDescriptor()
    axis.tag = "wght"
    axis.name = "Weight"   # <-- axis NAME
    axis.minimum = 100
    axis.default = 400
    axis.maximum = 900
    doc.addAxis(axis)

    # NOTE: locations must use axis.name ("Weight"), NOT axis.tag ("wght")
    s_min = SourceDescriptor()
    s_min.path = str(master_min_path)
    s_min.name = "master_min"
    s_min.location = {"Weight": 100}
    doc.addSource(s_min)

    s_reg = SourceDescriptor()
    s_reg.path = str(master_reg_path)
    s_reg.name = "master_reg"
    s_reg.location = {"Weight": 400}   # default => base master
    doc.addSource(s_reg)

    s_max = SourceDescriptor()
    s_max.path = str(master_max_path)
    s_max.name = "master_max"
    s_max.location = {"Weight": 900}
    doc.addSource(s_max)

    out_var_path.parent.mkdir(parents=True, exist_ok=True)
    varfont, _, _ = var_build(doc)
    varfont.save(out_var_path)



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="src", help="Input SVG folder (default: src)")
    ap.add_argument("--out-dir", default="dist/fonts", help="Output folder (default: dist/fonts)")
    ap.add_argument("--family", default="Distribution", help="Font family name (default: Distribution)")
    ap.add_argument("--basename", default="distribution", help="Output base filename (default: distribution)")
    ap.add_argument("--upm", type=int, default=DEFAULT_UPM, help="Units per em (default: 1000)")
    ap.add_argument("--stroke-min", type=float, default=6.0, help="Min stroke width in SVG units (default: 6.0)")
    ap.add_argument("--stroke-max", type=float, default=14.0, help="Max stroke width in SVG units (default: 14.0)")
    ap.add_argument("--exterior-pts", type=int, default=DEFAULT_EXTERIOR_PTS, help="Points per exterior contour (default: 96)")
    ap.add_argument("--hole-pts", type=int, default=DEFAULT_HOLE_PTS, help="Points per hole contour (default: 48)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    specs = load_glyphs_from_src(src_dir)

    # scale so 320 SVG units == UPM
    scale = args.upm / SVG_VIEW_H
    ascent = int(round((SVG_BASELINE_Y - 0.0) * scale))
    descent = -int(round((SVG_VIEW_H - SVG_BASELINE_Y) * scale))

    build_tmp = out_dir / "_build_temp"
    build_tmp.mkdir(parents=True, exist_ok=True)

    master_min = build_tmp / f"{args.basename}-master-min.ttf"
    master_reg = build_tmp / f"{args.basename}-master-reg.ttf"
    master_max = build_tmp / f"{args.basename}-master-max.ttf"

    # Map wght 100..900 -> stroke_min..stroke_max linearly; default is wght=400
    w_min, w_def, w_max = 100.0, 400.0, 900.0
    t = (w_def - w_min) / (w_max - w_min)
    stroke_reg = args.stroke_min + t * (args.stroke_max - args.stroke_min)

    print(f"Loading {len(specs)} glyph(s) from {src_dir}…")
    print("Building masters:")
    print(f" - {master_min.name} (stroke={args.stroke_min})")
    print(f" - {master_reg.name} (stroke={stroke_reg:.3f})")
    print(f" - {master_max.name} (stroke={args.stroke_max})")

    build_master_ttf(
        specs=specs,
        out_path=master_min,
        family=args.family,
        style="MasterMin",
        upm=args.upm,
        ascent=ascent,
        descent=descent,
        stroke_width_svg=args.stroke_min,
        scale=scale,
        exterior_pts=args.exterior_pts,
        hole_pts=args.hole_pts,
    )

    build_master_ttf(
        specs=specs,
        out_path=master_reg,
        family=args.family,
        style="MasterReg",
        upm=args.upm,
        ascent=ascent,
        descent=descent,
        stroke_width_svg=stroke_reg,
        scale=scale,
        exterior_pts=args.exterior_pts,
        hole_pts=args.hole_pts,
    )

    build_master_ttf(
        specs=specs,
        out_path=master_max,
        family=args.family,
        style="MasterMax",
        upm=args.upm,
        ascent=ascent,
        descent=descent,
        stroke_width_svg=args.stroke_max,
        scale=scale,
        exterior_pts=args.exterior_pts,
        hole_pts=args.hole_pts,
    )

    out_var_ttf = out_dir / f"{args.basename}.ttf"
    out_var_woff2 = out_dir / f"{args.basename}.woff2"

    print(f"Building variable font: {out_var_ttf}")
    build_variable_font(
        master_min_path=master_min,
        master_reg_path=master_reg,
        master_max_path=master_max,
        out_var_path=out_var_ttf,
        family=args.family,
    )

    # WOFF2
    try:
        tt = TTFont(out_var_ttf)
        tt.flavor = "woff2"
        tt.save(out_var_woff2)
        print(f"Wrote {out_var_woff2}")
    except Exception as e:
        print("Could not write WOFF2 (environment may lack woff2 support).")
        print("Try: pip install brotli")
        print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
