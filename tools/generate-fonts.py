#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate-fonts.py

Variable font builder for centerline SVG glyphs.
Axis: wght (mapped to stroke width).

Upgrades in this version:
- Merge all buffered components (arcs/lines/dots) into ONE filled shape per glyph per master
  using shapely unary_union, before converting to contours. This removes overlap seams and
  drastically reduces interpolation artifacts (gaps/wiggles) between weights.
- Keep multi-master build (default 9 masters) + landmark-anchored resampling for stable
  point correspondence.

Input:
  src/character-uXXXX.svg

Output:
  dist/fonts/distribution.ttf
  dist/fonts/distribution.woff
  dist/fonts/distribution.woff2

Dependencies:
  pip install shapely fonttools brotli
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# -----------------------------
# Dependencies
# -----------------------------
try:
    from shapely.geometry import LineString, LinearRing, Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
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
SVG_VIEW_H = 320.0
SVG_BASELINE_Y = 240.0  # baseline at y=0 in font coords

DEFAULT_UPM = 1000

# Point counts (must be fixed across masters)
DEFAULT_EXTERIOR_PTS = 180
DEFAULT_HOLE_PTS = 110
DEFAULT_DOT_PTS = 44

# Dense sampling for stable landmarks
DEFAULT_LANDMARK_SAMPLES = 4096

# Arc sampling for SVG A command
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
    r: float


@dataclass
class GlyphSpec:
    codepoint: int
    glyph_name: str
    adv_w_svg: float
    centerlines: List[CenterlinePath]
    dots: List[DotSpec]
    linecap: str = "round"
    linejoin: str = "round"


# -----------------------------
# Helpers
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


def ensure_direction_f(pts: List[Tuple[float, float]], clockwise: bool) -> List[Tuple[float, float]]:
    a = signed_area(pts)
    is_ccw = a > 0
    if clockwise and is_ccw:
        pts = list(reversed(pts))
    if (not clockwise) and (not is_ccw):
        pts = list(reversed(pts))
    return pts


def svg_to_font_xy(x: float, y: float, scale: float) -> Tuple[float, float]:
    return (x * scale, (SVG_BASELINE_Y - y) * scale)


def round_half_away_from_zero(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def cap_style_from_svg(linecap: str) -> int:
    # Shapely cap_style: 1 round, 2 flat, 3 square
    lc = (linecap or "").strip().lower()
    if lc in ("butt", "flat"):
        return 2
    if lc in ("square",):
        return 3
    return 1


def join_style_from_svg(linejoin: str) -> int:
    # Shapely join_style: 1 round, 2 mitre, 3 bevel
    lj = (linejoin or "").strip().lower()
    if lj in ("miter", "mitre"):
        return 2
    if lj in ("bevel",):
        return 3
    return 1


def nudge_consecutive_duplicates(ring: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(ring) < 3:
        return ring
    out: List[Tuple[int, int]] = []
    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i, (x, y) in enumerate(ring):
        if out and (x, y) == out[-1]:
            dx, dy = offsets[i % 4]
            x2, y2 = x + dx, y + dy
            if (x2, y2) == out[-1]:
                x2, y2 = x + 2 * dx, y + 2 * dy
            x, y = x2, y2
        out.append((x, y))
    if out[-1] == out[0]:
        x, y = out[-1]
        out[-1] = (x + 1, y)
    return out


# -----------------------------
# SVG path parsing (M, L, A only)
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
    x1: float, y1: float, x2: float, y2: float, r: float, large: int, sweep: int
) -> Tuple[float, float, float, float]:
    dx2 = (x1 - x2) / 2.0
    dy2 = (y1 - y2) / 2.0
    x1p, y1p = dx2, dy2

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
    x1: float, y1: float, x2: float, y2: float, r: float, large: int, sweep: int
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
            x = parse_float(toks[i]); y = parse_float(toks[i + 1]); i += 2
            cur = (x, y)
            pts.append(cur)

        elif cmd == "L":
            if cur is None:
                raise ValueError("L without current point")
            x = parse_float(toks[i]); y = parse_float(toks[i + 1]); i += 2
            cur = (x, y)
            pts.append(cur)

        elif cmd == "A":
            if cur is None:
                raise ValueError("A without current point")
            rx = parse_float(toks[i]); ry = parse_float(toks[i + 1]); i += 2
            _rot = parse_float(toks[i]); i += 1
            large = int(float(toks[i])); sweep = int(float(toks[i + 1])); i += 2
            x2 = parse_float(toks[i]); y2 = parse_float(toks[i + 1]); i += 2
            if abs(rx - ry) > 1e-6:
                raise ValueError("Only circular arcs (rx==ry) supported.")
            pts.extend(sample_arc(cur[0], cur[1], x2, y2, rx, large, sweep))
            cur = (x2, y2)

        else:
            raise ValueError(f"Unsupported SVG path command: {cmd}")

    closed = False
    if len(pts) >= 3 and dist(pts[0], pts[-1]) < 1e-6:
        closed = True
        pts[-1] = pts[0]
    return CenterlinePath(pts=pts, closed=closed)


# -----------------------------
# Landmark-anchored ring resampling
# -----------------------------
def resample_linestring_closed(ls: LineString, n: int, start_d: float = 0.0) -> List[Tuple[float, float]]:
    L = ls.length
    if L <= 1e-9:
        p = ls.coords[0]
        return [(float(p[0]), float(p[1])) for _ in range(n)]
    start_d = start_d % L
    step = L / n
    out: List[Tuple[float, float]] = []
    for k in range(n):
        d = (start_d + k * step) % L
        p = ls.interpolate(d)
        out.append((p.x, p.y))
    return out


def _landmark_distances(ls: LineString, samples: int) -> List[float]:
    L = ls.length
    if L <= 1e-9:
        return [0.0]
    dense = resample_linestring_closed(ls, samples, start_d=0.0)
    xs = [p[0] for p in dense]
    ys = [p[1] for p in dense]

    idx_minx = min(range(len(dense)), key=lambda i: xs[i])
    idx_maxx = max(range(len(dense)), key=lambda i: xs[i])
    idx_miny = min(range(len(dense)), key=lambda i: ys[i])
    idx_maxy = max(range(len(dense)), key=lambda i: ys[i])

    cand = [dense[idx_minx], dense[idx_maxx], dense[idx_miny], dense[idx_maxy]]
    ds = [ls.project(Point(x, y)) for (x, y) in cand]

    ds2: List[float] = []
    eps = max(1e-6, L * 1e-6)
    for d in sorted(ds):
        if not ds2 or abs(d - ds2[-1]) > eps:
            ds2.append(d)
    return ds2 if ds2 else [0.0]


def resample_ring_with_landmarks(
    ls: LineString,
    n: int,
    seed_svg: Tuple[float, float],
    landmark_samples: int,
) -> List[Tuple[float, float]]:
    L = ls.length
    if L <= 1e-9:
        p = ls.coords[0]
        return [(float(p[0]), float(p[1])) for _ in range(n)]

    seed_d = ls.project(Point(seed_svg[0], seed_svg[1])) % L
    lm = _landmark_distances(ls, landmark_samples)

    rel = sorted(((d - seed_d) % L) for d in lm)
    if not rel or rel[0] > 1e-9:
        rel = [0.0] + rel
    if rel[-1] < L - 1e-9:
        rel = rel + [L]

    seg_lens = [rel[i + 1] - rel[i] for i in range(len(rel) - 1)]
    total = sum(seg_lens) if seg_lens else L

    seg_counts = [max(1, int(round(n * (sl / total)))) for sl in seg_lens]
    while sum(seg_counts) > n:
        i = max(range(len(seg_counts)), key=lambda k: seg_counts[k])
        if seg_counts[i] > 1:
            seg_counts[i] -= 1
        else:
            break
    while sum(seg_counts) < n:
        i = max(range(len(seg_counts)), key=lambda k: seg_lens[k])
        seg_counts[i] += 1

    out: List[Tuple[float, float]] = []
    for i, cnt in enumerate(seg_counts):
        a = rel[i]
        b = rel[i + 1]
        seg_len = max(1e-12, b - a)
        for k in range(cnt):
            t = k / cnt
            d = seed_d + a + t * seg_len
            p = ls.interpolate(d % L)
            out.append((p.x, p.y))

    if len(out) > n:
        out = out[:n]
    while len(out) < n:
        out.append(out[-1])

    return out


# -----------------------------
# Buffering + merging
# -----------------------------
def buffer_centerline(
    center: CenterlinePath,
    radius: float,
    cap_style: int,
    join_style: int,
) -> Polygon:
    if center.closed:
        ring = LinearRing(center.pts)
        geom = ring.buffer(radius, cap_style=1, join_style=join_style, resolution=16)
    else:
        ls = LineString(center.pts)
        geom = ls.buffer(radius, cap_style=cap_style, join_style=join_style, resolution=16)

    try:
        geom = geom.buffer(0)
    except Exception:
        pass

    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        # caller merges anyway; pick largest piece as polygon element
        # (still deterministic)
        polys = list(geom.geoms)
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys[0]
    raise RuntimeError(f"Unexpected buffer result: {type(geom)}")


def ring_sort_key_simple(ring: LinearRing) -> Tuple[float, float, float]:
    coords = list(ring.coords)
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    a = abs(signed_area([(float(x), float(y)) for (x, y) in coords[: min(256, len(coords))]]))
    return (miny + maxy, minx + maxx, -a)


def geometry_to_contours_fixed(
    geom: Union[Polygon, MultiPolygon],
    exterior_pts: int,
    hole_pts: int,
    scale: float,
    seed_svg: Tuple[float, float],
    landmark_samples: int,
    do_nudge: bool = False,
) -> List[List[Tuple[int, int]]]:
    """
    Convert a Polygon or MultiPolygon into a deterministic list of contours.

    Determinism:
    - If MultiPolygon: sort polygons by area desc
    - For each polygon: exterior first, then holes sorted
    """
    contours: List[List[Tuple[int, int]]] = []

    polys: List[Polygon]
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        polys.sort(key=lambda p: p.area, reverse=True)
    else:
        raise RuntimeError(f"Unexpected geometry: {type(geom)}")

    for poly in polys:
        if poly.is_empty:
            continue

        # exterior
        ext_ring = LinearRing(poly.exterior.coords)
        ext_ls = LineString(ext_ring.coords)
        ext_svg = resample_ring_with_landmarks(ext_ls, exterior_pts, seed_svg, landmark_samples)
        ext_font_f = [svg_to_font_xy(x, y, scale) for (x, y) in ext_svg]
        ext_font_f = ensure_direction_f(ext_font_f, clockwise=True)
        ext_int = [(round_half_away_from_zero(x), round_half_away_from_zero(y)) for (x, y) in ext_font_f]
        if do_nudge:
            ext_int = nudge_consecutive_duplicates(ext_int)
        contours.append(ext_int)

        # holes
        holes = [LinearRing(r.coords) for r in poly.interiors]
        holes.sort(key=ring_sort_key_simple)

        for hr in holes:
            hole_ls = LineString(hr.coords)
            hole_svg = resample_ring_with_landmarks(hole_ls, hole_pts, seed_svg, landmark_samples)
            hole_font_f = [svg_to_font_xy(x, y, scale) for (x, y) in hole_svg]
            hole_font_f = ensure_direction_f(hole_font_f, clockwise=False)
            hole_int = [(round_half_away_from_zero(x), round_half_away_from_zero(y)) for (x, y) in hole_font_f]
            if do_nudge:
                hole_int = nudge_consecutive_duplicates(hole_int)
            contours.append(hole_int)

    return contours


# -----------------------------
# TT glyph construction (MERGED)
# -----------------------------
def build_glyph_tt(
    spec: GlyphSpec,
    stroke_width_svg: float,
    scale: float,
    exterior_pts: int,
    hole_pts: int,
    dot_pts: int,
    cap_style: int,
    join_style: int,
    landmark_samples: int,
) -> Tuple[object, int]:
    pen = TTGlyphPen(None)
    radius = stroke_width_svg / 2.0

    # Buffer all components
    shapes: List[Polygon] = []

    # Seed for resampling: first point of first centerline if available,
    # else first dot.
    seed_svg: Tuple[float, float]
    if spec.centerlines:
        seed_svg = (spec.centerlines[0].pts[0][0], spec.centerlines[0].pts[0][1])
    elif spec.dots:
        seed_svg = (spec.dots[0].cx + (spec.dots[0].r + radius), spec.dots[0].cy)
    else:
        seed_svg = (0.0, 0.0)

    for cl in spec.centerlines:
        shapes.append(buffer_centerline(cl, radius, cap_style=cap_style, join_style=join_style))

    # Dots: effective radius = dot_r + stroke/2 (filled+stroked circle in SVG)
    for d in spec.dots:
        rr = d.r + radius
        shapes.append(Point(d.cx, d.cy).buffer(rr, resolution=32))

    if not shapes:
        glyph = pen.glyph()
        adv_w = int(round(spec.adv_w_svg * scale))
        return glyph, adv_w

    # MERGE ALL SHAPES FIRST
    merged = unary_union(shapes)
    try:
        merged = merged.buffer(0)
    except Exception:
        pass

    # Convert merged geometry into contours (use dot_pts only for dot-only glyphs)
    use_ext_pts = exterior_pts
    use_dot_mode = (len(spec.centerlines) == 0 and len(spec.dots) > 0)
    if use_dot_mode:
        use_ext_pts = dot_pts

    contours = geometry_to_contours_fixed(
        geom=merged,
        exterior_pts=use_ext_pts,
        hole_pts=hole_pts,
        scale=scale,
        seed_svg=seed_svg,
        landmark_samples=landmark_samples,
        do_nudge=use_dot_mode,
    )

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
# SVG loading (read stroke-linecap/linejoin)
# -----------------------------
def _find_main_stroke_style(root: ET.Element) -> Tuple[str, str]:
    best_linecap = "round"
    best_linejoin = "round"
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("g", "path"):
            continue
        stroke = (el.attrib.get("stroke") or "").strip().lower()
        sw = (el.attrib.get("stroke-width") or "").strip()
        if stroke in ("#000", "#000000", "black") and sw:
            lc = (el.attrib.get("stroke-linecap") or "").strip().lower()
            lj = (el.attrib.get("stroke-linejoin") or "").strip().lower()
            if lc:
                best_linecap = lc
            if lj:
                best_linejoin = lj
            return best_linecap, best_linejoin
    return best_linecap, best_linejoin


def parse_svg_glyph(svg_path: Path) -> GlyphSpec:
    m = SVG_FILE_RE.match(svg_path.name)
    if not m:
        raise ValueError(f"Not a glyph SVG: {svg_path}")
    cp = int(m.group(1), 16)

    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = (root.attrib.get("viewBox") or "").strip()
    if not vb:
        raise ValueError(f"Missing viewBox in {svg_path}")
    parts = vb.split()
    if len(parts) != 4:
        raise ValueError(f"Unexpected viewBox in {svg_path}: {vb!r}")
    adv_w = float(parts[2])
    h = float(parts[3])
    if abs(h - SVG_VIEW_H) > 1e-3:
        print(f"Warning: {svg_path.name} viewBox height={h}, expected {SVG_VIEW_H}")

    linecap, linejoin = _find_main_stroke_style(root)

    centerlines: List[CenterlinePath] = []
    for el in root.iter():
        if el.tag.endswith("path"):
            d = (el.attrib.get("d") or "").strip()
            if d:
                centerlines.append(path_d_to_centerline(d))

    dots: List[DotSpec] = []
    for el in root.iter():
        if el.tag.endswith("circle"):
            try:
                r = float(el.attrib.get("r", "0"))
                if r <= 0 or r > 2.0:
                    continue
                cx = float(el.attrib["cx"])
                cy = float(el.attrib["cy"])
                dots.append(DotSpec(cx=cx, cy=cy, r=r))
            except Exception:
                continue

    return GlyphSpec(
        codepoint=cp,
        glyph_name=glyph_name_from_cp(cp),
        adv_w_svg=adv_w,
        centerlines=centerlines,
        dots=dots,
        linecap=linecap,
        linejoin=linejoin,
    )


def load_glyphs_from_src(src_dir: Path) -> List[GlyphSpec]:
    svgs = sorted([p for p in src_dir.iterdir() if p.is_file() and SVG_FILE_RE.match(p.name)])
    if not svgs:
        raise FileNotFoundError(f"No glyph SVGs found in {src_dir} (expected src/character-uXXXX.svg).")
    return [parse_svg_glyph(p) for p in svgs]


# -----------------------------
# Master TTF builder
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
    dot_pts: int,
    cap_style: int,
    join_style: int,
    landmark_samples: int,
) -> None:
    glyph_order = [".notdef", "space"] + [s.glyph_name for s in specs]
    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder(glyph_order)

    glyf: Dict[str, object] = {}
    hmtx: Dict[str, Tuple[int, int]] = {}

    # .notdef rectangle
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
            spec=s,
            stroke_width_svg=stroke_width_svg,
            scale=scale,
            exterior_pts=exterior_pts,
            hole_pts=hole_pts,
            dot_pts=dot_pts,
            cap_style=cap_style,
            join_style=join_style,
            landmark_samples=landmark_samples,
        )
        glyf[s.glyph_name] = g
        hmtx[s.glyph_name] = (aw, 0)

    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(hmtx)

    cmap: Dict[int, str] = {0x20: "space"}
    for s in specs:
        cmap[s.codepoint] = s.glyph_name
    if 0x201D in cmap and 0x0022 not in cmap:
        cmap[0x0022] = cmap[0x201D]
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
# Variable font builder (multi-master)
# -----------------------------
def build_variable_font(
    sources: List[Tuple[int, Path]],  # (wght, ttf_path)
    out_var_path: Path,
) -> None:
    from fontTools.designspaceLib import DesignSpaceDocument, AxisDescriptor, SourceDescriptor

    doc = DesignSpaceDocument()

    axis = AxisDescriptor()
    axis.tag = "wght"
    axis.name = "Weight"
    axis.minimum = 100
    axis.default = 400
    axis.maximum = 900
    doc.addAxis(axis)

    for wght, p in sources:
        sd = SourceDescriptor()
        sd.path = str(p)
        sd.name = f"master_{wght}"
        sd.location = {"Weight": wght}
        doc.addSource(sd)

    out_var_path.parent.mkdir(parents=True, exist_ok=True)
    varfont, _, _ = var_build(doc)
    varfont.save(out_var_path)


# -----------------------------
# Stroke mapping
# -----------------------------
def stroke_for_wght(w: int, stroke_min: float, stroke_reg: float, stroke_max: float) -> float:
    if w <= 400:
        t = (w - 100) / (400 - 100)
        return stroke_min + t * (stroke_reg - stroke_min)
    t = (w - 400) / (900 - 400)
    return stroke_reg + t * (stroke_max - stroke_reg)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build/fonts", help="Intermediate build folder (default: build/fonts)")
    ap.add_argument("--src-dir", default="src", help="Input SVG folder (default: src)")
    ap.add_argument("--out-dir", default="dist/fonts", help="Output folder (default: dist/fonts)")
    ap.add_argument("--family", default="Distribution", help="Font family name (default: Distribution)")
    ap.add_argument("--basename", default="distribution", help="Output base filename (default: distribution)")
    ap.add_argument("--upm", type=int, default=DEFAULT_UPM)

    ap.add_argument("--stroke-min", type=float, default=2.5)
    ap.add_argument("--stroke-reg", type=float, default=9.0)
    ap.add_argument("--stroke-max", type=float, default=26.0)

    ap.add_argument("--masters", type=int, default=9,
                    help="Number of masters across 100..900. Default 9 => 100,200,...,900")

    ap.add_argument("--exterior-pts", type=int, default=DEFAULT_EXTERIOR_PTS)
    ap.add_argument("--hole-pts", type=int, default=DEFAULT_HOLE_PTS)
    ap.add_argument("--dot-pts", type=int, default=DEFAULT_DOT_PTS)
    ap.add_argument("--landmark-samples", type=int, default=DEFAULT_LANDMARK_SAMPLES)

    ap.add_argument("--linecap", default="", help="Override: round|butt|square (default: read from SVG)")
    ap.add_argument("--linejoin", default="", help="Override: round|miter|bevel (default: read from SVG)")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    build_tmp = Path(args.build_dir)
    build_tmp.mkdir(parents=True, exist_ok=True)

    specs = load_glyphs_from_src(src_dir)

    base_cap = (args.linecap.strip().lower() or specs[0].linecap.strip().lower() or "round")
    base_join = (args.linejoin.strip().lower() or specs[0].linejoin.strip().lower() or "round")
    cap_style = cap_style_from_svg(base_cap)
    join_style = join_style_from_svg(base_join)

    scale = args.upm / SVG_VIEW_H
    ascent = int(round((SVG_BASELINE_Y - 0.0) * scale))
    descent = -int(round((SVG_VIEW_H - SVG_BASELINE_Y) * scale))

    m = max(3, int(args.masters))
    ws = sorted({int(round(100 + i * (800 / (m - 1)))) for i in range(m)} | {100, 400, 900})
    ws = [min(900, max(100, w)) for w in ws]
    ws = sorted(set(ws))
    if 400 not in ws:
        ws.append(400)
        ws = sorted(ws)

    print(f"Loading {len(specs)} glyph(s) from {src_dir}…")
    print("Style:")
    print(f" - stroke-linecap: {base_cap} (cap_style={cap_style})")
    print(f" - stroke-linejoin: {base_join} (join_style={join_style})")
    print(f"Masters: {ws}")

    sources: List[Tuple[int, Path]] = []

    for w in ws:
        sw = stroke_for_wght(w, args.stroke_min, args.stroke_reg, args.stroke_max)
        master_path = build_tmp / f"{args.basename}-master-{w}.ttf"
        print(f" - building {master_path.name} (wght={w}, stroke={sw:.3f})")

        build_master_ttf(
            specs=specs,
            out_path=master_path,
            family=args.family,
            style=f"Master{w}",
            upm=args.upm,
            ascent=ascent,
            descent=descent,
            stroke_width_svg=sw,
            scale=scale,
            exterior_pts=args.exterior_pts,
            hole_pts=args.hole_pts,
            dot_pts=args.dot_pts,
            cap_style=cap_style,
            join_style=join_style,
            landmark_samples=args.landmark_samples,
        )
        sources.append((w, master_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_var_ttf = out_dir / f"{args.basename}.ttf"
    out_var_woff = out_dir / f"{args.basename}.woff"
    out_var_woff2 = out_dir / f"{args.basename}.woff2"

    print(f"Building variable font: {out_var_ttf}")
    build_variable_font(sources=sources, out_var_path=out_var_ttf)

    # WOFF
    try:
        tt = TTFont(out_var_ttf)
        tt.flavor = "woff"
        tt.save(out_var_woff)
        print(f"Wrote {out_var_woff}")
    except Exception as e:
        print("Could not write WOFF.")
        print(f"Error: {e}")

    # WOFF2
    try:
        tt = TTFont(out_var_ttf)
        tt.flavor = "woff2"
        tt.save(out_var_woff2)
        print(f"Wrote {out_var_woff2}")
    except Exception as e:
        print("Could not write WOFF2 (try: pip install brotli).")
        print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
