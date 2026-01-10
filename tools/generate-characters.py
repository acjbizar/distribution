#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate-characters-gridbased-v3.py

Flower-of-life (hex packed) grid-based monoline glyph generator.

Key properties
- Grid circle radius R, spacing: dx=R, dy=R*sqrt(3)/2, odd rows offset by dx/2.
- Glyph curves are built from arcs of those grid circles.
- "Smooth out" connections: arcs are emitted as cubic Beziers, with endpoint handle easing.
  (This softens curvature near joins and makes circle->line transitions look less "mechanical".)
- Outputs ONLY: src/character-u{codepoint}.svg
- Generates ONLY the requested characters (deduped).

This script intentionally keeps the glyph logic “instructional”:
each glyph is built from a small set of circle modules + lines between grid-anchored points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# -----------------------------
# User set
# -----------------------------
REQUESTED = (
    "AbCdEfghIJ"
    "kLmnopqrStUvwxyZ.„”?"
    "0123456789"
    "AcGHKMNX"
)

# -----------------------------
# Output / styling
# -----------------------------
OUT_DIR = Path("src")
VIEW_W = 240.0
VIEW_H = 240.0

# Grid circle radius (tuned so one “unit circle” looks close to your screenshot scale)
R = 40.0

# Stroke (monoline)
STROKE = 10.0

# Easing for arc endpoints (0..1):
# 1.0 = true circle arc; lower = “flatter” near endpoints, nicer transitions to lines
ARC_EASE_ENDS = 0.72

# How many degrees max per cubic arc segment
ARC_SEG_MAX_DEG = 90.0

# -----------------------------
# Grid math
# -----------------------------
@dataclass(frozen=True)
class HexGrid:
    R: float
    origin_x: float
    origin_y: float

    @property
    def dx(self) -> float:
        return self.R

    @property
    def dy(self) -> float:
        return self.R * math.sqrt(3) / 2.0

    def center(self, col: float, row: float) -> Tuple[float, float]:
        """
        col,row can be int or half-int. Only row parity matters for offset when row is integer.
        If row is non-integer, no parity offset is applied (we use it only for real grid rows).
        """
        # if row is effectively integer, apply odd-row offset
        if abs(row - round(row)) < 1e-9:
            r_int = int(round(row))
            x = col * self.dx + (self.dx / 2.0 if (r_int & 1) else 0.0)
            y = row * self.dy
        else:
            # for non-integer rows, treat as “continuous” (no parity offset)
            x = col * self.dx
            y = row * self.dy
        return (self.origin_x + x, self.origin_y + y)

# -----------------------------
# Geometry helpers
# -----------------------------
@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

def _deg(a: float) -> float:
    return a * math.pi / 180.0

def circle_point(c: Circle, ang_deg: float) -> Tuple[float, float]:
    """
    SVG coords: +x right, +y down.
    Angle 0° = rightmost point, 90° = down, 180° = left, 270° = up.
    """
    a = _deg(ang_deg)
    return (c.cx + c.r * math.cos(a), c.cy + c.r * math.sin(a))

def circle_circle_intersections(c1: Circle, c2: Circle) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Returns intersection points (pA, pB) or None if no proper intersections.
    """
    dx = c2.cx - c1.cx
    dy = c2.cy - c1.cy
    d = math.hypot(dx, dy)
    if d <= 1e-9:
        return None
    if d > c1.r + c2.r or d < abs(c1.r - c2.r):
        return None

    # distance from c1 to the line between intersections along centerline
    a = (c1.r * c1.r - c2.r * c2.r + d * d) / (2.0 * d)
    h_sq = c1.r * c1.r - a * a
    if h_sq < 0:
        h_sq = 0
    h = math.sqrt(h_sq)

    xm = c1.cx + a * dx / d
    ym = c1.cy + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return (p1, p2)

def left_right_of_two_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (p1, p2) if p1[0] <= p2[0] else (p2, p1)

def fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")

# -----------------------------
# Path builder (cubic-only)
# -----------------------------
class Path:
    def __init__(self) -> None:
        self.cmds: List[str] = []
        self._cur: Optional[Tuple[float, float]] = None

    def M(self, p: Tuple[float, float]) -> "Path":
        self._cur = p
        self.cmds.append(f"M{fmt(p[0])},{fmt(p[1])}")
        return self

    def L(self, p: Tuple[float, float]) -> "Path":
        self._cur = p
        self.cmds.append(f"L{fmt(p[0])},{fmt(p[1])}")
        return self

    def C(self, c1: Tuple[float, float], c2: Tuple[float, float], p: Tuple[float, float]) -> "Path":
        self._cur = p
        self.cmds.append(f"C{fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])} {fmt(p[0])},{fmt(p[1])}")
        return self

    def Z(self) -> "Path":
        self.cmds.append("Z")
        return self

    def d(self) -> str:
        return " ".join(self.cmds)

def arc_to_cubics(
    c: Circle,
    a0: float,
    a1: float,
    ease_ends: float = 1.0,
    max_seg_deg: float = 90.0,
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
    Returns list of cubic segments approximating a circle arc from a0 to a1 (degrees),
    where each segment is (ctrl1, ctrl2, end).

    We split into <= max_seg_deg segments.
    Control handles are the standard circle-approximation handles, but with "easing"
    applied near the very start/end segments to flatten curvature slightly.
    """
    # normalize direction: we will step with sign of delta
    delta = a1 - a0
    if abs(delta) < 1e-9:
        return []

    # choose step count
    segs = max(1, int(math.ceil(abs(delta) / max_seg_deg)))
    step = delta / segs

    cubics: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
    ang = a0

    for i in range(segs):
        ang0 = ang
        ang1 = ang + step
        ang = ang1

        # standard kappa for an arc segment
        theta = _deg(ang1 - ang0)
        k = 4.0 / 3.0 * math.tan(abs(theta) / 4.0) * c.r

        # easing: scale handles on first & last segment (or both if only one)
        if segs == 1:
            s = ease_ends
        elif i == 0 or i == segs - 1:
            s = ease_ends
        else:
            s = 1.0

        k *= s

        p0 = circle_point(c, ang0)
        p1 = circle_point(c, ang1)

        # tangent directions (derivative) at endpoints
        # Circle param: (cos, sin); derivative: (-sin, cos) in x/y, but y-down matches same formula.
        t0 = (-math.sin(_deg(ang0)), math.cos(_deg(ang0)))
        t1 = (-math.sin(_deg(ang1)), math.cos(_deg(ang1)))

        c1 = (p0[0] + t0[0] * k, p0[1] + t0[1] * k)
        c2 = (p1[0] - t1[0] * k, p1[1] - t1[1] * k)

        cubics.append((c1, c2, p1))

    return cubics

def add_arc(path: Path, c: Circle, a0: float, a1: float, ease_ends: float) -> None:
    cubics = arc_to_cubics(c, a0, a1, ease_ends=ease_ends, max_seg_deg=ARC_SEG_MAX_DEG)
    for (c1, c2, p1) in cubics:
        path.C(c1, c2, p1)

# -----------------------------
# SVG doc
# -----------------------------
def svg_doc(paths: List[str], title: str) -> str:
    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(VIEW_W)} {fmt(VIEW_H)}" '
        f'width="{fmt(VIEW_W)}" height="{fmt(VIEW_H)}">'
    )
    parts.append(f"  <title>{title}</title>")
    parts.append(
        f'  <g fill="none" stroke="#000" stroke-width="{fmt(STROKE)}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    )
    for d in paths:
        parts.append(f'    <path d="{d}" />')
    parts.append("  </g>")
    parts.append("</svg>")
    parts.append("")
    return "\n".join(parts)

def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def uniq_chars(s: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ch in s:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out

# -----------------------------
# Glyph “modules”
# -----------------------------
def module_circle_full(c: Circle) -> str:
    p = Path()
    p0 = circle_point(c, 0.0)
    p.M(p0)
    add_arc(p, c, 0.0, 360.0, ease_ends=1.0)  # full circle: true
    return p.d()

def module_circle_open_right(c: Circle, gap_deg: float = 60.0) -> str:
    """
    Like your lowercase c: circle missing a chunk on the right.
    """
    p = Path()
    a0 = gap_deg / 2.0
    a1 = 360.0 - gap_deg / 2.0
    p.M(circle_point(c, a0))
    add_arc(p, c, a0, a1, ease_ends=ARC_EASE_ENDS)
    return p.d()

def module_double_lobe_closed_right(
    c_top: Circle,
    c_bot: Circle,
) -> str:
    """
    This matches your 'a' core: two overlapping circles, joined at LEFT intersection,
    and closed on the RIGHT by a vertical tangent stem at the circles' 0° points.
    """
    inter = circle_circle_intersections(c_top, c_bot)
    if inter is None:
        raise ValueError("Top/bottom circles do not intersect for double-lobe module.")
    pL, pR = left_right_of_two_points(inter[0], inter[1])  # left and right intersections
    # We join lobes at LEFT intersection (waist)
    waist = pL

    # Right tangent points (0° points) define the vertical closing stem
    rt_top = circle_point(c_top, 0.0)
    rt_bot = circle_point(c_bot, 0.0)

    # Build one closed path:
    # start at top-right tangent -> arc around top (CCW via left) to waist -> arc around bottom to bottom-right tangent -> line up to start
    p = Path().M(rt_top)

    # Top arc from 0° to the angle at waist (choose CCW through left side)
    # Determine waist angle on top circle
    ang_waist_top = math.degrees(math.atan2(waist[1] - c_top.cy, waist[0] - c_top.cx)) % 360.0
    # go CCW from 0 to ang_waist_top (if ang_waist_top < 0, etc). We want through left side, so ensure end > start by adding 360 if needed.
    end = ang_waist_top
    if end <= 0.0:
        end += 360.0
    add_arc(p, c_top, 0.0, end, ease_ends=ARC_EASE_ENDS)

    # Now bottom arc from waist angle on bottom to 0° (CCW through left/bottom)
    ang_waist_bot = math.degrees(math.atan2(waist[1] - c_bot.cy, waist[0] - c_bot.cx)) % 360.0
    # We are currently at waist. Go CCW from ang_waist_bot to 360 (or 0) to reach 0°.
    # Ensure we move forward CCW: end must be > start.
    start_b = ang_waist_bot
    end_b = 360.0
    add_arc(p, c_bot, start_b, end_b, ease_ends=ARC_EASE_ENDS)

    # Line up right stem back to top-right
    p.L(rt_top)
    p.Z()
    return p.d()

def module_double_lobe_open_right(
    c_top: Circle,
    c_bot: Circle,
    open_pull_deg: float = 35.0,
) -> str:
    """
    This matches your 'E' (Ɛ-like): same double-lobe waist on LEFT,
    but OPEN on right: endpoints are pulled off the 0° tangent to create openings.

    We draw:
      start near top circle (just past 0°), arc CCW to left waist,
      then arc CCW around bottom circle to just before 0°,
      and STOP (no closing stem).
    """
    inter = circle_circle_intersections(c_top, c_bot)
    if inter is None:
        raise ValueError("Top/bottom circles do not intersect for double-lobe module.")
    pL, _pR = left_right_of_two_points(inter[0], inter[1])
    waist = pL

    # pick start/end angles slightly off the rightmost tangent to create the open gaps
    start_ang = open_pull_deg
    end_ang = 360.0 - open_pull_deg

    # angles of waist on each circle
    ang_waist_top = math.degrees(math.atan2(waist[1] - c_top.cy, waist[0] - c_top.cx)) % 360.0
    ang_waist_bot = math.degrees(math.atan2(waist[1] - c_bot.cy, waist[0] - c_bot.cx)) % 360.0

    # path
    p = Path().M(circle_point(c_top, start_ang))

    # Top: go CCW from start_ang to waist (through left side). Ensure end > start by adding 360 if needed.
    end_top = ang_waist_top
    if end_top <= start_ang:
        end_top += 360.0
    add_arc(p, c_top, start_ang, end_top, ease_ends=ARC_EASE_ENDS)

    # Bottom: go CCW from waist angle to end_ang (near 360) (through bottom/left).
    # Ensure end > start: end_ang is near 360, so if start > end, add 360.
    start_b = ang_waist_bot
    end_b = end_ang
    if end_b <= start_b:
        end_b += 360.0
    add_arc(p, c_bot, start_b, end_b, ease_ends=ARC_EASE_ENDS)

    return p.d()

# -----------------------------
# Glyph placement (pick consistent grid anchors)
# -----------------------------
def build_grid() -> HexGrid:
    """
    We place a small grid patch into the viewBox.
    Choose an origin so that the “main” bowl circles for lowercase sit nicely centered.
    """
    # Center-ish placement: choose a grid origin near top-left with some margins.
    # (All points remain mathematically on-grid; origin just translates the whole lattice.)
    margin_x = 40.0
    margin_y = 25.0
    return HexGrid(R=R, origin_x=margin_x, origin_y=margin_y)

def circle_at(grid: HexGrid, col: float, row: float, r: float = None) -> Circle:
    if r is None:
        r = grid.R
    cx, cy = grid.center(col, row)
    return Circle(cx=cx, cy=cy, r=r)

# -----------------------------
# Glyph definitions (grid-anchored)
# -----------------------------
def glyph_a(grid: HexGrid) -> List[str]:
    # Two overlapping circles in a vertical stack (rows separated by 2)
    c_top = circle_at(grid, col=2.0, row=2.0)
    c_bot = circle_at(grid, col=2.0, row=4.0)
    return [module_double_lobe_closed_right(c_top, c_bot)]

def glyph_E(grid: HexGrid) -> List[str]:
    # Same double-lobe, open right (Ɛ-like)
    c_top = circle_at(grid, col=2.0, row=2.0)
    c_bot = circle_at(grid, col=2.0, row=4.0)
    return [module_double_lobe_open_right(c_top, c_bot, open_pull_deg=35.0)]

def glyph_c(grid: HexGrid) -> List[str]:
    c = circle_at(grid, col=2.0, row=3.0)
    return [module_circle_open_right(c, gap_deg=70.0)]

def glyph_o(grid: HexGrid) -> List[str]:
    c = circle_at(grid, col=2.0, row=3.0)
    return [module_circle_full(c)]

def glyph_i(grid: HexGrid) -> List[str]:
    # In your sheet, i (and I) are basically straight stems; dot is not shown for I.
    # We'll do lowercase i as a stem + dot; uppercase I as stem only.
    # Lowercase i stem: from row 2.2 to row 4.2 on same x
    x, _ = grid.center(2.0, 2.0)
    y1 = grid.center(2.0, 2.2)[1]
    y2 = grid.center(2.0, 4.2)[1]
    stem = Path().M((x, y1)).L((x, y2)).d()
    dot_c = Circle(cx=x, cy=grid.center(2.0, 1.2)[1], r=grid.R * 0.16)
    dot = module_circle_full(dot_c)
    return [stem, dot]

def glyph_I(grid: HexGrid) -> List[str]:
    x, _ = grid.center(2.0, 1.6)
    y1 = grid.center(2.0, 1.2)[1]
    y2 = grid.center(2.0, 4.8)[1]
    return [Path().M((x, y1)).L((x, y2)).d()]

def glyph_b(grid: HexGrid) -> List[str]:
    # Stem on left, bowl is a circle tangent to stem
    bowl = circle_at(grid, col=2.4, row=3.6)  # slightly right & lower
    stem_x = bowl.cx - bowl.r  # tangent at 180°
    y_top = grid.center(2.0, 1.2)[1]
    touch = circle_point(bowl, 180.0)
    p = Path().M((stem_x, y_top)).L(touch)
    # loop around bowl (full circle) from 180° back to 180°
    add_arc(p, bowl, 180.0, 540.0, ease_ends=1.0)
    return [p.d()]

def glyph_d(grid: HexGrid) -> List[str]:
    # Mirror of b: stem on right, bowl tangent
    bowl = circle_at(grid, col=1.6, row=3.6)
    stem_x = bowl.cx + bowl.r  # tangent at 0°
    y_top = grid.center(2.0, 1.2)[1]
    touch = circle_point(bowl, 0.0)
    p = Path().M((stem_x, y_top)).L(touch)
    add_arc(p, bowl, 0.0, 360.0, ease_ends=1.0)
    return [p.d()]

def glyph_h(grid: HexGrid) -> List[str]:
    # Stem + right arch (single circle arc) + short right stem
    stem_x, _ = grid.center(1.4, 1.2)
    y_top = grid.center(1.4, 1.2)[1]
    y_bot = grid.center(1.4, 4.8)[1]
    stem = Path().M((stem_x, y_top)).L((stem_x, y_bot)).d()

    arch_c = circle_at(grid, col=2.4, row=3.2)
    # connect from left tangent of arch to right-bottom area, then down a short stem
    left_touch = circle_point(arch_c, 180.0)
    right_touch = circle_point(arch_c, 0.0)
    p = Path().M(left_touch)
    add_arc(p, arch_c, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)  # top half-ish
    # right stem down
    p.L((right_touch[0], y_bot))
    return [stem, p.d()]

def glyph_f(grid: HexGrid) -> List[str]:
    # Tall stem + small top hook made from an arc of a circle
    x, _ = grid.center(2.0, 1.0)
    y_top = grid.center(2.0, 0.8)[1]
    y_bot = grid.center(2.0, 4.8)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()

    hook = circle_at(grid, col=1.7, row=1.6)
    p = Path().M(circle_point(hook, 300.0))
    add_arc(p, hook, 300.0, 80.0, ease_ends=ARC_EASE_ENDS)
    return [stem, p.d()]

def glyph_g(grid: HexGrid) -> List[str]:
    # Top full bowl + bottom open bowl (like your screenshot g)
    top = circle_at(grid, col=2.6, row=2.6)
    bot = circle_at(grid, col=2.6, row=4.4)
    paths = [module_circle_full(top)]
    # bottom: open on left more than right
    p = Path().M(circle_point(bot, 40.0))
    add_arc(p, bot, 40.0, 320.0, ease_ends=ARC_EASE_ENDS)
    # tail down from ~300° point
    tail_start = circle_point(bot, 300.0)
    tail_end = (tail_start[0], grid.center(2.6, 6.2)[1])
    p2 = Path().M(tail_start).L(tail_end).d()
    paths.append(p.d())
    paths.append(p2)
    return paths

def glyph_J(grid: HexGrid) -> List[str]:
    # Stem + bottom hook
    x, _ = grid.center(2.8, 1.2)
    y_top = grid.center(2.8, 1.0)[1]
    y_mid = grid.center(2.8, 4.0)[1]
    stem = Path().M((x, y_top)).L((x, y_mid)).d()

    hook = circle_at(grid, col=2.0, row=4.4)
    p = Path().M((x, y_mid))
    # go down into hook arc to left and up a bit
    add_arc(p, hook, 350.0, 200.0, ease_ends=ARC_EASE_ENDS)
    return [stem, p.d()]

# --- The rest are reasonable grid-anchored approximations (all still “on-grid”) ---

def glyph_C(grid: HexGrid) -> List[str]:
    # Tall open capsule (your capital C)
    top = circle_at(grid, col=2.2, row=2.0)
    bot = circle_at(grid, col=2.2, row=4.6)
    # left stem at x = cx - r, connect left tangency points
    xL = top.cx - top.r
    p_topL = circle_point(top, 180.0)
    p_botL = circle_point(bot, 180.0)
    # start near top-right gap
    start = circle_point(top, 40.0)
    p = Path().M(start)
    add_arc(p, top, 40.0, 180.0, ease_ends=ARC_EASE_ENDS)
    p.L(p_botL)
    add_arc(p, bot, 180.0, 320.0, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def glyph_L(grid: HexGrid) -> List[str]:
    x, _ = grid.center(1.6, 1.2)
    y_top = grid.center(1.6, 1.2)[1]
    y_bot = grid.center(1.6, 4.8)[1]
    base = circle_at(grid, col=2.2, row=4.8)
    p = Path().M((x, y_top)).L((x, y_bot))
    # small curve at bottom into the right (quarter-ish)
    add_arc(p, base, 180.0, 270.0, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def glyph_k(grid: HexGrid) -> List[str]:
    x, _ = grid.center(1.4, 1.2)
    y_top = grid.center(1.4, 1.2)[1]
    y_bot = grid.center(1.4, 4.8)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()

    mid = grid.center(1.4, 3.0)
    up = grid.center(2.9, 2.0)
    dn = grid.center(2.7, 4.3)
    arm1 = Path().M(mid).L(up).d()
    arm2 = Path().M(mid).L(dn).d()
    return [stem, arm1, arm2]

def glyph_m(grid: HexGrid) -> List[str]:
    # stem + two arches
    x, _ = grid.center(1.2, 3.0)
    y_top = grid.center(1.2, 2.2)[1]
    y_bot = grid.center(1.2, 4.8)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()

    a1 = circle_at(grid, col=2.1, row=3.4)
    a2 = circle_at(grid, col=3.2, row=3.4)
    p1 = Path().M(circle_point(a1, 180.0))
    add_arc(p1, a1, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)
    p2 = Path().M(circle_point(a2, 180.0))
    add_arc(p2, a2, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)
    return [stem, p1.d(), p2.d()]

def glyph_n(grid: HexGrid) -> List[str]:
    x, _ = grid.center(1.4, 3.0)
    y_top = grid.center(1.4, 2.2)[1]
    y_bot = grid.center(1.4, 4.8)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()
    arch = circle_at(grid, col=2.4, row=3.4)
    p = Path().M(circle_point(arch, 180.0))
    add_arc(p, arch, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)
    return [stem, p.d()]

def glyph_p(grid: HexGrid) -> List[str]:
    stem_x, _ = grid.center(1.4, 2.2)
    y_top = grid.center(1.4, 2.0)[1]
    y_bot = grid.center(1.4, 6.0)[1]
    bowl = circle_at(grid, col=2.4, row=3.2)
    touch = circle_point(bowl, 180.0)
    p = Path().M((stem_x, y_top)).L((stem_x, y_bot)).M(touch)
    add_arc(p, bowl, 180.0, 540.0, ease_ends=1.0)
    return [p.d()]

def glyph_q(grid: HexGrid) -> List[str]:
    stem_x, _ = grid.center(3.0, 2.2)
    y_top = grid.center(3.0, 2.0)[1]
    y_bot = grid.center(3.0, 6.0)[1]
    bowl = circle_at(grid, col=2.0, row=3.2)
    touch = circle_point(bowl, 0.0)
    p = Path().M((stem_x, y_top)).L((stem_x, y_bot)).M(touch)
    add_arc(p, bowl, 0.0, 360.0, ease_ends=1.0)
    return [p.d()]

def glyph_r(grid: HexGrid) -> List[str]:
    x, _ = grid.center(1.4, 3.0)
    y_top = grid.center(1.4, 2.2)[1]
    y_bot = grid.center(1.4, 4.8)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()
    sh = circle_at(grid, col=2.0, row=3.0, r=grid.R * 0.75)
    p = Path().M(circle_point(sh, 180.0))
    add_arc(p, sh, 180.0, 330.0, ease_ends=ARC_EASE_ENDS)
    return [stem, p.d()]

def glyph_S(grid: HexGrid) -> List[str]:
    top = circle_at(grid, col=2.4, row=2.6)
    bot = circle_at(grid, col=2.0, row=4.3)
    p = Path().M(circle_point(top, 20.0))
    add_arc(p, top, 20.0, 220.0, ease_ends=ARC_EASE_ENDS)
    add_arc(p, bot, 40.0, 340.0, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def glyph_t(grid: HexGrid) -> List[str]:
    x, _ = grid.center(2.4, 2.0)
    y_top = grid.center(2.4, 1.6)[1]
    y_bot = grid.center(2.4, 4.9)[1]
    stem = Path().M((x, y_top)).L((x, y_bot)).d()
    bar_y = grid.center(2.4, 2.6)[1]
    xL = grid.center(1.6, 2.6)[0]
    xR = grid.center(3.2, 2.6)[0]
    bar = Path().M((xL, bar_y)).L((xR, bar_y)).d()
    return [stem, bar]

def glyph_U(grid: HexGrid) -> List[str]:
    left_x = grid.center(1.6, 2.0)[0]
    right_x = grid.center(3.2, 2.0)[0]
    y_top = grid.center(2.0, 1.6)[1]
    y_mid = grid.center(2.0, 4.6)[1]
    base = circle_at(grid, col=2.4, row=4.6)
    p = Path().M((left_x, y_top)).L((left_x, y_mid))
    add_arc(p, base, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)
    p.L((right_x, y_top))
    return [p.d()]

def glyph_v(grid: HexGrid) -> List[str]:
    a = grid.center(1.6, 2.2)
    b = grid.center(2.4, 4.8)
    c = grid.center(3.2, 2.2)
    return [Path().M(a).L(b).L(c).d()]

def glyph_w(grid: HexGrid) -> List[str]:
    a = grid.center(1.2, 2.2)
    b = grid.center(1.9, 4.8)
    c = grid.center(2.4, 3.4)
    d = grid.center(2.9, 4.8)
    e = grid.center(3.6, 2.2)
    return [Path().M(a).L(b).L(c).L(d).L(e).d()]

def glyph_x(grid: HexGrid) -> List[str]:
    a = grid.center(1.6, 2.2)
    b = grid.center(3.2, 4.8)
    c = grid.center(3.2, 2.2)
    d = grid.center(1.6, 4.8)
    return [Path().M(a).L(b).d(), Path().M(c).L(d).d()]

def glyph_y(grid: HexGrid) -> List[str]:
    a = grid.center(1.6, 2.2)
    b = grid.center(2.4, 4.0)
    c = grid.center(3.2, 2.2)
    tail = grid.center(2.4, 6.0)
    return [Path().M(a).L(b).L(c).M(b).L(tail).d()]

def glyph_Z(grid: HexGrid) -> List[str]:
    # Rounded-ish Z: top/bottom bars + diagonal
    top_y = grid.center(2.4, 2.0)[1]
    bot_y = grid.center(2.4, 4.8)[1]
    xL = grid.center(1.4, 2.0)[0]
    xR = grid.center(3.4, 2.0)[0]
    p1 = Path().M((xL, top_y)).L((xR, top_y)).d()
    p2 = Path().M((xR, top_y)).L((xL, bot_y)).d()
    p3 = Path().M((xL, bot_y)).L((xR, bot_y)).d()
    return [p1, p2, p3]

def glyph_dot(grid: HexGrid) -> List[str]:
    cx, cy = grid.center(2.4, 5.8)
    c = Circle(cx=cx, cy=cy, r=grid.R * 0.14)
    return [module_circle_full(c)]

def glyph_qmark(grid: HexGrid) -> List[str]:
    # Hook (open circle arc) + dot
    hook = circle_at(grid, col=2.4, row=3.0)
    p = Path().M(circle_point(hook, 230.0))
    add_arc(p, hook, 230.0, 20.0, ease_ends=ARC_EASE_ENDS)
    # short stem down
    end = circle_point(hook, 20.0)
    mid = (end[0], grid.center(2.4, 4.6)[1])
    stem = Path().M(end).L(mid).d()
    dot = glyph_dot(grid)[0]
    return [p.d(), stem, dot]

def glyph_low_quote(grid: HexGrid) -> List[str]:
    # „ (U+201E): two small comma-ish arcs near bottom
    c1 = circle_at(grid, col=2.1, row=5.7, r=grid.R * 0.22)
    c2 = circle_at(grid, col=2.6, row=5.7, r=grid.R * 0.22)
    p1 = Path().M(circle_point(c1, 300.0)); add_arc(p1, c1, 300.0, 140.0, ease_ends=ARC_EASE_ENDS)
    p2 = Path().M(circle_point(c2, 300.0)); add_arc(p2, c2, 300.0, 140.0, ease_ends=ARC_EASE_ENDS)
    return [p1.d(), p2.d()]

def glyph_high_quote(grid: HexGrid) -> List[str]:
    # ” (U+201D): two small comma-ish arcs near top
    c1 = circle_at(grid, col=2.1, row=1.2, r=grid.R * 0.22)
    c2 = circle_at(grid, col=2.6, row=1.2, r=grid.R * 0.22)
    p1 = Path().M(circle_point(c1, 220.0)); add_arc(p1, c1, 220.0, 20.0, ease_ends=ARC_EASE_ENDS)
    p2 = Path().M(circle_point(c2, 220.0)); add_arc(p2, c2, 220.0, 20.0, ease_ends=ARC_EASE_ENDS)
    return [p1.d(), p2.d()]

# Digits (simple, still on-grid)
def digit_0(grid: HexGrid) -> List[str]: return glyph_o(grid)
def digit_1(grid: HexGrid) -> List[str]:
    x, _ = grid.center(2.4, 2.0)
    y1 = grid.center(2.4, 2.0)[1]
    y2 = grid.center(2.4, 5.0)[1]
    return [Path().M((x, y1)).L((x, y2)).d()]

def digit_8(grid: HexGrid) -> List[str]:
    top = circle_at(grid, col=2.4, row=2.6)
    bot = circle_at(grid, col=2.4, row=4.4)
    return [module_circle_full(top), module_circle_full(bot)]

def digit_3(grid: HexGrid) -> List[str]:
    # Like 'E' but joined on RIGHT instead of left: mirror by swapping join side.
    c_top = circle_at(grid, col=2.4, row=2.6)
    c_bot = circle_at(grid, col=2.4, row=4.4)
    inter = circle_circle_intersections(c_top, c_bot)
    if inter is None:
        return [module_circle_open_right(c_top, 70.0)]
    pL, pR = left_right_of_two_points(inter[0], inter[1])
    waist = pR  # right join
    # open on left: start/end pulled off 180°
    open_pull = 35.0
    start_ang = 180.0 + open_pull
    end_ang = 180.0 - open_pull
    angw_top = math.degrees(math.atan2(waist[1] - c_top.cy, waist[0] - c_top.cx)) % 360.0
    angw_bot = math.degrees(math.atan2(waist[1] - c_bot.cy, waist[0] - c_bot.cx)) % 360.0

    p = Path().M(circle_point(c_top, start_ang))
    # go CW-ish by going decreasing angles: use CCW with +360 tricks
    end_top = angw_top
    # ensure CCW progression from start to end through right side:
    if end_top <= start_ang:
        end_top += 360.0
    add_arc(p, c_top, start_ang, end_top, ease_ends=ARC_EASE_ENDS)

    start_b = angw_bot
    end_b = end_ang
    if end_b <= start_b:
        end_b += 360.0
    add_arc(p, c_bot, start_b, end_b, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def digit_2(grid: HexGrid) -> List[str]:
    top = circle_at(grid, col=2.4, row=2.6)
    bot = circle_at(grid, col=2.4, row=4.6)
    p = Path().M(circle_point(top, 210.0))
    add_arc(p, top, 210.0, 10.0, ease_ends=ARC_EASE_ENDS)
    # diagonal down-left to bottom bar
    p.L(grid.center(1.6, 5.0))
    p.L(grid.center(3.4, 5.0))
    return [p.d()]

def digit_4(grid: HexGrid) -> List[str]:
    a = grid.center(3.2, 2.0)
    b = grid.center(1.8, 4.0)
    c = grid.center(3.2, 4.0)
    d = grid.center(3.2, 5.0)
    return [Path().M(a).L(b).L(c).M(c).L(d).d()]

def digit_5(grid: HexGrid) -> List[str]:
    top_y = grid.center(2.4, 2.2)[1]
    xL = grid.center(1.6, 2.2)[0]
    xR = grid.center(3.2, 2.2)[0]
    mid = grid.center(1.8, 3.2)
    bot = circle_at(grid, col=2.4, row=4.4)
    p = Path().M((xR, top_y)).L((xL, top_y)).L(mid)
    p.M(circle_point(bot, 180.0))
    add_arc(p, bot, 180.0, 360.0, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def digit_6(grid: HexGrid) -> List[str]:
    c = circle_at(grid, col=2.4, row=4.2)
    p = Path().M(circle_point(c, 40.0))
    add_arc(p, c, 40.0, 400.0, ease_ends=ARC_EASE_ENDS)
    return [p.d()]

def digit_7(grid: HexGrid) -> List[str]:
    top_y = grid.center(2.4, 2.2)[1]
    xL = grid.center(1.6, 2.2)[0]
    xR = grid.center(3.4, 2.2)[0]
    end = grid.center(2.0, 5.0)
    return [Path().M((xL, top_y)).L((xR, top_y)).L(end).d()]

def digit_9(grid: HexGrid) -> List[str]:
    c = circle_at(grid, col=2.4, row=3.0)
    p = Path().M(circle_point(c, 220.0))
    add_arc(p, c, 220.0, 520.0, ease_ends=ARC_EASE_ENDS)
    # tail down
    t0 = circle_point(c, 40.0)
    t1 = grid.center(2.8, 5.4)
    return [p.d(), Path().M(t0).L(t1).d()]

# Uppercase H/G/K/M/N/X approximations (still grid anchored)
def glyph_H(grid: HexGrid) -> List[str]:
    xL = grid.center(1.6, 1.6)[0]
    xR = grid.center(3.2, 1.6)[0]
    yT = grid.center(2.0, 1.2)[1]
    yB = grid.center(2.0, 4.8)[1]
    yM = grid.center(2.0, 3.0)[1]
    return [
        Path().M((xL, yT)).L((xL, yB)).d(),
        Path().M((xR, yT)).L((xR, yB)).d(),
        Path().M((xL, yM)).L((xR, yM)).d(),
    ]

def glyph_G(grid: HexGrid) -> List[str]:
    c = circle_at(grid, col=2.4, row=3.0)
    arc = module_circle_open_right(c, gap_deg=70.0)
    # inner bar
    y = grid.center(2.4, 3.6)[1]
    x1 = grid.center(2.4, 3.6)[0]
    x2 = x1 + grid.R * 0.9
    bar = Path().M((x1, y)).L((x2, y)).d()
    return [arc, bar]

def glyph_K(grid: HexGrid) -> List[str]:
    x = grid.center(1.6, 1.2)[0]
    yT = grid.center(1.6, 1.2)[1]
    yB = grid.center(1.6, 4.8)[1]
    mid = grid.center(1.6, 3.0)
    up = grid.center(3.2, 1.8)
    dn = grid.center(3.2, 4.6)
    return [
        Path().M((x, yT)).L((x, yB)).d(),
        Path().M(mid).L(up).d(),
        Path().M(mid).L(dn).d(),
    ]

def glyph_M(grid: HexGrid) -> List[str]:
    xL = grid.center(1.4, 4.8)[0]
    xR = grid.center(3.6, 4.8)[0]
    yT = grid.center(2.0, 1.2)[1]
    yB = grid.center(2.0, 4.8)[1]
    apex = grid.center(2.5, 2.2)
    return [
        Path().M((xL, yB)).L((xL, yT)).L(apex).L((xR, yT)).L((xR, yB)).d()
    ]

def glyph_N(grid: HexGrid) -> List[str]:
    xL = grid.center(1.6, 1.2)[0]
    xR = grid.center(3.2, 1.2)[0]
    yT = grid.center(2.0, 1.2)[1]
    yB = grid.center(2.0, 4.8)[1]
    return [
        Path().M((xL, yB)).L((xL, yT)).L((xR, yB)).L((xR, yT)).d()
    ]

def glyph_X(grid: HexGrid) -> List[str]: return glyph_x(grid)

# -----------------------------
# Builder map for requested chars
# -----------------------------
BUILDERS: Dict[str, callable] = {
    # lower
    "a": glyph_a,
    "b": glyph_b,
    "c": glyph_c,
    "d": glyph_d,
    "f": glyph_f,
    "g": glyph_g,
    "h": glyph_h,
    "i": glyph_i,
    "k": glyph_k,
    "m": glyph_m,
    "n": glyph_n,
    "o": glyph_o,
    "p": glyph_p,
    "q": glyph_q,
    "r": glyph_r,
    "t": glyph_t,
    "v": glyph_v,
    "w": glyph_w,
    "x": glyph_x,
    "y": glyph_y,
    "z": glyph_Z,  # lowercase z not shown; your Z is uppercase. Still provides a Z-like.
    # upper (explicit keys)
    "A": glyph_a,   # in your sheet, A resembles the “double lobe” family; adjust if needed later
    "C": glyph_C,
    "E": glyph_E,
    "G": glyph_G,
    "H": glyph_H,
    "I": glyph_I,
    "J": glyph_J,
    "K": glyph_K,
    "L": glyph_L,
    "M": glyph_M,
    "N": glyph_N,
    "S": glyph_S,
    "U": glyph_U,
    "X": glyph_X,
    "Z": glyph_Z,
    # punctuation
    ".": glyph_dot,
    "?": glyph_qmark,
    "„": glyph_low_quote,
    "”": glyph_high_quote,
    # digits
    "0": digit_0,
    "1": digit_1,
    "2": digit_2,
    "3": digit_3,
    "4": digit_4,
    "5": digit_5,
    "6": digit_6,
    "7": digit_7,
    "8": digit_8,
    "9": digit_9,
}

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    grid = build_grid()
    chars = uniq_chars(REQUESTED)

    written = 0
    missing: List[str] = []

    for ch in chars:
        fn = BUILDERS.get(ch)
        if fn is None:
            # try case fallbacks for letters
            if ch.isalpha():
                fn = BUILDERS.get(ch.lower()) or BUILDERS.get(ch.upper())
        if fn is None:
            missing.append(ch)
            continue

        paths = fn(grid)
        svg = svg_doc(paths, title=f"glyph {ch} (U+{ord(ch):04X})")
        out_name = f"character-u{ord(ch):04x}.svg"
        write_text_lf(OUT_DIR / out_name, svg)
        written += 1
        print(f"✓ {ch} -> {out_name}")

    print(f"\nDone. Wrote {written} glyph(s) to: {OUT_DIR.resolve()}")
    if missing:
        print("Missing builders for:", "".join(missing))

if __name__ == "__main__":
    main()
