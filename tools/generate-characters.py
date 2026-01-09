#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate-characters-v2.py

Generates SVG glyphs (ONLY the characters present in the screenshot) using:
- A mathematically precise hex / Flower-of-Life circle grid
- Glyph strokes built from:
    * circular arcs (SVG A commands) and
    * straight lines
  with tangential arc↔line joins (smooth transitions)

Outputs into ../src (relative to this script).

Usage:
  python generate-characters-v2.py
  python generate-characters-v2.py --grid        # overlay grid for debugging
  python generate-characters-v2.py --r 18 --sw 14
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Writing helper (Windows-safe LF)
# -----------------------------
def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# -----------------------------
# Geometry / primitives
# -----------------------------
Point = Tuple[float, float]

@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def pt_on_circle(c: Circle, ang_deg: float) -> Point:
    """
    SVG-like coordinates: x right, y down.
    Angle convention:
      0° = +x (right)
      90° = +y (down)
      180° = -x (left)
      270° = -y (up)
    """
    a = deg2rad(ang_deg)
    return (c.cx + c.r * math.cos(a), c.cy + c.r * math.sin(a))

def norm_ang(a: float) -> float:
    a %= 360.0
    if a < 0:
        a += 360.0
    return a

def ang_delta_cw(a0: float, a1: float) -> float:
    """Clockwise delta from a0 to a1 under the y-down angle convention above."""
    a0 = norm_ang(a0)
    a1 = norm_ang(a1)
    d = a1 - a0
    if d < 0:
        d += 360.0
    return d

def ang_delta_ccw(a0: float, a1: float) -> float:
    """Counterclockwise delta (negative sweep)."""
    d = ang_delta_cw(a1, a0)
    return d

def svg_arc_to(r: float, end: Point, large_arc: int, sweep: int) -> str:
    x, y = end
    # x-axis-rotation = 0 (circles), rx=ry=r
    return f"A {r:.3f} {r:.3f} 0 {large_arc} {sweep} {x:.3f} {y:.3f}"

class PathBuilder:
    def __init__(self) -> None:
        self.cmds: List[str] = []
        self.cur: Optional[Point] = None

    def M(self, p: Point) -> "PathBuilder":
        self.cur = p
        self.cmds.append(f"M {p[0]:.3f} {p[1]:.3f}")
        return self

    def L(self, p: Point) -> "PathBuilder":
        if self.cur is None:
            return self.M(p)
        self.cur = p
        self.cmds.append(f"L {p[0]:.3f} {p[1]:.3f}")
        return self

    def arc(self, c: Circle, a0: float, a1: float, cw: bool = True) -> "PathBuilder":
        """
        Add an arc on circle c from angle a0 to a1.

        IMPORTANT: to keep it smooth, you typically call this with the current point
        already at pt_on_circle(c, a0) (we enforce that in glyph builders).
        """
        p0 = pt_on_circle(c, a0)
        p1 = pt_on_circle(c, a1)

        if self.cur is None:
            self.M(p0)
        else:
            # If we're not exactly at p0, connect (still ok; but ideally avoid)
            if (abs(self.cur[0] - p0[0]) > 1e-3) or (abs(self.cur[1] - p0[1]) > 1e-3):
                self.L(p0)

        if cw:
            d = ang_delta_cw(a0, a1)
            sweep = 1
        else:
            d = ang_delta_ccw(a0, a1)
            sweep = 0

        large_arc = 1 if d > 180.0 else 0
        self.cmds.append(svg_arc_to(c.r, p1, large_arc, sweep))
        self.cur = p1
        return self

    def d(self) -> str:
        return " ".join(self.cmds)

# -----------------------------
# Hex / Flower-of-Life grid
# -----------------------------
@dataclass(frozen=True)
class HexGrid:
    """
    Circle centers laid out in a standard hex packing:
      - circle radius = R
      - centers horizontally spaced by R
      - rows vertically spaced by R*sqrt(3)/2
      - odd rows offset by R/2
    """
    R: float
    ox: float
    oy: float

    @property
    def dy(self) -> float:
        return self.R * math.sqrt(3) / 2.0

    def center(self, col: int, row: int) -> Point:
        x = self.ox + col * self.R + (self.R / 2.0 if (row & 1) else 0.0)
        y = self.oy + row * self.dy
        return (x, y)

    def circle(self, col: int, row: int, k: float = 1.0) -> Circle:
        cx, cy = self.center(col, row)
        return Circle(cx, cy, self.R * k)

# -----------------------------
# SVG document
# -----------------------------
def svg_doc(
    paths: List[str],
    *,
    w: int,
    h: int,
    stroke_w: float,
    title: str,
    grid_paths: Optional[List[str]] = None,
) -> str:
    bg = ""
    if grid_paths:
        bg = "\n".join([f'    <path d="{d}" />' for d in grid_paths])
        bg = (
            '  <g fill="none" stroke="#777" stroke-opacity="0.35" '
            f'stroke-width="{max(1.0, stroke_w*0.10):.3f}" stroke-linecap="round" stroke-linejoin="round">\n'
            f"{bg}\n"
            "  </g>\n"
        )

    body = "\n".join([f'    <path d="{d}" />' for d in paths])

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f"  <title>{title}</title>\n"
        f"{bg}"
        '  <g fill="none" stroke="#000" '
        f'stroke-width="{stroke_w:.3f}" stroke-linecap="round" stroke-linejoin="round">\n'
        f"{body}\n"
        "  </g>\n"
        "</svg>\n"
    )

# -----------------------------
# Character set (ONLY screenshot)
# -----------------------------
CHAR_LINES = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]

def ordered_unique_chars(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        for ch in line:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
    return out

# -----------------------------
# Naming (Windows-safe, no case collisions)
# -----------------------------
def slug_for_char(ch: str) -> str:
    # Always safe, always unique
    return f"U{ord(ch):04X}"

def friendly_filename(ch: str) -> str:
    # Avoid A/a collisions on case-insensitive FS by prefixing
    if "A" <= ch <= "Z":
        return f"character-upper-{ch.lower()}.svg"
    if "a" <= ch <= "z":
        return f"character-lower-{ch}.svg"
    if "0" <= ch <= "9":
        return f"character-digit-{ch}.svg"
    specials = {
        ".": "period",
        "?": "question",
        "„": "quotedblbase",
        "”": "quotedblright",
    }
    if ch in specials:
        return f"character-{specials[ch]}.svg"
    return f"character-{slug_for_char(ch)}.svg"

# -----------------------------
# Glyph construction helpers
# -----------------------------
@dataclass(frozen=True)
class Metrics:
    cap: float
    xh: float
    base: float
    desc: float
    cx: float

def make_metrics(grid: HexGrid, w: int, h: int) -> Metrics:
    """
    Put key horizontal/vertical anchors on exact grid-derived values.
    We use row indices to define cap/x-height/baseline/descender.
    """
    # row indices (tune to taste, but grid-math exact)
    cap_row = 1
    xh_row = 4
    base_row = 10
    desc_row = 12

    cap = grid.oy + cap_row * grid.dy
    xh  = grid.oy + xh_row  * grid.dy
    base = grid.oy + base_row * grid.dy
    desc = grid.oy + desc_row * grid.dy

    cx = w / 2.0
    return Metrics(cap=cap, xh=xh, base=base, desc=desc, cx=cx)

def circle_full(pb: PathBuilder, c: Circle) -> None:
    # full circle as two arcs (smooth, no stray close-line)
    pb.M(pt_on_circle(c, 0)).arc(c, 0, 180, cw=True).arc(c, 180, 360, cw=True)

def dot_circle_paths(center: Point, r: float) -> str:
    c = Circle(center[0], center[1], r)
    pb = PathBuilder()
    circle_full(pb, c)
    return pb.d()

def comma_shape(x: float, y: float, r: float, tail: float, up: bool) -> str:
    """
    A single “comma-like” stroke built from a small circle arc + a short tail.
    up=False -> tail goes down (low quote „)
    up=True  -> tail goes up (high quote ”)
    """
    c = Circle(x, y, r)
    pb = PathBuilder()
    # Use a 240° arc so it reads like a droplet
    a0, a1 = (210, 510) if not up else (30, 330)
    pb.M(pt_on_circle(c, a0)).arc(c, a0, a1, cw=True)
    # Tail tangent: at endpoint a1, tangent angle is a1+90 (y-down)
    end = pt_on_circle(c, a1)
    tdir = norm_ang(a1 + 90.0)
    dx = math.cos(deg2rad(tdir)) * tail
    dy = math.sin(deg2rad(tdir)) * tail
    pb.L((end[0] + dx, end[1] + dy))
    return pb.d()

# -----------------------------
# Glyphs (ONLY the screenshot set)
# -----------------------------
def build_glyphs(grid: HexGrid, w: int, h: int) -> Dict[str, List[str]]:
    m = make_metrics(grid, w, h)
    dy = grid.dy

    # Common radii derived from the grid
    # (Multiples of dy give nice vertical alignment; still fully grid-derived.)
    R_bowl = 3.0 * dy     # main bowl radius for o/a/b/d/p/q/0/6/9
    R_arch = 2.0 * dy     # arches for n/m/h/u/v/w
    R_hook = 1.5 * dy     # hooks (f, t, J, ?)
    R_small = 0.6 * dy    # punctuation dots / quote heads

    # Common x anchors (col-based, exact)
    x_left  = grid.ox + 3 * grid.R
    x_right = grid.ox + 9 * grid.R
    x_mid   = m.cx

    # Bowl center between x-height and baseline (exact)
    cy_bowl = (m.xh + m.base) / 2.0
    c_bowl  = Circle(x_mid, cy_bowl, R_bowl)

    # Arch centers and y positions
    # For top arches (n/m/h): center slightly below x-height so arch “hangs” nicely.
    cy_arch_top = m.xh + 0.5 * dy
    # For bottom arches (u/v/w/U): center slightly above baseline.
    cy_arch_bot = m.base - 0.5 * dy

    def glyph_o() -> List[str]:
        pb = PathBuilder()
        circle_full(pb, c_bowl)
        return [pb.d()]

    def glyph_c(lower: bool = True) -> List[str]:
        # Open on the right: leave a gap around angle ~0
        pb = PathBuilder()
        # Draw a 300° arc leaving 60° gap on right
        pb.M(pt_on_circle(c_bowl, 40)).arc(c_bowl, 40, 320, cw=True)
        return [pb.d()]

    def glyph_a(upper: bool = False) -> List[str]:
        # One-storey a: open-ish bowl + right stem reaching cap (A is same but taller top)
        pb = PathBuilder()
        # bowl with small right gap
        pb.M(pt_on_circle(c_bowl, 60)).arc(c_bowl, 60, 360 + 300, cw=True)
        # right stem tangent at angle 0 (tangent vertical)
        p_right = pt_on_circle(c_bowl, 0)
        top_y = m.cap if upper else m.xh - 0.5 * dy
        pb.L((p_right[0], top_y))
        return [pb.d()]

    def glyph_b() -> List[str]:
        # Stem left, bowl attached tangentially at angle 180
        pb = PathBuilder()
        p_left = pt_on_circle(c_bowl, 180)
        pb.M((p_left[0], m.cap)).L(p_left).arc(c_bowl, 180, 540, cw=True)
        return [pb.d()]

    def glyph_d() -> List[str]:
        pb = PathBuilder()
        p_right = pt_on_circle(c_bowl, 0)
        pb.M((p_right[0], m.cap)).L(p_right).arc(c_bowl, 0, 360, cw=True)
        return [pb.d()]

    def glyph_p() -> List[str]:
        # Like b, but stem continues into descender
        pb = PathBuilder()
        p_left = pt_on_circle(c_bowl, 180)
        pb.M((p_left[0], m.xh - 0.5 * dy)).L((p_left[0], m.desc)).L(p_left).arc(c_bowl, 180, 540, cw=True)
        return [pb.d()]

    def glyph_q() -> List[str]:
        pb = PathBuilder()
        p_right = pt_on_circle(c_bowl, 0)
        pb.M((p_right[0], m.xh - 0.5 * dy)).L((p_right[0], m.desc)).L(p_right).arc(c_bowl, 0, 360, cw=True)
        return [pb.d()]

    def glyph_e(upper: bool = True) -> List[str]:
        # E: like C plus a mid stroke into the bowl (tangent-ish)
        pb = PathBuilder()
        pb.M(pt_on_circle(c_bowl, 40)).arc(c_bowl, 40, 320, cw=True)
        # mid arm: from left-ish point toward center
        mid_y = cy_bowl
        arm_x0 = pt_on_circle(c_bowl, 180)[0]
        pb.M((arm_x0, mid_y)).L((x_mid, mid_y))
        return [pb.d()]

    def glyph_h() -> List[str]:
        # left stem to cap, then shoulder arch to right stem (tangent joins)
        pb = PathBuilder()
        xL = x_left
        xR = x_right
        # left stem
        pb.M((xL, m.cap)).L((xL, m.base))
        # shoulder: circle centered at mid between stems, lower than cap
        c = Circle((xL + xR) / 2.0, cy_arch_top, R_arch)
        # start at leftmost (tangent vertical), go via top to rightmost
        pb.M(pt_on_circle(c, 180)).arc(c, 180, 360, cw=True).L((xR, m.base))
        # right stem segment up to shoulder join (visually)
        pb.M((xR, m.base)).L((xR, m.xh))
        return [pb.d()]

    def glyph_n() -> List[str]:
        pb = PathBuilder()
        xL = x_left
        xR = x_right
        pb.M((xL, m.xh)).L((xL, m.base))
        c = Circle((xL + xR) / 2.0, cy_arch_top, R_arch)
        # connect from leftmost to rightmost along top
        pb.M(pt_on_circle(c, 180)).arc(c, 180, 360, cw=True).L((xR, m.base))
        pb.M((xR, m.base)).L((xR, m.xh))
        return [pb.d()]

    def glyph_m() -> List[str]:
        # two shoulders
        pb = PathBuilder()
        x1 = x_left
        x3 = x_right
        x2 = (x1 + x3) / 2.0
        pb.M((x1, m.xh)).L((x1, m.base))

        c1 = Circle((x1 + x2) / 2.0, cy_arch_top, R_arch)
        pb.M(pt_on_circle(c1, 180)).arc(c1, 180, 360, cw=True).L((x2, m.base)).M((x2, m.base)).L((x2, m.xh))

        c2 = Circle((x2 + x3) / 2.0, cy_arch_top, R_arch)
        pb.M(pt_on_circle(c2, 180)).arc(c2, 180, 360, cw=True).L((x3, m.base)).M((x3, m.base)).L((x3, m.xh))
        return [pb.d()]

    def glyph_u(upper: bool = False) -> List[str]:
        pb = PathBuilder()
        xL = x_left
        xR = x_right
        top_y = m.cap if upper else m.xh
        pb.M((xL, top_y)).L((xL, m.base))
        c = Circle((xL + xR) / 2.0, cy_arch_bot, R_arch)
        # bottom half: rightmost -> leftmost via bottom
        pb.M(pt_on_circle(c, 0)).arc(c, 0, 180, cw=True)
        pb.L((xR, m.base)).L((xR, top_y))
        return [pb.d()]

    def glyph_v() -> List[str]:
        # a sharper u: two arcs meeting at a bottom point
        pb = PathBuilder()
        xL, xR = x_left, x_right
        top_y = m.xh
        pb.M((xL, top_y)).L((xL, m.base))
        # two circles, each contributes a quarter-ish arc to meet at bottom mid
        midx = (xL + xR) / 2.0
        bottom = (midx, m.base)
        c1 = Circle(xL + R_arch, cy_arch_bot, R_arch)
        c2 = Circle(xR - R_arch, cy_arch_bot, R_arch)
        pb.M(pt_on_circle(c1, 90)).arc(c1, 90, 0, cw=False).L(bottom).L(pt_on_circle(c2, 180)).arc(c2, 180, 90, cw=True)
        pb.L((xR, m.base)).L((xR, top_y))
        return [pb.d()]

    def glyph_w() -> List[str]:
        # double u
        pb = PathBuilder()
        top_y = m.xh
        x0 = x_left
        x3 = x_right
        x1 = (2*x0 + x3)/3
        x2 = (x0 + 2*x3)/3
        pb.M((x0, top_y)).L((x0, m.base))

        c1 = Circle((x0 + x1)/2, cy_arch_bot, R_arch)
        pb.M(pt_on_circle(c1, 0)).arc(c1, 0, 180, cw=True).L((x1, m.base)).L((x1, top_y)).L((x1, m.base))

        c2 = Circle((x2 + x3)/2, cy_arch_bot, R_arch)
        pb.M(pt_on_circle(c2, 0)).arc(c2, 0, 180, cw=True).L((x3, m.base)).L((x3, top_y))
        return [pb.d()]

    def glyph_y() -> List[str]:
        # v + descender stem
        pb = PathBuilder()
        xL, xR = x_left, x_right
        top_y = m.xh
        pb.M((xL, top_y)).L((xL, m.base))
        c = Circle((xL + xR)/2, cy_arch_bot, R_arch)
        pb.M(pt_on_circle(c, 0)).arc(c, 0, 180, cw=True).L((xR, m.base)).L((xR, top_y))
        # descender from mid
        pb.M(((xL + xR)/2, m.base)).L(((xL + xR)/2, m.desc))
        return [pb.d()]

    def glyph_t() -> List[str]:
        pb = PathBuilder()
        x = x_mid
        pb.M((x, m.cap)).L((x, m.base))
        # hook at top using a circle
        c = Circle(x - R_hook, m.cap + R_hook, R_hook)
        pb.M(pt_on_circle(c, 270)).arc(c, 270, 360, cw=True)
        return [pb.d()]

    def glyph_f() -> List[str]:
        pb = PathBuilder()
        x = x_mid
        pb.M((x, m.cap)).L((x, m.base))
        # top hook
        c = Circle(x - R_hook, m.cap + R_hook, R_hook)
        pb.M(pt_on_circle(c, 270)).arc(c, 270, 360, cw=True)
        # mid arm (small)
        pb.M((x - 0.8*R_hook, m.xh + 0.6*dy)).L((x + 0.6*R_hook, m.xh + 0.6*dy))
        return [pb.d()]

    def glyph_g() -> List[str]:
        pb = PathBuilder()
        circle_full(pb, c_bowl)
        # descender hook on right
        xR = pt_on_circle(c_bowl, 0)[0]
        pb.M((xR, m.base)).L((xR, m.desc - 0.2*dy))
        c = Circle(xR - 0.6*R_hook, m.desc - 0.2*dy, 0.6*R_hook)
        pb.M(pt_on_circle(c, 0)).arc(c, 0, 180, cw=True)
        return [pb.d()]

    def glyph_r() -> List[str]:
        pb = PathBuilder()
        xL = x_left
        pb.M((xL, m.xh)).L((xL, m.base))
        # small shoulder
        c = Circle(xL + R_hook, m.xh + 0.5*R_hook, R_hook)
        pb.M(pt_on_circle(c, 180)).arc(c, 180, 300, cw=True)
        return [pb.d()]

    def glyph_k(upper: bool = False) -> List[str]:
        pb = PathBuilder()
        xL = x_left
        top_y = m.cap if upper else m.xh
        pb.M((xL, m.cap if upper else m.xh)).L((xL, m.base))
        # two diagonals with rounded-ish endpoints (still line; joins are round)
        midy = (m.xh + m.base)/2
        pb.M((xL, midy)).L((x_right, top_y))
        pb.M((xL, midy)).L((x_right, m.base))
        return [pb.d()]

    def glyph_x(upper: bool = False) -> List[str]:
        # Use two opposing circle arcs to get the rounded “X” feel.
        pb1 = PathBuilder()
        pb2 = PathBuilder()
        top_y = m.cap if upper else m.xh
        bot_y = m.base
        # Circle centers placed so arcs cross near center
        c1 = Circle(x_mid - R_arch*0.25, (top_y + bot_y)/2, R_arch*1.1)
        c2 = Circle(x_mid + R_arch*0.25, (top_y + bot_y)/2, R_arch*1.1)
        # arc1: upper-left to lower-right
        pb1.M(pt_on_circle(c1, 240)).arc(c1, 240, 60, cw=True)
        # arc2: lower-left to upper-right
        pb2.M(pt_on_circle(c2, 120)).arc(c2, 120, 300, cw=True)
        return [pb1.d(), pb2.d()]

    def glyph_I() -> List[str]:
        pb = PathBuilder()
        pb.M((x_mid, m.cap)).L((x_mid, m.base))
        return [pb.d()]

    def glyph_J() -> List[str]:
        pb = PathBuilder()
        x = x_right
        # top hook leftwards
        ctop = Circle(x - R_hook, m.cap + R_hook, R_hook)
        pb.M(pt_on_circle(ctop, 270)).arc(ctop, 270, 360, cw=True)
        pb.L((x, m.base - 0.3*dy))
        # bottom hook to left
        cb = Circle(x - R_hook, m.base - 0.3*dy, R_hook)
        pb.arc(cb, 0, 180, cw=True)
        return [pb.d()]

    def glyph_L() -> List[str]:
        pb = PathBuilder()
        pb.M((x_left, m.cap)).L((x_left, m.base)).L((x_right, m.base))
        return [pb.d()]

    def glyph_C() -> List[str]:
        # Tall C: top half circle + left stem + bottom half circle, open on right
        pb = PathBuilder()
        r = R_bowl
        # two circles (top + bottom) share same radius and left stem connects at leftmost points
        ctop = Circle(x_mid, m.cap + r, r)
        cb   = Circle(x_mid, m.base - r, r)
        pb.M(pt_on_circle(ctop, 0)).arc(ctop, 0, 180, cw=True)  # top half via up
        pb.L(pt_on_circle(cb, 180))
        pb.arc(cb, 180, 0, cw=True)  # bottom half via down
        return [pb.d()]

    def glyph_E() -> List[str]:
        # C plus a middle arm inward
        paths = glyph_C()
        pb = PathBuilder()
        pb.M((x_left, (m.cap + m.base)/2)).L((x_mid, (m.cap + m.base)/2))
        paths.append(pb.d())
        return paths

    def glyph_S() -> List[str]:
        # Two opposing arcs (grid-derived radii)
        pb = PathBuilder()
        c1 = Circle(x_mid, m.xh + R_arch*0.6, R_arch)
        c2 = Circle(x_mid, m.base - R_arch*0.6, R_arch)
        pb.M(pt_on_circle(c1, 200)).arc(c1, 200, 20, cw=True)
        pb.M(pt_on_circle(c2, 160)).arc(c2, 160, 340, cw=True)
        return [pb.d()]

    def glyph_U() -> List[str]:
        return glyph_u(upper=True)

    def glyph_Z() -> List[str]:
        # Zigzag with rounded joins (stroke-linejoin round)
        pb = PathBuilder()
        pb.M((x_left, m.cap)).L((x_right, m.cap)).L((x_left, m.base)).L((x_right, m.base))
        return [pb.d()]

    def glyph_G() -> List[str]:
        paths = glyph_C()
        pb = PathBuilder()
        pb.M((x_mid, cy_bowl)).L((x_right, cy_bowl))
        paths.append(pb.d())
        return paths

    def glyph_H() -> List[str]:
        pb = PathBuilder()
        pb.M((x_left, m.cap)).L((x_left, m.base))
        pb.M((x_right, m.cap)).L((x_right, m.base))
        pb.M((x_left, (m.xh + m.base)/2)).L((x_right, (m.xh + m.base)/2))
        return [pb.d()]

    def glyph_M() -> List[str]:
        # Like m but cap height
        pb = PathBuilder()
        x1 = x_left
        x3 = x_right
        x2 = (x1 + x3)/2
        pb.M((x1, m.cap)).L((x1, m.base))
        pb.M((x1, m.cap)).L((x2, m.base - 0.5*dy)).L((x3, m.cap)).L((x3, m.base))
        return [pb.d()]

    def glyph_N() -> List[str]:
        pb = PathBuilder()
        pb.M((x_left, m.cap)).L((x_left, m.base))
        pb.M((x_left, m.cap)).L((x_right, m.base))
        pb.M((x_right, m.cap)).L((x_right, m.base))
        return [pb.d()]

    def glyph_K() -> List[str]:
        return glyph_k(upper=True)

    def glyph_X() -> List[str]:
        return glyph_x(upper=True)

    # Digits
    def digit_0() -> List[str]:
        return glyph_o()

    def digit_1() -> List[str]:
        pb = PathBuilder()
        pb.M((x_mid, m.cap)).L((x_mid, m.base))
        return [pb.d()]

    def digit_2() -> List[str]:
        pb = PathBuilder()
        ctop = Circle(x_mid, m.cap + R_arch, R_arch)
        pb.M(pt_on_circle(ctop, 180)).arc(ctop, 180, 360, cw=True)
        pb.L((x_left, m.base)).L((x_right, m.base))
        return [pb.d()]

    def digit_3() -> List[str]:
        pb = PathBuilder()
        c1 = Circle(x_mid, m.cap + R_arch, R_arch)
        c2 = Circle(x_mid, m.base - R_arch, R_arch)
        pb.M(pt_on_circle(c1, 180)).arc(c1, 180, 360, cw=True)
        pb.M(pt_on_circle(c2, 180)).arc(c2, 180, 360, cw=True)
        return [pb.d()]

    def digit_4() -> List[str]:
        pb = PathBuilder()
        pb.M((x_right, m.cap)).L((x_right, m.base))
        pb.M((x_left, m.xh + 1.2*dy)).L((x_right, m.xh + 1.2*dy))
        pb.M((x_left, m.cap + 2*dy)).L((x_left, m.xh + 1.2*dy))
        return [pb.d()]

    def digit_5() -> List[str]:
        pb = PathBuilder()
        pb.M((x_right, m.cap)).L((x_left, m.cap)).L((x_left, (m.xh+m.base)/2))
        c = Circle(x_mid, m.base - R_arch*0.4, R_arch)
        pb.M(pt_on_circle(c, 200)).arc(c, 200, 20, cw=True)
        return [pb.d()]

    def digit_6() -> List[str]:
        pb = PathBuilder()
        pb.M(pt_on_circle(c_bowl, 60)).arc(c_bowl, 60, 420, cw=True)
        # inward curl
        c = Circle(x_mid, cy_bowl, R_arch*0.7)
        pb.M(pt_on_circle(c, 200)).arc(c, 200, 520, cw=True)
        return [pb.d()]

    def digit_7() -> List[str]:
        pb = PathBuilder()
        pb.M((x_left, m.cap)).L((x_right, m.cap)).L((x_left, m.base))
        return [pb.d()]

    def digit_8() -> List[str]:
        pb = PathBuilder()
        ctop = Circle(x_mid, m.xh + R_arch*0.4, R_arch*0.8)
        cb   = Circle(x_mid, m.base - R_arch*0.4, R_arch*0.8)
        circle_full(pb, ctop)
        pb2 = PathBuilder()
        circle_full(pb2, cb)
        return [pb.d(), pb2.d()]

    def digit_9() -> List[str]:
        pb = PathBuilder()
        circle_full(pb, c_bowl)
        pb.M((pt_on_circle(c_bowl, 0)[0], cy_bowl)).L((pt_on_circle(c_bowl, 0)[0], m.base))
        return [pb.d()]

    # Punctuation
    def glyph_period() -> List[str]:
        return [dot_circle_paths((x_mid, m.base), R_small*0.35)]

    def glyph_question() -> List[str]:
        pb = PathBuilder()
        c = Circle(x_mid, m.cap + R_hook, R_hook)
        pb.M(pt_on_circle(c, 200)).arc(c, 200, 20, cw=True)
        pb.L((x_mid, m.xh + 1.2*dy))
        dot = dot_circle_paths((x_mid, m.base), R_small*0.35)
        return [pb.d(), dot]

    def glyph_quotedblbase() -> List[str]:
        # low quotes „ : two comma marks near baseline
        x1 = x_mid - 0.6*R_small
        x2 = x_mid + 0.6*R_small
        y = m.base - 0.2*dy
        return [
            comma_shape(x1, y, R_small*0.45, tail=R_small*0.7, up=False),
            comma_shape(x2, y, R_small*0.45, tail=R_small*0.7, up=False),
        ]

    def glyph_quotedblright() -> List[str]:
        # high quotes ” : two comma marks near cap/xh region (tails up)
        x1 = x_mid - 0.6*R_small
        x2 = x_mid + 0.6*R_small
        y = m.xh - 0.8*dy
        return [
            comma_shape(x1, y, R_small*0.45, tail=R_small*0.7, up=True),
            comma_shape(x2, y, R_small*0.45, tail=R_small*0.7, up=True),
        ]

    glyphs: Dict[str, List[str]] = {}

    # Lowercase
    glyphs["a"] = glyph_a(upper=False)
    glyphs["b"] = glyph_b()
    glyphs["c"] = glyph_c(lower=True)
    glyphs["d"] = glyph_d()
    glyphs["f"] = glyph_f()
    glyphs["g"] = glyph_g()
    glyphs["h"] = glyph_h()
    glyphs["k"] = glyph_k(upper=False)
    glyphs["m"] = glyph_m()
    glyphs["n"] = glyph_n()
    glyphs["o"] = glyph_o()
    glyphs["p"] = glyph_p()
    glyphs["q"] = glyph_q()
    glyphs["r"] = glyph_r()
    glyphs["t"] = glyph_t()
    glyphs["u"] = glyph_u(upper=False)
    glyphs["v"] = glyph_v()
    glyphs["w"] = glyph_w()
    glyphs["x"] = glyph_x(upper=False)
    glyphs["y"] = glyph_y()

    # Uppercase (only those in screenshot list)
    glyphs["A"] = glyph_a(upper=True)
    glyphs["C"] = glyph_C()
    glyphs["E"] = glyph_E()
    glyphs["I"] = glyph_I()
    glyphs["J"] = glyph_J()
    glyphs["L"] = glyph_L()
    glyphs["S"] = glyph_S()
    glyphs["U"] = glyph_U()
    glyphs["Z"] = glyph_Z()
    glyphs["G"] = glyph_G()
    glyphs["H"] = glyph_H()
    glyphs["K"] = glyph_K()
    glyphs["M"] = glyph_M()
    glyphs["N"] = glyph_N()
    glyphs["X"] = glyph_X()

    # Digits
    glyphs["0"] = digit_0()
    glyphs["1"] = digit_1()
    glyphs["2"] = digit_2()
    glyphs["3"] = digit_3()
    glyphs["4"] = digit_4()
    glyphs["5"] = digit_5()
    glyphs["6"] = digit_6()
    glyphs["7"] = digit_7()
    glyphs["8"] = digit_8()
    glyphs["9"] = digit_9()

    # Punctuation (only those in screenshot list)
    glyphs["."] = glyph_period()
    glyphs["?"] = glyph_question()
    glyphs["„"] = glyph_quotedblbase()
    glyphs["”"] = glyph_quotedblright()

    return glyphs

def build_grid_overlay(grid: HexGrid, w: int, h: int, cols: int = 12, rows: int = 12) -> List[str]:
    """
    Optional debug overlay: a Flower-of-Life-ish patch of circles.
    """
    out: List[str] = []
    # Draw circles centered in a rectangle around the glyph
    for r in range(-2, rows + 2):
        for c in range(-2, cols + 2):
            cx, cy = grid.center(c, r)
            # only keep if near canvas
            if -grid.R*2 <= cx <= w + grid.R*2 and -grid.R*2 <= cy <= h + grid.R*2:
                circle = Circle(cx, cy, grid.R)
                pb = PathBuilder()
                circle_full(pb, circle)
                out.append(pb.d())
    return out

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=20.0, help="Base grid circle radius (default: 20.0)")
    ap.add_argument("--sw", type=float, default=14.0, help="Stroke width (default: 14.0)")
    ap.add_argument("--w", type=int, default=240, help="SVG width (default: 240)")
    ap.add_argument("--h", type=int, default=240, help="SVG height (default: 240)")
    ap.add_argument("--grid", action="store_true", help="Overlay background grid circles in each glyph")
    args = ap.parse_args()

    out_dir = (Path(__file__).resolve().parent / "../src").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid origin: align nicely inside the glyph box (tweakable)
    grid = HexGrid(R=args.r, ox=args.r * 0.5, oy=args.r * 0.5)

    wanted = ordered_unique_chars(CHAR_LINES)
    glyphs = build_glyphs(grid, args.w, args.h)

    grid_overlay = build_grid_overlay(grid, args.w, args.h) if args.grid else None

    manifest_lines: List[str] = ["char\tcodepoint\tfile_codepoint\tfile_friendly"]

    for ch in wanted:
        if ch not in glyphs:
            # Intentionally do not invent missing glyphs
            print(f"[skip] no glyph defined for {repr(ch)}")
            continue

        paths = glyphs[ch]
        title = f"glyph {repr(ch)}"

        svg = svg_doc(
            paths,
            w=args.w,
            h=args.h,
            stroke_w=args.sw,
            title=title,
            grid_paths=grid_overlay,
        )

        cp = ord(ch)
        file_cp = f"character-U{cp:04X}.svg"
        file_friendly = friendly_filename(ch)

        write_text_lf(out_dir / file_cp, svg)
        write_text_lf(out_dir / file_friendly, svg)

        manifest_lines.append(f"{repr(ch)}\tU{cp:04X}\t{file_cp}\t{file_friendly}")

    write_text_lf(out_dir / "glyph-manifest.tsv", "\n".join(manifest_lines) + "\n")

    print(f"Done. Wrote glyphs to: {out_dir}")
    print(f"Manifest: {out_dir / 'glyph-manifest.tsv'}")

if __name__ == "__main__":
    main()
