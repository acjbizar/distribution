#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path as FSPath
from typing import Callable, Dict, List, Tuple, Any

# -----------------------------
# Grid (circles touch; rows/cols)
# - Circle radius R
# - Centers at 40/120/200/(280) => circles span 0..320 vertically
# - We reserve row 0 as TOP MARGIN.
# -----------------------------
R = 40.0
DX = 2.0 * R
DY = 2.0 * R

X0 = 40.0
Y0 = 40.0

GRID_ROWS = 4            # 4 circles high
GRID_COLS_DEFAULT = 3    # 3 circles wide by default
GRID_COLS_WIDE = 4       # for m / M / w
GRID_COLS_NARROW = 2     # for I

BASE_ROW = 1             # glyphs start at row 1 (row 0 is top margin)

VIEW_H = int(GRID_ROWS * DY)              # 320
VIEW_W_DEFAULT = int(GRID_COLS_DEFAULT * DX)  # 240
VIEW_W_WIDE = int(GRID_COLS_WIDE * DX)        # 320
VIEW_W_NARROW = int(GRID_COLS_NARROW * DX)    # 160

STROKE = 9.0

# Debug overlay
GRID_OPACITY = 0.35
GRID_STROKE = 1.0
ANCHOR_R = 3.5

DOT_R = 1.0  # ✅ requested: radius 1 for all dots


# -----------------------------
# Utils
# -----------------------------
def uniq(s: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ch in s:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def write_text_lf(path: FSPath, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


# -----------------------------
# Geometry helpers
# -----------------------------
@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def L(self) -> Tuple[float, float]: return (self.cx - self.r, self.cy)
    def R(self) -> Tuple[float, float]: return (self.cx + self.r, self.cy)
    def T(self) -> Tuple[float, float]: return (self.cx, self.cy - self.r)
    def B(self) -> Tuple[float, float]: return (self.cx, self.cy + self.r)


@dataclass(frozen=True)
class Grid:
    r: float
    x0: float
    y0: float

    def center(self, col: int, row: int) -> Tuple[float, float]:
        return (self.x0 + col * DX, self.y0 + row * DY)

    def circle(self, col: int, row: int) -> Circle:
        cx, cy = self.center(col, row)
        return Circle(cx=cx, cy=cy, r=self.r)

    def x_tan(self, col_left: int) -> float:
        # vertical tangent between col_left and col_left+1
        return self.x0 + col_left * DX + self.r

    def y_tan(self, row_top: int) -> float:
        # horizontal tangent between row_top and row_top+1
        return self.y0 + row_top * DY + self.r

    def tangency_vertical(self, col: int, row_top: int) -> Tuple[float, float]:
        # point where circles at (col,row_top) and (col,row_top+1) touch
        return self.circle(col, row_top).B()


class SvgPath:
    def __init__(self) -> None:
        self.parts: List[str] = []

    def M(self, p: Tuple[float, float]) -> "SvgPath":
        self.parts.append(f"M {fmt(p[0])} {fmt(p[1])}")
        return self

    def L(self, p: Tuple[float, float]) -> "SvgPath":
        self.parts.append(f"L {fmt(p[0])} {fmt(p[1])}")
        return self

    def A(self, r: float, large: int, sweep: int, p: Tuple[float, float]) -> "SvgPath":
        self.parts.append(f"A {fmt(r)} {fmt(r)} 0 {large} {sweep} {fmt(p[0])} {fmt(p[1])}")
        return self

    def d(self) -> str:
        return " ".join(self.parts)


def circle_full(c: Circle, sweep: int = 1) -> str:
    # R -> B -> L -> T -> R
    p = SvgPath().M(c.R())
    p.A(c.r, 0, sweep, c.B())
    p.A(c.r, 0, sweep, c.L())
    p.A(c.r, 0, sweep, c.T())
    p.A(c.r, 0, sweep, c.R())
    return p.d()


# -----------------------------
# Metrics (with top-margin row)
# -----------------------------
def ascender_top(grid: Grid) -> float:
    # top tangent of row BASE_ROW circle
    return grid.circle(0, BASE_ROW).T()[1]

def xheight_top(grid: Grid) -> float:
    # top tangent of row BASE_ROW+1 circle
    return grid.circle(0, BASE_ROW + 1).T()[1]

def baseline(grid: Grid) -> float:
    # bottom tangent of row BASE_ROW+1 circle
    return grid.circle(0, BASE_ROW + 1).B()[1]

def descender_bottom(grid: Grid) -> float:
    return grid.circle(0, GRID_ROWS - 1).B()[1]


# -----------------------------
# Glyph builders
# -----------------------------
# Backward compatible: a glyph fn may return (paths, anchors) or (paths, anchors, dots)
Dot = Tuple[float, float, float]  # (cx, cy, r)
GlyphRes2 = Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]
GlyphRes3 = Tuple[List[str], List[Tuple[str, Tuple[float, float]]], List[Dot]]
GlyphFn = Callable[[Grid], Any]


# ---- LOCKED / FIXED GLYPHS ----

def glyph_a(grid: Grid) -> GlyphRes2:
    # Two stacked circles at col=1 rows BASE_ROW and BASE_ROW+1
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    A = cu.L()
    B = cu.T()
    C = cu.R()
    D = cl.R()
    E = cl.B()
    F = cl.L()
    G = grid.tangency_vertical(1, BASE_ROW)  # meet between the two

    outer = (
        SvgPath()
        .M(A).A(cu.r, 0, 1, B).A(cu.r, 0, 1, C)
        .L(D)
        .A(cl.r, 0, 1, E).A(cl.r, 0, 1, F)
        .d()
    )
    # closure arcs (approved)
    inner_GC = SvgPath().M(G).A(cu.r, 0, 0, C).d()
    inner_GF = SvgPath().M(G).A(cl.r, 0, 0, F).d()

    anchors = [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F), ("G", G)]
    return [outer, inner_GC, inner_GF], anchors


def glyph_A(grid: Grid) -> GlyphRes2:
    return glyph_a(grid)


def glyph_b(grid: Grid) -> GlyphRes2:
    # bowl at (1, BASE_ROW+1), stem on x_tan(0) down to bowl left tangent
    bowl = grid.circle(1, BASE_ROW + 1)
    x_stem = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_join = bowl.L()[1]

    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    bowl_path = circle_full(bowl)
    return [stem, bowl_path], [("stemTop", (x_stem, y_top)), ("join", (x_stem, y_join))]


def glyph_C(grid: Grid) -> GlyphRes2:
    """
    Fixed C:
      top half-circle (upper circle) + left connector + bottom half-circle (lower circle)
      Open on the right.
      (This is the version you approved: line on LEFT.)
    """
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    TR = cu.R()
    TT = cu.T()
    TL = cu.L()

    BL = cl.L()
    BB = cl.B()
    BR = cl.R()

    p = SvgPath().M(TR)
    # top half: TR -> TT -> TL (go via top) => sweep=0
    p.A(cu.r, 0, 0, TT).A(cu.r, 0, 0, TL)
    # left connector
    p.L(BL)
    # bottom half: BL -> BB -> BR (go via bottom) => sweep=0
    p.A(cl.r, 0, 0, BB).A(cl.r, 0, 0, BR)

    return [p.d()], [("TR", TR), ("TL", TL), ("BL", BL), ("BR", BR)]


def glyph_E(grid: Grid) -> GlyphRes2:
    """
    Fixed E = continuous wrap across two stacked circles through tangency,
    using sweep=0 to wrap the left side.
    """
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    p = SvgPath().M(cu.R())
    p.A(cu.r, 0, 0, cu.T()).A(cu.r, 0, 0, cu.L()).A(cu.r, 0, 0, G)
    p.A(cl.r, 0, 0, cl.L()).A(cl.r, 0, 0, cl.B()).A(cl.r, 0, 0, cl.R())

    return [p.d()], [("G", G)]

def glyph_G(grid: Grid):
    # G = fixed C + top-right quarter-circle arc at the lower-right (same-circle principle)
    paths, _ = glyph_C(grid)

    cl = grid.circle(1, BASE_ROW + 1)
    BR = cl.R()
    end = cl.T()  # top-right quarter of the LOWER circle: R -> T

    turn = SvgPath().M(BR).A(cl.r, 0, 0, end).d()

    return paths + [turn], [("turnEnd", end)]

def glyph_k(grid: Grid) -> GlyphRes2:
    """
    Fixed k:
      - left stem: full height (ascender_top -> baseline)
      - main arch: circle at (1, BASE_ROW+1): L->T->R (sweep=1)
      - quarter arc: from tangency G to upper circle R (sweep=0), CONNECTED
      - right stem: ONLY from nearest arc point (cl.R) down to baseline
    """
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)

    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    stem_left = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    arch = SvgPath().M(cl.L()).A(cl.r, 0, 1, cl.T()).A(cl.r, 0, 1, cl.R()).d()

    quarter = SvgPath().M(G).A(cu.r, 0, 0, cu.R()).d()

    stem_right = SvgPath().M(cl.R()).L((cl.R()[0], y_base)).d()

    anchors = [("G", G), ("clR", cl.R()), ("baseR", (cl.R()[0], y_base))]
    return [stem_left, arch, quarter, stem_right], anchors


# ---- OTHER GLYPHS ----

def glyph_c(grid: Grid) -> GlyphRes2:
    # c = u rotated 90° (opens to the right)
    c = grid.circle(1, BASE_ROW + 1)
    xR = grid.x_tan(1)

    stem_top = SvgPath().M(c.T()).L((xR, c.T()[1])).d()
    stem_bot = SvgPath().M(c.B()).L((xR, c.B()[1])).d()

    arc = (
        SvgPath()
        .M(c.T())
        .A(c.r, 0, 0, c.L())
        .A(c.r, 0, 0, c.B())
        .d()
    )

    return [stem_top, arc, stem_bot], []


def glyph_d(grid: Grid) -> GlyphRes2:
    bowl = grid.circle(1, BASE_ROW + 1)
    x_stem = grid.x_tan(1)
    y_top = ascender_top(grid)
    y_join = bowl.R()[1]
    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    return [stem, circle_full(bowl)], []


def glyph_f(grid: Grid) -> GlyphRes2:
    # f: stem baseline->cu.L, top arch, mid hook quarter arc only
    x = grid.x_tan(0)
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    y_base = baseline(grid)

    stem = SvgPath().M((x, y_base)).L(cu.L()).d()

    top_arch = (
        SvgPath()
        .M(cu.L())
        .A(cu.r, 0, 1, cu.T())
        .A(cu.r, 0, 1, cu.R())
        .d()
    )

    mid_hook = SvgPath().M(cl.L()).A(cl.r, 0, 1, cl.T()).d()
    return [stem, top_arch, mid_hook], []


def glyph_g(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    top = circle_full(cu)

    # bottom loop open at the top-left: draw T->R->B->L (leave L->T open)
    bottom = (
        SvgPath()
        .M(cl.T())
        .A(cl.r, 0, 1, cl.R())
        .A(cl.r, 0, 1, cl.B())
        .A(cl.r, 0, 1, cl.L())
        .d()
    )

    return [top, bottom], []


def glyph_h(grid: Grid) -> GlyphRes2:
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()
    arch = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    leg = SvgPath().M(c.R()).L((c.R()[0], y_base)).d()
    return [stem, arch, leg], []

def glyph_i(grid: Grid):
    # i = short stem (1 circle high) + dot (like period) one circle above the stem
    x = grid.x_tan(0)

    y_base = baseline(grid)
    y_top = xheight_top(grid)              # exactly 1 circle (80px) above baseline
    stem = SvgPath().M((x, y_base)).L((x, y_top)).d()

    dot_y = y_top - DY                     # one circle above the stem
    dots = [(x, dot_y, DOT_R)]             # DOT_R is 1.0 in your script

    return [stem], [], dots

def glyph_I(grid: Grid) -> GlyphRes2:
    x = grid.x_tan(0)
    return [SvgPath().M((x, ascender_top(grid))).L((x, baseline(grid))).d()], []


def glyph_J(grid: Grid) -> GlyphRes2:
    x = grid.x_tan(1)
    y_top = ascender_top(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stem = SvgPath().M((x, y_top)).L(c.R()).d()
    hook = SvgPath().M(c.R()).A(c.r, 0, 1, c.B()).A(c.r, 0, 1, c.L()).d()
    return [stem, hook], []


def glyph_L(grid: Grid) -> GlyphRes2:
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)

    stem = SvgPath().M((x, y_top)).L(c.L()).d()
    corner = SvgPath().M(c.L()).A(c.r, 0, 0, c.B()).d()  # bend direction fixed
    base = SvgPath().M(c.B()).L((grid.x_tan(1), y_base)).d()
    return [stem, corner, base], []


def glyph_m(grid: Grid) -> GlyphRes2:
    c1 = grid.circle(1, BASE_ROW + 1)
    c2 = grid.circle(2, BASE_ROW + 1)

    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    x2 = grid.x_tan(2)

    y_base = baseline(grid)
    y_join = c1.cy

    stem0 = SvgPath().M((x0, y_base)).L((x0, y_join)).d()
    stem1 = SvgPath().M((x1, y_base)).L((x1, y_join)).d()
    stem2 = SvgPath().M((x2, y_base)).L((x2, y_join)).d()

    arch1 = SvgPath().M(c1.L()).A(c1.r, 0, 1, c1.T()).A(c1.r, 0, 1, c1.R()).d()
    arch2 = SvgPath().M(c2.L()).A(c2.r, 0, 1, c2.T()).A(c2.r, 0, 1, c2.R()).d()

    return [stem0, arch1, stem1, arch2, stem2], []


def glyph_n(grid: Grid) -> GlyphRes2:
    c = grid.circle(1, BASE_ROW + 1)
    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    y_base = baseline(grid)
    y_join = c.cy

    stem0 = SvgPath().M((x0, y_base)).L((x0, y_join)).d()
    stem1 = SvgPath().M((x1, y_base)).L((x1, y_join)).d()
    arch = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()

    return [stem0, arch, stem1], []


def glyph_o(grid: Grid) -> GlyphRes2:
    return [circle_full(grid.circle(1, BASE_ROW + 1))], []


def glyph_p(grid: Grid) -> GlyphRes2:
    # long descender stem but STOP at bowl join
    bowl = grid.circle(1, BASE_ROW + 1)
    x = grid.x_tan(0)
    y_join = bowl.cy
    y_bot = descender_bottom(grid)
    stem = SvgPath().M((x, y_bot)).L((x, y_join)).d()
    return [stem, circle_full(bowl)], []

def glyph_P(grid: Grid):
    # P = high left stem + top half-bowl connected with two horizontals
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)

    cu = grid.circle(1, BASE_ROW)  # use the upper circle for the bowl
    y_mid = cu.B()[1]              # bottom of the half-circle (middle join)

    # left stem (high)
    stem = SvgPath().M((xL, baseline(grid))).L((xL, y_top)).d()

    # top horizontal: left stem -> bowl start
    top_bar = SvgPath().M((xL, y_top)).L(cu.T()).d()

    # right half-circle: T -> R -> B
    bowl = (
        SvgPath()
        .M(cu.T())
        .A(cu.r, 0, 1, cu.R())
        .A(cu.r, 0, 1, cu.B())
        .d()
    )

    # middle horizontal: bowl end -> left stem
    mid_bar = SvgPath().M(cu.B()).L((xL, y_mid)).d()

    return [stem, top_bar, bowl, mid_bar], []

def glyph_q(grid: Grid) -> GlyphRes2:
    # mirrored p: long descender stem on the right, STOP at bowl join
    bowl = grid.circle(1, BASE_ROW + 1)
    x = grid.x_tan(1)
    y_join = bowl.cy
    y_bot = descender_bottom(grid)
    stem = SvgPath().M((x, y_bot)).L((x, y_join)).d()
    return [stem, circle_full(bowl)], []


def glyph_r(grid: Grid) -> GlyphRes2:
    c = grid.circle(1, BASE_ROW + 1)
    x = grid.x_tan(0)
    y_base = baseline(grid)
    y_join = c.cy
    stem = SvgPath().M((x, y_base)).L((x, y_join)).d()
    shoulder = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).d()
    return [stem, shoulder], []

def glyph_R(grid: Grid):
    # R = P + K-like bendy bottom-right leg
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)

    cu = grid.circle(1, BASE_ROW)        # upper circle for P-bowl
    cl = grid.circle(1, BASE_ROW + 1)    # lower circle for the leg

    # --- P part ---
    stem = SvgPath().M((xL, y_base)).L((xL, y_top)).d()
    top_bar = SvgPath().M((xL, y_top)).L(cu.T()).d()

    bowl = (
        SvgPath()
        .M(cu.T())
        .A(cu.r, 0, 1, cu.R())
        .A(cu.r, 0, 1, cu.B())
        .d()
    )

    mid_bar = SvgPath().M(cu.B()).L((xL, cu.B()[1])).d()

    # --- K-like bendy leg ---
    # start at the P “middle join” (cu.B), curve into the lower circle’s right side,
    # then drop vertically to baseline (like the k right stem behavior).
    leg_arc = SvgPath().M(cu.B()).A(cl.r, 0, 1, cl.R()).d()
    leg_stem = SvgPath().M(cl.R()).L((cl.R()[0], y_base)).d()

    return [stem, top_bar, bowl, mid_bar, leg_arc, leg_stem], []

def glyph_S(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    p = SvgPath().M(cu.R())
    p.A(cu.r, 0, 0, cu.T()).A(cu.r, 0, 0, cu.L())
    p.A(cu.r, 0, 0, G)
    p.A(cl.r, 0, 1, cl.R()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.L())
    return [p.d()], [("G", G)]


def glyph_t(grid: Grid) -> GlyphRes2:
    # final t: long left stem, middle arc only, bottom bowl
    x = grid.x_tan(0)
    y_top = ascender_top(grid)

    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    stem = SvgPath().M((x, y_top)).L(cl.L()).d()

    arm = SvgPath().M(cu.L()).A(cu.r, 0, 0, cu.B()).d()

    bowl = (
        SvgPath()
        .M(cl.L())
        .A(cl.r, 0, 0, cl.B())
        .A(cl.r, 0, 0, cl.R())
        .d()
    )

    return [stem, arm, bowl], []


def glyph_U(grid: Grid):
    # U = two long stems + bottom half-circle.
    # Stems stop exactly at the arc endpoints.
    c = grid.circle(1, BASE_ROW + 1)  # bottom bowl circle

    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    y_top = ascender_top(grid)

    stemL = SvgPath().M((xL, y_top)).L(c.L()).d()
    stemR = SvgPath().M((xR, y_top)).L(c.R()).d()

    # bottom half circle: R -> B -> L (SWEEP flipped)
    bowl = (
        SvgPath()
        .M(c.R())
        .A(c.r, 0, 1, c.B())
        .A(c.r, 0, 1, c.L())
        .d()
    )

    return [stemL, stemR, bowl], []

def glyph_u(grid: Grid) -> GlyphRes2:
    # u = half a w: stems go UP from join to x-height, single bowl
    y_top = xheight_top(grid)
    c = grid.circle(1, BASE_ROW + 1)
    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    y_join = c.cy

    stemL = SvgPath().M((x0, y_join)).L((x0, y_top)).d()
    stemR = SvgPath().M((x1, y_join)).L((x1, y_top)).d()

    bowl = SvgPath().M(c.L()).A(c.r, 0, 0, c.B()).A(c.r, 0, 0, c.R()).d()
    return [stemL, bowl, stemR], []


def glyph_v(grid: Grid) -> GlyphRes2:
    topL = (grid.x_tan(0), xheight_top(grid))
    topR = (grid.x_tan(1), xheight_top(grid))
    bottom = (grid.circle(1, BASE_ROW + 1).cx, baseline(grid))
    return [SvgPath().M(topL).L(bottom).L(topR).d()], []


def glyph_w(grid: Grid) -> GlyphRes2:
    # wide w: stems go UP (join->xheight)
    y_top = xheight_top(grid)
    c1 = grid.circle(1, BASE_ROW + 1)
    c2 = grid.circle(2, BASE_ROW + 1)

    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    x2 = grid.x_tan(2)

    y_join = c1.cy

    leg0 = SvgPath().M((x0, y_join)).L((x0, y_top)).d()
    leg1 = SvgPath().M((x1, y_join)).L((x1, y_top)).d()
    leg2 = SvgPath().M((x2, y_join)).L((x2, y_top)).d()

    bowl1 = SvgPath().M(c1.L()).A(c1.r, 0, 0, c1.B()).A(c1.r, 0, 0, c1.R()).d()
    bowl2 = SvgPath().M(c2.L()).A(c2.r, 0, 0, c2.B()).A(c2.r, 0, 0, c2.R()).d()

    return [leg0, bowl1, leg1, bowl2, leg2], []

def glyph_W(grid: Grid):
    # W = double U (wide glyph): three long stems + two bottom bowls
    # Stems stop exactly at arc endpoints.
    c1 = grid.circle(1, BASE_ROW + 1)  # left bowl circle
    c2 = grid.circle(2, BASE_ROW + 1)  # right bowl circle

    x0 = grid.x_tan(0)  # left stem
    x1 = grid.x_tan(1)  # middle stem (shared)
    x2 = grid.x_tan(2)  # right stem

    y_top = ascender_top(grid)

    # stems stop at the arc endpoints (at y = c*.cy)
    stemL = SvgPath().M((x0, y_top)).L(c1.L()).d()  # to (x0, c1.cy)
    stemM = SvgPath().M((x1, y_top)).L((x1, c1.cy)).d()
    stemR = SvgPath().M((x2, y_top)).L(c2.R()).d()  # to (x2, c2.cy)

    # bowls: bottom halves (SWEEP=1 to match fixed U)
    bowl1 = SvgPath().M(c1.R()).A(c1.r, 0, 1, c1.B()).A(c1.r, 0, 1, c1.L()).d()
    bowl2 = SvgPath().M(c2.R()).A(c2.r, 0, 1, c2.B()).A(c2.r, 0, 1, c2.L()).d()

    return [stemL, stemM, stemR, bowl1, bowl2], []

def glyph_x(grid: Grid) -> GlyphRes2:
    # two half circles touching in the middle
    c = grid.circle(1, BASE_ROW + 1)
    left_cx = c.cx - c.r
    right_cx = c.cx + c.r
    cy = c.cy
    r = c.r

    left_top = (left_cx, cy - r)
    left_bot = (left_cx, cy + r)
    left_half = SvgPath().M(left_top).A(r, 0, 1, left_bot).d()

    right_top = (right_cx, cy - r)
    right_bot = (right_cx, cy + r)
    right_half = SvgPath().M(right_top).A(r, 0, 0, right_bot).d()

    return [left_half, right_half], []


def glyph_y(grid: Grid) -> GlyphRes2:
    # y = u + long stem (descender) on the right
    c = grid.circle(1, BASE_ROW + 1)
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)

    y_top = xheight_top(grid)
    y_join = c.cy
    y_desc = descender_bottom(grid)

    stemL = SvgPath().M((xL, y_join)).L((xL, y_top)).d()
    stemR_up = SvgPath().M((xR, y_join)).L((xR, y_top)).d()
    bowl = SvgPath().M(c.L()).A(c.r, 0, 0, c.B()).A(c.r, 0, 0, c.R()).d()
    stemR_down = SvgPath().M((xR, y_join)).L((xR, y_desc)).d()

    return [stemL, bowl, stemR_up, stemR_down], []


def glyph_Z(grid: Grid) -> GlyphRes2:
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = xheight_top(grid)
    yB = baseline(grid)
    top = SvgPath().M((xL, yT)).L((xR, yT)).d()
    diag = SvgPath().M((xR, yT)).L((xL, yB)).d()
    bot = SvgPath().M((xL, yB)).L((xR, yB)).d()
    return [top, diag, bot], []


# -----------------------------
# Dot / quotes / question mark (DOT radius=1, real circle elements)
# -----------------------------
def glyph_dot(grid: Grid):
    # . = dot centered between four grid circles in the NARROW viewbox (one unit left)
    cx = grid.x_tan(0)
    cy = baseline(grid)
    return [], [], [(cx, cy, DOT_R)]

def glyph_quote_low(grid: Grid):
    # „ = two bottom-right quarter arcs of the MAIN grid circles (large)
    c1 = grid.circle(1, GRID_ROWS - 1)
    c2 = grid.circle(2, GRID_ROWS - 1)

    q1 = SvgPath().M(c1.B()).A(c1.r, 0, 0, c1.R()).d()  # bottom -> right
    q2 = SvgPath().M(c2.B()).A(c2.r, 0, 0, c2.R()).d()

    return [q1, q2], []

def glyph_quote_high(grid: Grid):
    # ” = two bottom-right quarter arcs of the MAIN grid circles,
    # positioned one circle (one row) higher than BASE_ROW.
    row = BASE_ROW - 1  # == 0

    c1 = grid.circle(1, row)
    c2 = grid.circle(2, row)

    q1 = SvgPath().M(c1.B()).A(c1.r, 0, 0, c1.R()).d()  # bottom -> right
    q2 = SvgPath().M(c2.B()).A(c2.r, 0, 0, c2.R()).d()

    return [q1, q2], []

def glyph_qmark(grid: Grid) -> GlyphRes3:
    # Hook closely matching the reference draft:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    # point on cl circle around ~200° (matches the reference "tail" point)
    ang = math.radians(200.0)
    tail = (cl.cx + cl.r * math.cos(ang), cl.cy + cl.r * math.sin(ang))

    hook = (
        SvgPath()
        .M(cu.L())
        .A(cu.r, 0, 1, cu.T())
        .A(cu.r, 0, 1, cu.R())
        .A(cu.r, 0, 1, cu.B())
        .A(cu.r, 0, 0, tail)
        .d()
    )

    # Dot: real circle at (x_tan(0), baseline)
    dot = (grid.x_tan(0), baseline(grid), DOT_R)
    return [hook], [], [dot]


# -----------------------------
# Digits (adjusted)
# -----------------------------
def digit_0(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    p = SvgPath().M(cu.L())
    p.A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.R())
    p.L(cl.R())
    p.A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.L())
    p.L(cu.L())
    return [p.d()], []

def digit_1(grid: Grid):
    # 1 = vertical stem + quarter-circle arc to the top-right
    # (cap is ONE circle LOWER than before)
    cap = grid.circle(1, BASE_ROW)  # ✅ was BASE_ROW-1

    x = cap.R()[0]
    y_base = baseline(grid)

    # stem goes up to where it meets the arc (cap.R)
    stem = SvgPath().M((x, y_base)).L(cap.R()).d()

    # quarter arc at top-right: R -> T
    arc = SvgPath().M(cap.R()).A(cap.r, 0, 0, cap.T()).d()

    return [stem, arc], []

def digit_2(grid: Grid) -> GlyphRes2:
    # connected like S via tangency G
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    xR = grid.x_tan(1)
    y_base = baseline(grid)

    p = SvgPath().M(cu.L())
    p.A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.R()).A(cu.r, 0, 1, G)
    p.A(cl.r, 0, 0, cl.L()).A(cl.r, 0, 0, cl.B())
    p.L((xR, y_base))
    return [p.d()], [("G", G)]


def digit_3(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    p = SvgPath().M(cu.L())
    p.A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.R()).A(cu.r, 0, 1, G)
    p.A(cl.r, 0, 1, cl.R()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.L())
    return [p.d()], []


def digit_4(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    xL = grid.x_tan(0)
    xR = grid.x_tan(1)

    y_top = ascender_top(grid)
    y_base = baseline(grid)

    stemL = SvgPath().M((xL, y_top)).L(cu.L()).d()

    connector = (
        SvgPath()
        .M(cu.L())
        .A(cu.r, 0, 0, cu.B())   # bottom-left bend direction fixed
        .A(cl.r, 0, 1, cl.R())
        .d()
    )

    stemR = SvgPath().M((xR, y_top)).L((xR, y_base)).d()
    return [stemL, connector, stemR], []


def digit_5(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    xR = grid.x_tan(1)
    y_top = cu.T()[1]

    top_bar = SvgPath().M(cu.T()).L((xR, y_top)).d()

    upper_c = (
        SvgPath()
        .M(cu.T())
        .A(cu.r, 0, 0, cu.L())
        .A(cu.r, 0, 0, cu.B())
        .d()
    )

    # bottom: 3/4 circle (T -> R -> B -> L)
    lower_c = (
        SvgPath()
        .M(cl.T())
        .A(cl.r, 0, 1, cl.R())
        .A(cl.r, 0, 1, cl.B())
        .A(cl.r, 0, 1, cl.L())
        .d()
    )

    return [top_bar, upper_c, lower_c], []


def digit_6(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    top = (
        SvgPath()
        .M(cu.L())
        .A(cu.r, 0, 1, cu.T())
        .A(cu.r, 0, 1, cu.R())
        .d()
    )

    stem = SvgPath().M(cu.L()).L(cl.L()).d()
    loop = circle_full(cl)
    return [top, stem, loop], []


def digit_7(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    y_top = cu.T()[1]
    y_base = baseline(grid)

    top = SvgPath().M((xL, y_top)).L(cu.T()).d()
    corner = SvgPath().M(cu.T()).A(cu.r, 0, 1, cu.R()).d()
    stem = SvgPath().M((xR, cu.R()[1])).L((xR, y_base)).d()
    return [top, corner, stem], []


def digit_8(grid: Grid) -> GlyphRes2:
    return [circle_full(grid.circle(1, BASE_ROW)), circle_full(grid.circle(1, BASE_ROW + 1))], []


def digit_9(grid: Grid) -> GlyphRes2:
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)

    top = circle_full(cu)
    stem = SvgPath().M(cu.R()).L(cl.R()).d()

    bottom = (
        SvgPath()
        .M(cl.R())
        .A(cl.r, 0, 1, cl.B())
        .A(cl.r, 0, 1, cl.L())
        .d()
    )

    return [top, stem, bottom], []


# -----------------------------
# Uppercase extras
# -----------------------------
def glyph_H(grid: Grid):
    """
    H = like K, but the right side is a single straight stem (same as left stem).
    Keep K's middle structure (lower arch + upper quarter connector),
    and flip the quarter connector so it bends the other way.
    """
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)

    y_top = ascender_top(grid)
    y_base = baseline(grid)

    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    G = grid.tangency_vertical(1, BASE_ROW)

    # left stem (full height)
    stem_left = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    # K-like middle arch (lower circle)
    arch = SvgPath().M(cl.L()).A(cl.r, 0, 1, cl.T()).A(cl.r, 0, 1, cl.R()).d()

    # right stem: one straight stem, full height
    stem_right = SvgPath().M((xR, y_top)).L((xR, y_base)).d()

    return [stem_left, arch, stem_right], [("G", G)]

def glyph_K(grid: Grid) -> GlyphRes2:
    return glyph_k(grid)


def glyph_M(grid: Grid) -> GlyphRes2:
    # M = like m, but arches at TOP and stems stop at arches
    c1 = grid.circle(1, BASE_ROW)
    c2 = grid.circle(2, BASE_ROW)

    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    x2 = grid.x_tan(2)

    y_base = baseline(grid)
    y_join = c1.cy

    stem0 = SvgPath().M((x0, y_base)).L((x0, y_join)).d()
    stem1 = SvgPath().M((x1, y_base)).L((x1, y_join)).d()
    stem2 = SvgPath().M((x2, y_base)).L((x2, y_join)).d()

    arch1 = SvgPath().M(c1.L()).A(c1.r, 0, 1, c1.T()).A(c1.r, 0, 1, c1.R()).d()
    arch2 = SvgPath().M(c2.L()).A(c2.r, 0, 1, c2.T()).A(c2.r, 0, 1, c2.R()).d()

    return [stem0, arch1, stem1, arch2, stem2], []


def glyph_N(grid: Grid) -> GlyphRes2:
    # N = like n, but arch at TOP and stems stop at arch
    cu = grid.circle(1, BASE_ROW)

    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)

    y_base = baseline(grid)
    y_join = cu.cy

    stem0 = SvgPath().M((x0, y_base)).L((x0, y_join)).d()
    stem1 = SvgPath().M((x1, y_base)).L((x1, y_join)).d()
    arch = SvgPath().M(cu.L()).A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.R()).d()

    return [stem0, arch, stem1], []


def glyph_X(grid: Grid) -> GlyphRes2:
    # X = lowercase u on top of lowercase n
    # top-u in ascender area
    cu = grid.circle(1, BASE_ROW)
    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    y_top_u = ascender_top(grid)
    y_join_u = cu.cy

    stem_u_L = SvgPath().M((x0, y_join_u)).L((x0, y_top_u)).d()
    stem_u_R = SvgPath().M((x1, y_join_u)).L((x1, y_top_u)).d()
    bowl_u = SvgPath().M(cu.L()).A(cu.r, 0, 0, cu.B()).A(cu.r, 0, 0, cu.R()).d()

    # bottom-n in x-height area
    cn = grid.circle(1, BASE_ROW + 1)
    y_base = baseline(grid)
    y_join_n = cn.cy

    stem_n_L = SvgPath().M((x0, y_base)).L((x0, y_join_n)).d()
    stem_n_R = SvgPath().M((x1, y_base)).L((x1, y_join_n)).d()
    arch_n = SvgPath().M(cn.L()).A(cn.r, 0, 1, cn.T()).A(cn.r, 0, 1, cn.R()).d()

    return [stem_u_L, bowl_u, stem_u_R, stem_n_L, arch_n, stem_n_R], []


# -----------------------------
# Builder map (must cover all requested chars)
# -----------------------------
BUILDERS: Dict[str, GlyphFn] = {
    # locked
    "A": glyph_A,
    "a": glyph_a,
    "b": glyph_b,
    "C": glyph_C,
    "E": glyph_E,
    "G": glyph_G,
    "k": glyph_k,
    "R": glyph_R,

    # rest
    "c": glyph_c,
    "d": glyph_d,
    "e": glyph_E,     # temp
    "f": glyph_f,
    "g": glyph_g,
    "h": glyph_h,
    "i": glyph_i,
    "m": glyph_m,
    "n": glyph_n,
    "o": glyph_o,
    "p": glyph_p,
    "q": glyph_q,
    "r": glyph_r,
    "S": glyph_S,
    "t": glyph_t,
    "U": glyph_U,
    "u": glyph_u,
    "v": glyph_v,
    "w": glyph_w,
    "x": glyph_x,
    "y": glyph_y,
    "Z": glyph_Z,
    ".": glyph_dot,
    "„": glyph_quote_low,
    "”": glyph_quote_high,
    "?": glyph_qmark,

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

    # extra uppercase set
    "H": glyph_H,
    "I": glyph_I,
    "J": glyph_J,
    "L": glyph_L,
    "K": glyph_K,
    "M": glyph_M,
    "N": glyph_N,
    "P": glyph_P,
    "W": glyph_W,
    "X": glyph_X,
}


# -----------------------------
# SVG rendering (variable width)
# -----------------------------
def glyph_view_width(ch: str) -> int:
    if ch in ("I", ".", "i", "j"):
        return VIEW_W_NARROW
    return VIEW_W_WIDE if ch in ("m", "M", "w", "W") else VIEW_W_DEFAULT

def glyph_grid_cols(ch: str) -> int:
    if ch in ("I", ".", "i", "j"):
        return GRID_COLS_NARROW
    return GRID_COLS_WIDE if ch in ("m", "M", "w", "W") else GRID_COLS_DEFAULT

def svg_glyph_doc(
    glyph_name: str,
    codepoint: int,
    paths_d: List[str],
    debug: bool,
    anchors: List[Tuple[str, Tuple[float, float]]],
    dots: List[Dot],
    grid: Grid,
    view_w: int,
    grid_cols: int,
) -> str:
    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {VIEW_H}" width="{view_w}" height="{VIEW_H}">')
    out.append(f'<desc>glyph: {glyph_name} U+{codepoint:04X}</desc>')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')

    if debug:
        out.append(f'<g fill="none" stroke="#bdbdbd" stroke-width="{fmt(GRID_STROKE)}" opacity="{fmt(GRID_OPACITY)}">')
        for row in range(0, GRID_ROWS):
            for col in range(0, grid_cols):
                c = grid.circle(col, row)
                out.append(f'<circle cx="{fmt(c.cx)}" cy="{fmt(c.cy)}" r="{fmt(c.r)}" />')
        out.append('</g>')

    out.append(
        f'<g fill="none" stroke="#000" stroke-width="{fmt(STROKE)}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    )
    for d in paths_d:
        out.append(f'<path d="{d}" />')
    out.append('</g>')

    # ✅ Dots: actual circles, filled, with the same stroke applied
    if dots:
        out.append(f'<g fill="#000" stroke="#000" stroke-width="{fmt(STROKE)}">')
        for (cx, cy, rr) in dots:
            out.append(f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(rr)}" />')
        out.append('</g>')

    if debug and anchors:
        out.append('<g fill="#1e6bff" stroke="none">')
        for _, (x, y) in anchors:
            out.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(ANCHOR_R)}" />')
        out.append('</g>')

    out.append('</svg>')
    out.append('')
    return "\n".join(out)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="src", help="Output directory (default: src)")
    ap.add_argument("--debug", action="store_true", help="Overlay grid + anchor dots")
    args = ap.parse_args()

    out_dir = FSPath(args.out_dir)
    grid = Grid(r=R, x0=X0, y0=Y0)

    missing: List[str] = []
    written = 0

    for ch in sorted(BUILDERS.keys(), key=lambda c: ord(c)):
        fn = BUILDERS[ch]

        res = fn(grid)
        if isinstance(res, tuple) and len(res) == 3:
            paths_d, anchors, dots = res
        else:
            paths_d, anchors = res
            dots = []

        vw = glyph_view_width(ch)
        cols = glyph_grid_cols(ch)

        svg = svg_glyph_doc(
            glyph_name=ch,
            codepoint=ord(ch),
            paths_d=paths_d,
            debug=args.debug,
            anchors=anchors,
            dots=dots,
            grid=grid,
            view_w=vw,
            grid_cols=cols,
        )

        out_name = f"character-u{ord(ch):04x}.svg"
        write_text_lf(out_dir / out_name, svg)
        written += 1
        print(f"✓ {ch} -> {out_name}")

    print(f"\nDone. Wrote {written} glyph(s) into {out_dir.resolve()}")

    if missing:
        print("No builder yet for:", "".join(missing))


if __name__ == "__main__":
    main()
