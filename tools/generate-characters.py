#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path as FSPath
from typing import Callable, Dict, List, Tuple


# -----------------------------
# Desired sheet rows (exact)
# -----------------------------
SHEET_ROWS = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]
REQUESTED = "".join(SHEET_ROWS)


# -----------------------------
# Grid / canvas (tight + correct)
# -----------------------------
R = 40.0
DX = 2.0 * R
DY = 2.0 * R

# Make a 3-col wide grid patch fully fit in 0..240:
# col centers: 40, 120, 200 => circles span 0..240
# row centers: 40, 120, 200 => circles span 0..240
X0 = 40.0
Y0 = 40.0

VIEW_W = int(3 * DX)  # 240
VIEW_H = int(3 * DY)  # 240

STROKE = 9.0

# Sheet spacing: exactly one radius between glyph boxes
SHEET_GAP = R

# Debug overlay
GRID_OPACITY = 0.35
GRID_STROKE = 1.0
ANCHOR_R = 3.5


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
# Geometry
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
        # tangent line between col_left and col_left+1
        return self.x0 + col_left * DX + self.r

    def y_tan(self, row_top: int) -> float:
        # tangent line between row_top and row_top+1
        return self.y0 + row_top * DY + self.r

    def tangency_vertical(self, col: int, row_top: int) -> Tuple[float, float]:
        # point where (col,row_top) and (col,row_top+1) touch
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
    # R -> B -> L -> T -> R (4 quarter arcs)
    p = SvgPath().M(c.R())
    p.A(c.r, 0, sweep, c.B())
    p.A(c.r, 0, sweep, c.L())
    p.A(c.r, 0, sweep, c.T())
    p.A(c.r, 0, sweep, c.R())
    return p.d()


def arc_LTB(c: Circle, sweep: int = 1) -> str:
    """Left->Top->Right (upper half-ish, but using two quarter arcs)"""
    return SvgPath().M(c.L()).A(c.r, 0, sweep, c.T()).A(c.r, 0, sweep, c.R()).d()


def arc_RBL(c: Circle, sweep: int = 1) -> str:
    """Right->Bottom->Left (lower half-ish)"""
    return SvgPath().M(c.R()).A(c.r, 0, sweep, c.B()).A(c.r, 0, sweep, c.L()).d()


# -----------------------------
# Metrics (derived from grid)
# -----------------------------
def ascender_top(grid: Grid) -> float:
    # top tangent of row0 circles
    return grid.circle(0, 0).T()[1]

def xheight_top(grid: Grid) -> float:
    # top tangent of row1 circles
    return grid.circle(0, 1).T()[1]

def baseline(grid: Grid) -> float:
    # bottom tangent of row1 circles
    return grid.circle(0, 1).B()[1]


# -----------------------------
# Glyph builders
# -----------------------------
GlyphFn = Callable[[Grid], Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]]


def glyph_a(grid: Grid):
    # Two stacked circles at col=1 rows 0 and 1
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)

    A = cu.L()
    B = cu.T()
    C = cu.R()
    D = cl.R()
    E = cl.B()
    F = cl.L()
    G = grid.tangency_vertical(1, 0)

    outer = SvgPath().M(A).A(cu.r, 0, 1, B).A(cu.r, 0, 1, C).L(D).A(cl.r, 0, 1, E).A(cl.r, 0, 1, F).d()
    inner_GC = SvgPath().M(G).A(cu.r, 0, 0, C).d()  # correct bow direction
    inner_GF = SvgPath().M(G).A(cl.r, 0, 0, F).d()

    anchors = [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F), ("G", G)]
    return [outer, inner_GC, inner_GF], anchors


def glyph_A(grid: Grid):
    return glyph_a(grid)


def glyph_b(grid: Grid):
    # Validated: bowl full circle at (col=1,row=1); stem stops at bowl's left tangent
    bowl = grid.circle(1, 1)
    x_stem = grid.x_tan(0)            # between col0 and col1
    y_top = ascender_top(grid)        # row0 top tangent
    y_join = bowl.L()[1]              # bowl left tangent y (bowl.cy)

    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    bowl_path = circle_full(bowl)

    anchors = [("stemTop", (x_stem, y_top)), ("join", (x_stem, y_join))]
    return [stem, bowl_path], anchors


def glyph_c(grid: Grid):
    c = grid.circle(1, 1)
    # Open on the right: T -> L -> B (left side curve)
    path = SvgPath().M(c.T()).A(c.r, 0, 1, c.L()).A(c.r, 0, 1, c.B()).d()
    return [path], []


def glyph_C(grid: Grid):
    """
    Continuous C:
      Two stacked circles (col=1, rows 0 and 1)
      Start at upper R, go around upper top+left to tangency G,
      then continue around lower left+bottom to lower R.
    """
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    G = grid.tangency_vertical(1, 0)

    # upper: R -> T -> L -> G
    p = SvgPath().M(cu.R()).A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.L()).A(cu.r, 0, 1, G)
    # lower: G -> L -> B -> R
    p.A(cl.r, 0, 1, cl.L()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.R())
    return [p.d()], [("G", G)]


def glyph_G(grid: Grid):
    """
    G = C + bar
    Bar at midline of lower circle from right tangent line inward.
    """
    paths, anchors = glyph_C(grid)
    cl = grid.circle(1, 1)
    y = cl.cy
    xR = grid.x_tan(1)     # right tangent line between col1 and col2
    xIn = cl.cx            # inward to center (can tweak later)
    bar = SvgPath().M((xR, y)).L((xIn, y)).d()
    return paths + [bar], anchors


def glyph_d(grid: Grid):
    # mirror-ish first pass: bowl + stem on right tangent line
    bowl = grid.circle(1, 1)
    x_stem = grid.x_tan(1)
    y_top = ascender_top(grid)
    y_join = bowl.R()[1]
    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    return [stem, circle_full(bowl)], []


def glyph_E(grid: Grid):
    # epsilon-ish E: upper curve + lower curve + mid tick
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    G = grid.tangency_vertical(1, 0)

    upper = SvgPath().M(cu.R()).A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.L()).A(cu.r, 0, 1, G).d()
    lower = SvgPath().M(G).A(cl.r, 0, 1, cl.L()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.R()).d()

    tick = SvgPath().M(G).L((grid.x_tan(1), G[1])).d()
    return [upper, lower, tick], [("G", G)]


def glyph_f(grid: Grid):
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_bot = baseline(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    y_arm = grid.circle(1, 1).cy
    arm = SvgPath().M((x, y_arm)).L((grid.x_tan(1), y_arm)).d()
    return [stem, arm], []


def glyph_g(grid: Grid):
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    return [circle_full(cu), circle_full(cl)], []


def glyph_h(grid: Grid):
    """
    h: ascender stem + arch + right leg
    """
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    c = grid.circle(1, 1)
    arch = arc_LTB(c, sweep=1)  # L->T->R
    right_leg = SvgPath().M(c.R()).L((c.R()[0], y_base)).d()
    return [stem, arch, right_leg], []


def glyph_n(grid: Grid):
    """
    n: x-height stem + arch + right leg
    """
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    c = grid.circle(1, 1)
    arch = arc_LTB(c, sweep=1)
    right_leg = SvgPath().M(c.R()).L((c.R()[0], y_base)).d()
    return [stem, arch, right_leg], []


def glyph_m(grid: Grid):
    """
    m: x-height stem + two arches + two right legs
    Uses circles at col=1,row=1 and col=2,row=1
    """
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    c1 = grid.circle(1, 1)
    c2 = grid.circle(2, 1)

    arch1 = arc_LTB(c1, sweep=1)
    leg1 = SvgPath().M(c1.R()).L((c1.R()[0], y_base)).d()

    arch2 = arc_LTB(c2, sweep=1)
    leg2 = SvgPath().M(c2.R()).L((c2.R()[0], y_base)).d()

    return [stem, arch1, leg1, arch2, leg2], []


def glyph_r(grid: Grid):
    """
    r: x-height stem + small shoulder (quarter-ish) + short right leg (optional)
    We'll do stem + shoulder arc from L->T, then a small down tick.
    """
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    c = grid.circle(1, 1)
    shoulder = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).d()
    tick = SvgPath().M(c.T()).L((c.T()[0], c.cy)).d()
    return [stem, shoulder, tick], []


def glyph_o(grid: Grid):
    return [circle_full(grid.circle(1, 1))], []


def glyph_I(grid: Grid):
    x = grid.x_tan(0)
    return [SvgPath().M((x, ascender_top(grid))).L((x, baseline(grid))).d()], []


def glyph_J(grid: Grid):
    x = grid.x_tan(1)
    y_top = ascender_top(grid)
    c = grid.circle(1, 1)
    stem = SvgPath().M((x, y_top)).L((x, c.B()[1])).d()
    hook = SvgPath().M((x, c.B()[1])).A(c.r, 0, 1, c.L()).d()
    return [stem, hook], []


def glyph_k(grid: Grid):
    x = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)
    y_mid = grid.circle(1, 1).cy

    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    arm1 = SvgPath().M((x, y_mid)).L((grid.x_tan(1), grid.circle(1, 0).cy)).d()
    arm2 = SvgPath().M((x, y_mid)).L((grid.x_tan(1), grid.circle(1, 2).cy)).d()
    return [stem, arm1, arm2], []


def glyph_L(grid: Grid):
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    base = SvgPath().M((x, y_base)).L((grid.x_tan(1), y_base)).d()
    return [stem, base], []


def glyph_p(grid: Grid):
    x = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_bot = grid.circle(0, 2).B()[1]
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    bowl = circle_full(grid.circle(1, 1))
    return [stem, bowl], []


def glyph_q(grid: Grid):
    bowl = circle_full(grid.circle(1, 1))
    tail = SvgPath().M(grid.circle(1, 1).B()).L((grid.x_tan(1), grid.circle(2, 2).cy)).d()
    return [bowl, tail], []


def glyph_S(grid: Grid):
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    top = SvgPath().M(cu.R()).A(cu.r, 0, 0, cu.L()).d()
    bot = SvgPath().M(cl.L()).A(cl.r, 0, 0, cl.R()).d()
    return [top, bot], []


def glyph_t(grid: Grid):
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    y_bar = grid.circle(1, 1).cy
    bar = SvgPath().M((grid.x_tan(0), y_bar)).L((grid.x_tan(1), y_bar)).d()
    return [stem, bar], []


def glyph_U(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    c = grid.circle(1, 1)
    # stems stop at c.T (top of bowl)
    stemL = SvgPath().M((xL, y_top)).L((xL, c.T()[1])).d()
    stemR = SvgPath().M((xR, y_top)).L((xR, c.T()[1])).d()
    bowl = arc_RBL(c, sweep=1)  # R->B->L
    return [stemL, stemR, bowl], []


def glyph_u(grid: Grid):
    return glyph_U(grid)


def glyph_v(grid: Grid):
    # leaving as simple first pass
    topL = (grid.x_tan(0), xheight_top(grid))
    topR = (grid.x_tan(1), xheight_top(grid))
    bottom = (grid.circle(1, 1).cx, baseline(grid))
    return [SvgPath().M(topL).L(bottom).L(topR).d()], []


def glyph_w(grid: Grid):
    """
    w must be rounded: 3 legs + 2 bottom arcs (no zig-zag).
    Use bottom halves of circles at col=1,row=1 and col=2,row=1.
    """
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    x0 = grid.x_tan(0)   # left leg
    x1 = grid.x_tan(1)   # middle leg (also col1/col2 tangent)
    x2 = grid.x_tan(2)   # right leg (between col2 and col3) - but we only have 0..2 cols
    # In our 3-col patch, the rightmost vertical we can use is the RIGHT tangent of col2 circle:
    c2 = grid.circle(2, 1)
    x2 = c2.R()[0]       # = 240, still inside viewBox

    # legs
    leg0 = SvgPath().M((x0, y_top)).L((x0, y_base)).d()
    leg1 = SvgPath().M((x1, y_top)).L((x1, y_base)).d()
    leg2 = SvgPath().M((x2, y_top)).L((x2, y_base)).d()

    # bottom arcs (two bowls)
    c1 = grid.circle(1, 1)  # spans x0..x1
    c2 = grid.circle(2, 1)  # spans x1..x2

    bowl1 = arc_RBL(c1, sweep=1)  # R->B->L (connects x1->bottom->x0)
    bowl2 = arc_RBL(c2, sweep=1)  # R->B->L (connects x2->bottom->x1)

    # We want a continuous w: draw bowls left-to-right:
    # easiest: reverse bowl1 to L->B->R by using sweep=0 from L
    bowl1_lr = SvgPath().M(c1.L()).A(c1.r, 0, 0, c1.B()).A(c1.r, 0, 0, c1.R()).d()
    bowl2_lr = SvgPath().M(c2.L()).A(c2.r, 0, 0, c2.B()).A(c2.r, 0, 0, c2.R()).d()

    return [leg0, bowl1_lr, leg1, bowl2_lr, leg2], []


def glyph_x(grid: Grid):
    a1 = (grid.x_tan(0), xheight_top(grid))
    a2 = (grid.x_tan(1), baseline(grid))
    b1 = (grid.x_tan(1), xheight_top(grid))
    b2 = (grid.x_tan(0), baseline(grid))
    return [SvgPath().M(a1).L(a2).d(), SvgPath().M(b1).L(b2).d()], []


def glyph_y(grid: Grid):
    topL = (grid.x_tan(0), xheight_top(grid))
    topR = (grid.x_tan(1), xheight_top(grid))
    mid = (grid.circle(1, 1).cx, grid.circle(1, 1).cy)
    tail = (mid[0], grid.circle(1, 2).cy)
    return [SvgPath().M(topL).L(mid).L(topR).d(), SvgPath().M(mid).L(tail).d()], []


def glyph_Z(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = xheight_top(grid)
    yB = baseline(grid)
    top = SvgPath().M((xL, yT)).L((xR, yT)).d()
    diag = SvgPath().M((xR, yT)).L((xL, yB)).d()
    bot = SvgPath().M((xL, yB)).L((xR, yB)).d()
    return [top, diag, bot], []


def glyph_dot(grid: Grid):
    c = grid.circle(2, 2)
    rr = grid.r * 0.18
    dot = SvgPath().M((c.cx + rr, c.cy)).A(rr, 0, 1, (c.cx, c.cy + rr)).A(rr, 0, 1, (c.cx - rr, c.cy)).A(rr, 0, 1, (c.cx, c.cy - rr)).A(rr, 0, 1, (c.cx + rr, c.cy)).d()
    return [dot], []


def glyph_quote_low(grid: Grid):
    c1 = grid.circle(1, 2)
    c2 = grid.circle(2, 2)
    rr = grid.r * 0.16
    y = c1.cy + grid.r * 0.35
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []


def glyph_quote_high(grid: Grid):
    c1 = grid.circle(1, 0)
    c2 = grid.circle(2, 0)
    rr = grid.r * 0.16
    y = c1.cy - grid.r * 0.35
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []


def glyph_qmark(grid: Grid):
    c = grid.circle(1, 0)
    hook = SvgPath().M(c.T()).A(c.r, 0, 1, c.R()).A(c.r, 0, 1, c.B()).d()
    stem = SvgPath().M(c.B()).L((c.B()[0], grid.circle(1, 1).cy)).d()
    dot = glyph_dot(grid)[0]
    return [hook, stem, dot], []


# Digits (still first-pass but grid-correct)
def digit_0(grid: Grid): return glyph_o(grid)
def digit_1(grid: Grid):
    x = grid.x_tan(1)
    return [SvgPath().M((x, xheight_top(grid))).L((x, baseline(grid))).d()], []
def digit_2(grid: Grid):
    c = grid.circle(1, 1)
    top = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    diag = SvgPath().M(c.R()).L((grid.x_tan(0), baseline(grid))).d()
    base = SvgPath().M((grid.x_tan(0), baseline(grid))).L((grid.x_tan(1), baseline(grid))).d()
    return [top, diag, base], []
def digit_3(grid: Grid):
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    top = SvgPath().M(cu.T()).A(cu.r, 0, 1, cu.R()).A(cu.r, 0, 1, cu.B()).d()
    bot = SvgPath().M(cl.T()).A(cl.r, 0, 1, cl.R()).A(cl.r, 0, 1, cl.B()).d()
    return [top, bot], []
def digit_4(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = xheight_top(grid)
    yM = grid.circle(1, 1).cy
    yB = baseline(grid)
    diag = SvgPath().M((xR, yT)).L((xL, yM)).d()
    bar = SvgPath().M((xL, yM)).L((xR, yM)).d()
    stem = SvgPath().M((xR, yT)).L((xR, yB)).d()
    return [diag, bar, stem], []
def digit_5(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = xheight_top(grid)
    yM = grid.circle(1, 1).cy
    top = SvgPath().M((xR, yT)).L((xL, yT)).L((xL, yM)).d()
    bowl = SvgPath().M(grid.circle(1, 1).L()).A(grid.circle(1, 1).r, 0, 0, grid.circle(1, 1).R()).d()
    return [top, bowl], []
def digit_6(grid: Grid):
    c = grid.circle(1, 1)
    loop = circle_full(c)
    hook = SvgPath().M(c.T()).L((grid.x_tan(0), grid.circle(1, 0).cy)).d()
    return [loop, hook], []
def digit_7(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = xheight_top(grid)
    p = SvgPath().M((xL, yT)).L((xR, yT)).L((xL, baseline(grid))).d()
    return [p], []
def digit_8(grid: Grid):
    return [circle_full(grid.circle(1, 0)), circle_full(grid.circle(1, 1))], []
def digit_9(grid: Grid):
    c = grid.circle(1, 0)
    loop = circle_full(c)
    tail = SvgPath().M(c.B()).L((grid.x_tan(1), grid.circle(1, 1).cy)).d()
    return [loop, tail], []


# -----------------------------
# Builder map (only what you need)
# -----------------------------
BUILDERS: Dict[str, GlyphFn] = {
    "A": glyph_A,
    "a": glyph_a,
    "b": glyph_b,
    "C": glyph_C,
    "c": glyph_c,
    "d": glyph_d,
    "E": glyph_E,
    "e": glyph_E,  # (we'll refine later if needed)
    "f": glyph_f,
    "g": glyph_g,
    "h": glyph_h,
    "I": glyph_I,
    "J": glyph_J,
    "k": glyph_k,
    "L": glyph_L,
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
    "G": glyph_G,
    "H": glyph_h,  # placeholder
    "K": glyph_k,
    "M": glyph_m,
    "N": glyph_n,
    "X": glyph_x,
}


# -----------------------------
# SVG rendering
# -----------------------------
def svg_glyph_doc(paths_d: List[str], debug: bool, anchors: List[Tuple[str, Tuple[float, float]]], grid: Grid) -> str:
    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}" width="{VIEW_W}" height="{VIEW_H}">')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')

    if debug:
        # exact grid used by glyph math: 3x3 circles touching
        out.append(f'<g fill="none" stroke="#bdbdbd" stroke-width="{fmt(GRID_STROKE)}" opacity="{fmt(GRID_OPACITY)}">')
        for row in range(0, 3):
            for col in range(0, 3):
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

    if debug and anchors:
        out.append('<g fill="#1e6bff" stroke="none">')
        for _, (x, y) in anchors:
            out.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(ANCHOR_R)}" />')
        out.append('</g>')

    out.append('</svg>')
    out.append('')
    return "\n".join(out)


def svg_sheet_doc(rows: List[str], glyph_paths: Dict[str, List[str]]) -> str:
    cell_w = float(VIEW_W)
    cell_h = float(VIEW_H)

    max_cols = max(len(r) for r in rows)
    sheet_w = max_cols * cell_w + (max_cols - 1) * SHEET_GAP
    sheet_h = len(rows) * cell_h + (len(rows) - 1) * SHEET_GAP

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(sheet_w)}" height="{fmt(sheet_h)}" viewBox="0 0 {fmt(sheet_w)} {fmt(sheet_h)}">')
    out.append(f'<rect x="0" y="0" width="{fmt(sheet_w)}" height="{fmt(sheet_h)}" fill="#fff"/>')

    out.append(
        f'<g fill="none" stroke="#000" stroke-width="{fmt(STROKE)}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    )

    y = 0.0
    for row in rows:
        x = 0.0
        for ch in row:
            parts = glyph_paths.get(ch)
            if parts:
                out.append(f'<g transform="translate({fmt(x)} {fmt(y)})">')
                out.extend(parts)
                out.append('</g>')
            x += cell_w + SHEET_GAP
        y += cell_h + SHEET_GAP

    out.append('</g>')
    out.append('</svg>')
    out.append('')
    return "\n".join(out)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="src", help="Output directory for glyphs + sheet.svg (default: src)")
    ap.add_argument("--debug", action="store_true", help="Include grid overlay in each glyph SVG")
    args = ap.parse_args()

    out_dir = FSPath(args.out_dir)
    grid = Grid(r=R, x0=X0, y0=Y0)

    chars = uniq(REQUESTED)
    missing: List[str] = []
    glyph_paths_for_sheet: Dict[str, List[str]] = {}

    written = 0
    for ch in chars:
        fn = BUILDERS.get(ch)
        if fn is None:
            missing.append(ch)
            continue

        paths_d, anchors = fn(grid)

        # write glyph file
        doc = svg_glyph_doc(paths_d, debug=args.debug, anchors=anchors, grid=grid)
        out_name = f"character-u{ord(ch):04x}.svg"
        write_text_lf(out_dir / out_name, doc)
        written += 1
        print(f"✓ {ch} -> {out_name}")

        # for sheet (no debug overlay)
        glyph_paths_for_sheet[ch] = [f'<path d="{d}" />' for d in paths_d]

    # write sheet into SAME dir
    sheet_svg = svg_sheet_doc(SHEET_ROWS, glyph_paths_for_sheet)
    write_text_lf(out_dir / "sheet.svg", sheet_svg)
    print(f"\n✓ sheet -> {(out_dir / 'sheet.svg').resolve()}")

    print(f"\nDone. Wrote {written} glyph(s) into {out_dir.resolve()}")
    if missing:
        print("No builder yet for:", "".join(missing))


if __name__ == "__main__":
    main()
