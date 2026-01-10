#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path as FSPath
from typing import Callable, Dict, List, Tuple


# -----------------------------
# Character set (exact)
# -----------------------------
SHEET_ROWS = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]
REQUESTED = "".join(SHEET_ROWS)


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
GRID_COLS_WIDE = 4       # for m / M

BASE_ROW = 1             # glyphs start at row 1 (row 0 is top margin)

VIEW_H = int(GRID_ROWS * DY)        # 320
VIEW_W_DEFAULT = int(GRID_COLS_DEFAULT * DX)  # 240
VIEW_W_WIDE = int(GRID_COLS_WIDE * DX)        # 320

STROKE = 9.0

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
GlyphFn = Callable[[Grid], Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]]


# ---- LOCKED / FIXED GLYPHS ----

def glyph_a(grid: Grid):
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


def glyph_A(grid: Grid):
    return glyph_a(grid)


def glyph_b(grid: Grid):
    # bowl at (1, BASE_ROW+1), stem on x_tan(0) down to bowl left tangent
    bowl = grid.circle(1, BASE_ROW + 1)
    x_stem = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_join = bowl.L()[1]

    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    bowl_path = circle_full(bowl)
    return [stem, bowl_path], [("stemTop", (x_stem, y_top)), ("join", (x_stem, y_join))]


def glyph_C(grid: Grid):
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


def glyph_E(grid: Grid):
    """
    Fixed E = what you said my earlier “C” looked like:
    continuous wrap across two stacked circles through tangency,
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
    # G = fixed C + bar (connected at lower right)
    paths, _ = glyph_C(grid)
    cl = grid.circle(1, BASE_ROW + 1)
    BR = cl.R()
    bar_end = (cl.cx, cl.cy)
    bar = SvgPath().M(BR).L(bar_end).d()
    return paths + [bar], [("barEnd", bar_end)]


def glyph_k(grid: Grid):
    """
    Fixed k (as last agreed):
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


# ---- OTHER GLYPHS (grid-faithful placeholders; not “locked” yet) ----

def glyph_c(grid: Grid):
    c = grid.circle(1, BASE_ROW + 1)
    path = SvgPath().M(c.T()).A(c.r, 0, 1, c.L()).A(c.r, 0, 1, c.B()).d()
    return [path], []

def glyph_d(grid: Grid):
    bowl = grid.circle(1, BASE_ROW + 1)
    x_stem = grid.x_tan(1)
    y_top = ascender_top(grid)
    y_join = bowl.R()[1]
    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    return [stem, circle_full(bowl)], []

def glyph_f(grid: Grid):
    # Match reference:
    # - left stem (ascender -> baseline)
    # - top half-arch on upper circle (L->T->R)
    # - mid hook: lower circle quarter-arch (L->T) then short vertical down to center

    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)

    cu = grid.circle(1, BASE_ROW)         # upper circle
    cl = grid.circle(1, BASE_ROW + 1)     # lower circle

    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()

    # top arch: starts at cu.L which lies exactly on the stem x
    top_arch = (
        SvgPath()
        .M(cu.L())
        .A(cu.r, 0, 1, cu.T())
        .A(cu.r, 0, 1, cu.R())
        .d()
    )

    # mid hook: from cl.L (on the stem) up to cl.T, then down a bit (to center)
    mid_hook = (
        SvgPath()
        .M(cl.L())
        .A(cl.r, 0, 1, cl.T())
        .L((cl.cx, cl.cy))
        .d()
    )

    return [stem, top_arch, mid_hook], []

def glyph_g(grid: Grid):
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    return [circle_full(cu), circle_full(cl)], []

def glyph_h(grid: Grid):
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()
    arch = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    leg = SvgPath().M(c.R()).L((c.R()[0], y_base)).d()
    return [stem, arch, leg], []

def glyph_I(grid: Grid):
    x = grid.x_tan(0)
    return [SvgPath().M((x, ascender_top(grid))).L((x, baseline(grid))).d()], []

def glyph_J(grid: Grid):
    x = grid.x_tan(1)
    y_top = ascender_top(grid)

    c = grid.circle(1, BASE_ROW + 1)

    # stem drops to the circle's RIGHT point so the hook arc starts on-circle
    stem = SvgPath().M((x, y_top)).L(c.R()).d()

    # hook: right -> bottom -> left
    hook = SvgPath().M(c.R()).A(c.r, 0, 1, c.B()).A(c.r, 0, 1, c.L()).d()

    return [stem, hook], []

def glyph_L(grid: Grid):
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    base = SvgPath().M((x, y_base)).L((grid.x_tan(1), y_base)).d()
    return [stem, base], []

def glyph_m(grid: Grid):
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)
    c1 = grid.circle(1, BASE_ROW + 1)
    c2 = grid.circle(2, BASE_ROW + 1)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()

    arch1 = SvgPath().M(c1.L()).A(c1.r, 0, 1, c1.T()).A(c1.r, 0, 1, c1.R()).d()
    leg1 = SvgPath().M(c1.R()).L((c1.R()[0], y_base)).d()

    arch2 = SvgPath().M(c2.L()).A(c2.r, 0, 1, c2.T()).A(c2.r, 0, 1, c2.R()).d()
    leg2 = SvgPath().M(c2.R()).L((c2.R()[0], y_base)).d()

    return [stem, arch1, leg1, arch2, leg2], []

def glyph_n(grid: Grid):
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()
    arch = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    leg = SvgPath().M(c.R()).L((c.R()[0], y_base)).d()
    return [stem, arch, leg], []

def glyph_o(grid: Grid):
    return [circle_full(grid.circle(1, BASE_ROW + 1))], []

def glyph_p(grid: Grid):
    x = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_bot = descender_bottom(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    bowl = circle_full(grid.circle(1, BASE_ROW + 1))
    return [stem, bowl], []

def glyph_q(grid: Grid):
    bowl = circle_full(grid.circle(1, BASE_ROW + 1))
    c = grid.circle(1, BASE_ROW + 1)
    tail = SvgPath().M(c.B()).L((grid.x_tan(1), descender_bottom(grid) - R)).d()
    return [bowl, tail], []

def glyph_r(grid: Grid):
    xL = grid.x_tan(0)
    y_top = xheight_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()
    shoulder = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).d()
    return [stem, shoulder], []

def glyph_S(grid: Grid):
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    top = SvgPath().M(cu.R()).A(cu.r, 0, 0, cu.L()).d()
    bot = SvgPath().M(cl.L()).A(cl.r, 0, 0, cl.R()).d()
    return [top, bot], []

def glyph_t(grid: Grid):
    x = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    y_bar = grid.circle(1, BASE_ROW + 1).cy
    bar = SvgPath().M((grid.x_tan(0), y_bar)).L((grid.x_tan(1), y_bar)).d()
    return [stem, bar], []

def glyph_U(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    y_top = xheight_top(grid)
    y_base = baseline(grid)
    c = grid.circle(1, BASE_ROW + 1)
    stemL = SvgPath().M((xL, y_top)).L((xL, c.T()[1])).d()
    stemR = SvgPath().M((xR, y_top)).L((xR, c.T()[1])).d()
    bowl = SvgPath().M(c.R()).A(c.r, 0, 1, c.B()).A(c.r, 0, 1, c.L()).d()
    return [stemL, stemR, bowl], []

def glyph_u(grid: Grid):
    return glyph_U(grid)

def glyph_v(grid: Grid):
    topL = (grid.x_tan(0), xheight_top(grid))
    topR = (grid.x_tan(1), xheight_top(grid))
    bottom = (grid.circle(1, BASE_ROW + 1).cx, baseline(grid))
    return [SvgPath().M(topL).L(bottom).L(topR).d()], []

def glyph_w(grid: Grid):
    # Wide w (4 cols): 3 stems + 2 bottom bowls between them
    y_top = xheight_top(grid)
    y_base = baseline(grid)

    x0 = grid.x_tan(0)
    x1 = grid.x_tan(1)
    x2 = grid.x_tan(2)

    # bowls sit on row BASE_ROW+1 (same as n/m)
    c1 = grid.circle(1, BASE_ROW + 1)  # between x0..x1
    c2 = grid.circle(2, BASE_ROW + 1)  # between x1..x2

    leg0 = SvgPath().M((x0, y_top)).L((x0, y_base)).d()
    leg1 = SvgPath().M((x1, y_top)).L((x1, y_base)).d()
    leg2 = SvgPath().M((x2, y_top)).L((x2, y_base)).d()

    # bottom bowls: left->bottom->right (use sweep=0 like your placeholder)
    bowl1 = SvgPath().M(c1.L()).A(c1.r, 0, 0, c1.B()).A(c1.r, 0, 0, c1.R()).d()
    bowl2 = SvgPath().M(c2.L()).A(c2.r, 0, 0, c2.B()).A(c2.r, 0, 0, c2.R()).d()

    return [leg0, bowl1, leg1, bowl2, leg2], []

def glyph_x(grid: Grid):
    a1 = (grid.x_tan(0), xheight_top(grid))
    a2 = (grid.x_tan(1), baseline(grid))
    b1 = (grid.x_tan(1), xheight_top(grid))
    b2 = (grid.x_tan(0), baseline(grid))
    return [SvgPath().M(a1).L(a2).d(), SvgPath().M(b1).L(b2).d()], []

def glyph_y(grid: Grid):
    topL = (grid.x_tan(0), xheight_top(grid))
    topR = (grid.x_tan(1), xheight_top(grid))
    mid = (grid.circle(1, BASE_ROW + 1).cx, grid.circle(1, BASE_ROW + 1).cy)
    tail = (mid[0], descender_bottom(grid) - R)
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
    c = grid.circle(2, GRID_ROWS - 1)
    rr = grid.r * 0.18
    dot = SvgPath().M((c.cx + rr, c.cy)).A(rr, 0, 1, (c.cx, c.cy + rr)).A(rr, 0, 1, (c.cx - rr, c.cy)).A(rr, 0, 1, (c.cx, c.cy - rr)).A(rr, 0, 1, (c.cx + rr, c.cy)).d()
    return [dot], []

def glyph_quote_low(grid: Grid):
    c1 = grid.circle(1, GRID_ROWS - 1)
    c2 = grid.circle(2, GRID_ROWS - 1)
    rr = grid.r * 0.16
    y = c1.cy + grid.r * 0.05
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []

def glyph_quote_high(grid: Grid):
    c1 = grid.circle(1, BASE_ROW)
    c2 = grid.circle(2, BASE_ROW)
    rr = grid.r * 0.16
    y = c1.cy - grid.r * 0.45
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []

def glyph_qmark(grid: Grid):
    c = grid.circle(1, BASE_ROW)
    hook = SvgPath().M(c.T()).A(c.r, 0, 1, c.R()).A(c.r, 0, 1, c.B()).d()
    stem = SvgPath().M(c.B()).L((c.B()[0], grid.circle(1, BASE_ROW + 1).cy)).d()
    dot = glyph_dot(grid)[0]
    return [hook, stem, dot], []


# Digits (still placeholders)
def digit_0(grid: Grid): return glyph_o(grid)
def digit_1(grid: Grid):
    x = grid.x_tan(1)
    return [SvgPath().M((x, xheight_top(grid))).L((x, baseline(grid))).d()], []
def digit_2(grid: Grid):
    c = grid.circle(1, BASE_ROW + 1)
    top = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    diag = SvgPath().M(c.R()).L((grid.x_tan(0), baseline(grid))).d()
    base = SvgPath().M((grid.x_tan(0), baseline(grid))).L((grid.x_tan(1), baseline(grid))).d()
    return [top, diag, base], []
def digit_3(grid: Grid):
    cu = grid.circle(1, BASE_ROW)
    cl = grid.circle(1, BASE_ROW + 1)
    top = SvgPath().M(cu.T()).A(cu.r, 0, 1, cu.R()).A(cu.r, 0, 1, cu.B()).d()
    bot = SvgPath().M(cl.T()).A(cl.r, 0, 1, cl.R()).A(cl.r, 0, 1, cl.B()).d()
    return [top, bot], []
def digit_4(grid: Grid):
    xL = grid.x_tan(0); xR = grid.x_tan(1)
    yT = xheight_top(grid); yM = grid.circle(1, BASE_ROW + 1).cy; yB = baseline(grid)
    diag = SvgPath().M((xR, yT)).L((xL, yM)).d()
    bar  = SvgPath().M((xL, yM)).L((xR, yM)).d()
    stem = SvgPath().M((xR, yT)).L((xR, yB)).d()
    return [diag, bar, stem], []
def digit_5(grid: Grid):
    xL = grid.x_tan(0); xR = grid.x_tan(1)
    yT = xheight_top(grid); yM = grid.circle(1, BASE_ROW + 1).cy
    top = SvgPath().M((xR, yT)).L((xL, yT)).L((xL, yM)).d()
    bowl = SvgPath().M(grid.circle(1, BASE_ROW + 1).L()).A(grid.circle(1, BASE_ROW + 1).r, 0, 0, grid.circle(1, BASE_ROW + 1).R()).d()
    return [top, bowl], []
def digit_6(grid: Grid):
    c = grid.circle(1, BASE_ROW + 1)
    loop = circle_full(c)
    hook = SvgPath().M(c.T()).L((grid.x_tan(0), grid.circle(1, BASE_ROW).cy)).d()
    return [loop, hook], []
def digit_7(grid: Grid):
    xL = grid.x_tan(0); xR = grid.x_tan(1)
    yT = xheight_top(grid)
    p = SvgPath().M((xL, yT)).L((xR, yT)).L((xL, baseline(grid))).d()
    return [p], []
def digit_8(grid: Grid):
    return [circle_full(grid.circle(1, BASE_ROW)), circle_full(grid.circle(1, BASE_ROW + 1))], []
def digit_9(grid: Grid):
    c = grid.circle(1, BASE_ROW)
    loop = circle_full(c)
    tail = SvgPath().M(c.B()).L((grid.x_tan(1), grid.circle(1, BASE_ROW + 1).cy)).d()
    return [loop, tail], []


# Uppercase placeholders for extra set (until we lock them)
def glyph_H(grid: Grid):
    # two stems + bar
    xL = grid.x_tan(0); xR = grid.x_tan(1)
    yT = ascender_top(grid); yB = baseline(grid)
    yM = grid.circle(1, BASE_ROW + 1).cy
    return [
        SvgPath().M((xL, yT)).L((xL, yB)).d(),
        SvgPath().M((xR, yT)).L((xR, yB)).d(),
        SvgPath().M((xL, yM)).L((xR, yM)).d(),
    ], []

def glyph_K(grid: Grid):
    # reuse lowercase k proportions for now but ascender stem
    return glyph_k(grid)

def glyph_M(grid: Grid):
    # wide; reuse lowercase m but with ascender stem
    xL = grid.x_tan(0)
    y_top = ascender_top(grid)
    y_base = baseline(grid)
    c1 = grid.circle(1, BASE_ROW + 1)
    c2 = grid.circle(2, BASE_ROW + 1)

    stem = SvgPath().M((xL, y_top)).L((xL, y_base)).d()
    arch1 = SvgPath().M(c1.L()).A(c1.r, 0, 1, c1.T()).A(c1.r, 0, 1, c1.R()).d()
    leg1 = SvgPath().M(c1.R()).L((c1.R()[0], y_base)).d()
    arch2 = SvgPath().M(c2.L()).A(c2.r, 0, 1, c2.T()).A(c2.r, 0, 1, c2.R()).d()
    leg2 = SvgPath().M(c2.R()).L((c2.R()[0], y_base)).d()
    return [stem, arch1, leg1, arch2, leg2], []

def glyph_N(grid: Grid):
    xL = grid.x_tan(0); xR = grid.x_tan(1)
    yT = ascender_top(grid); yB = baseline(grid)
    diag = SvgPath().M((xL, yT)).L((xR, yB)).d()
    return [
        SvgPath().M((xL, yT)).L((xL, yB)).d(),
        diag,
        SvgPath().M((xR, yT)).L((xR, yB)).d(),
    ], []

def glyph_X(grid: Grid):
    return glyph_x(grid)


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

    # rest
    "c": glyph_c,
    "d": glyph_d,
    "e": glyph_E,     # temp
    "f": glyph_f,
    "g": glyph_g,
    "h": glyph_h,
    "I": glyph_I,
    "J": glyph_J,
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

    # extra uppercase set
    "H": glyph_H,
    "K": glyph_K,
    "M": glyph_M,
    "N": glyph_N,
    "X": glyph_X,
}


# -----------------------------
# SVG rendering (variable width)
# -----------------------------
def glyph_view_width(ch: str) -> int:
    return VIEW_W_WIDE if ch in ("m", "M", "w") else VIEW_W_DEFAULT

def glyph_grid_cols(ch: str) -> int:
    return GRID_COLS_WIDE if ch in ("m", "M", "w") else GRID_COLS_DEFAULT

def svg_glyph_doc(
    glyph_name: str,
    codepoint: int,
    paths_d: List[str],
    debug: bool,
    anchors: List[Tuple[str, Tuple[float, float]]],
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

    for ch in uniq(REQUESTED):
        fn = BUILDERS.get(ch)
        if fn is None:
            missing.append(ch)
            continue

        paths_d, anchors = fn(grid)

        vw = glyph_view_width(ch)
        cols = glyph_grid_cols(ch)

        svg = svg_glyph_doc(
            glyph_name=ch,
            codepoint=ord(ch),
            paths_d=paths_d,
            debug=args.debug,
            anchors=anchors,
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
