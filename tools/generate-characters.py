#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tangent-circle glyph generator (orthogonal grid, circles touch, no overlap).

Outputs:
  src/character-u{codepoint}.svg

Method:
  - Orthogonal grid of equal circles (spacing = 2R).
  - Glyph strokes are arcs on these circles + straight lines on tangent lines.
  - Anchor points are tangent extremes (L/R/T/B) and tangency points between stacked circles.

Run:
  python tools/generate-characters.py --out-dir src --debug
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path as FSPath
from typing import Callable, Dict, List, Tuple


# -----------------------------
# Characters requested (exactly)
# -----------------------------
REQUESTED = (
    "AbCdEfghIJ"
    "kLmnopqrStUvwxyZ.„”?"
    "0123456789"
    "AcGHKMNX"
)

# -----------------------------
# Geometry / SVG config
# -----------------------------
VIEW_W, VIEW_H = 240, 360

R = 40.0                  # circle radius
DX = 2.0 * R              # grid spacing (cols)
DY = 2.0 * R              # grid spacing (rows)

# Choose grid so it matches the b/a examples we validated:
# centers: (60 + 80*col, 120 + 80*row)
X0 = 60.0
Y0 = 120.0

STROKE = 9.0              # glyph stroke width

# Debug overlay
GRID_OPACITY = 0.35
GRID_STROKE = 1.0
ANCHOR_R = 3.5

OUT_DIR_DEFAULT = FSPath("src")


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
# Grid / primitives
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

    # tangent lines between cols / rows
    def x_tan(self, col_left: int) -> float:
        """Vertical tangent line between col_left and col_left+1."""
        return self.x0 + col_left * DX + self.r

    def y_tan(self, row_top: int) -> float:
        """Horizontal tangent line between row_top and row_top+1."""
        return self.y0 + row_top * DY + self.r

    def tangency_vertical(self, col: int, row_top: int) -> Tuple[float, float]:
        """Point where circles at (col,row_top) and (col,row_top+1) touch."""
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
    # 4 quarter arcs: R -> B -> L -> T -> R
    p = SvgPath().M(c.R())
    p.A(c.r, 0, sweep, c.B())
    p.A(c.r, 0, sweep, c.L())
    p.A(c.r, 0, sweep, c.T())
    p.A(c.r, 0, sweep, c.R())
    return p.d()


def semicircle_top(c: Circle, start_at: Tuple[float, float], end_at: Tuple[float, float], sweep: int) -> str:
    # caller ensures start/end are tangent extremes; used as helper when composing paths
    return SvgPath().M(start_at).A(c.r, 0, sweep, c.T()).A(c.r, 0, sweep, end_at).d()


# -----------------------------
# Glyph builders
# -----------------------------
GlyphFn = Callable[[Grid], Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]]
# returns (list_of_path_d, debug_anchors)


def glyph_a(grid: Grid) -> Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]:
    """
    Your validated A/a method:
      Two stacked circles at col=1 rows 0 and 1
      Outer: left/top/right arc (upper), stem down, bottom/left arc (lower)
      Inner closure using tangency point G with correct bow directions (sweep=0)
    """
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)

    A = cu.L()
    B = cu.T()
    C = cu.R()
    D = cl.R()
    E = cl.B()
    F = cl.L()
    G = grid.tangency_vertical(1, 0)  # (cu.cx, cu.cy + R)

    outer = SvgPath().M(A).A(cu.r, 0, 1, B).A(cu.r, 0, 1, C).L(D).A(cl.r, 0, 1, E).A(cl.r, 0, 1, F).d()
    inner_GC = SvgPath().M(G).A(cu.r, 0, 0, C).d()  # bow fixed
    inner_GF = SvgPath().M(G).A(cl.r, 0, 0, F).d()

    anchors = [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F), ("G", G)]
    return [outer, inner_GC, inner_GF], anchors


def glyph_A(grid: Grid):
    # In your sheet A looks consistent with the same construction language; start with same.
    return glyph_a(grid)


def glyph_b(grid: Grid) -> Tuple[List[str], List[Tuple[str, Tuple[float, float]]]]:
    """
    Your validated b:
      Bowl: full circle at col=1,row=1
      Stem: vertical line on the LEFT tangent line (between col0 and col1),
            from top-tangent of the row0 circle down to the bowl LEFT tangent (not below).
    """
    bowl = grid.circle(1, 1)

    x_stem = grid.x_tan(0)                 # between col0 and col1 -> 100
    y_top = grid.circle(0, 0).T()[1]       # top tangent of row0 -> 80
    y_join = bowl.L()[1]                   # bowl left tangent y -> bowl.cy

    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    bowl_path = circle_full(bowl)

    anchors = [
        ("stemTop", (x_stem, y_top)),
        ("join", (x_stem, y_join)),
        ("bowlL", bowl.L()),
        ("bowlT", bowl.T()),
        ("bowlR", bowl.R()),
        ("bowlB", bowl.B()),
    ]
    return [stem, bowl_path], anchors


def glyph_c(grid: Grid):
    # One-circle C-like (open on right): draw T -> L -> B
    c = grid.circle(1, 1)
    path = SvgPath().M(c.T()).A(c.r, 0, 1, c.L()).A(c.r, 0, 1, c.B()).d()
    return [path], []


def glyph_C(grid: Grid):
    # Tall C: two circles stacked (col=1 rows 0 and 1), open on right
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    # upper: R -> T -> L
    upper = SvgPath().M(cu.R()).A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.L()).d()
    # left spine: cu.L -> cl.L
    spine = SvgPath().M(cu.L()).L(cl.L()).d()
    # lower: L -> B -> R
    lower = SvgPath().M(cl.L()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.R()).d()
    return [upper, spine, lower], []


def glyph_d(grid: Grid):
    # Mirror of b (first pass): bowl at (1,1), stem on RIGHT tangent line (between col1 and col2),
    # from row0 top tangent down to bowl RIGHT tangent.
    bowl = grid.circle(1, 1)
    x_stem = grid.x_tan(1)                 # between col1 and col2 -> 180
    y_top = grid.circle(2, 0).T()[1]       # top tangent of row0 -> 80
    y_join = bowl.R()[1]                   # bowl right tangent y -> bowl.cy
    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_join)).d()
    bowl_path = circle_full(bowl)
    return [stem, bowl_path], []


def glyph_E(grid: Grid):
    """
    E in your sample reads like a backward '3'/epsilon.
    Grid-faithful first pass:
      - Two stacked circles at col=1 rows 0 and 1
      - Draw 3/4 arc around left side of upper into the middle tangency,
        then 3/4 arc around left side of lower out to lower-right endpoint
      - Add a short middle tick from the tangency toward the right (like the notch).
    """
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    G = grid.tangency_vertical(1, 0)

    # upper: start at cu.R, go around top+left to G (bottom of upper)
    upper = SvgPath().M(cu.R()).A(cu.r, 0, 1, cu.T()).A(cu.r, 0, 1, cu.L()).A(cu.r, 0, 1, G).d()
    # lower: from G go around left+bottom to cl.R
    lower = SvgPath().M(G).A(cl.r, 0, 1, cl.L()).A(cl.r, 0, 1, cl.B()).A(cl.r, 0, 1, cl.R()).d()

    # middle notch/tick (short, grid-locked: from G toward the right tangent line)
    tick_end = (grid.x_tan(1), G[1])  # go to the right tangent line
    tick = SvgPath().M(G).L(tick_end).d()

    return [upper, lower, tick], [("G", G)]


def glyph_f(grid: Grid):
    # First pass: tall stem on left tangent line + a mid arm to the right.
    x = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_bot = grid.circle(0, 2).T()[1]  # extends lower
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()

    y_arm = grid.circle(1, 1).cy
    arm = SvgPath().M((x, y_arm)).L((grid.x_tan(1), y_arm)).d()
    return [stem, arm], []


def glyph_g(grid: Grid):
    # In your sample g resembles a stacked double-loop (like an 8), grid-faithful first pass:
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    return [circle_full(cu), circle_full(cl)], []


def glyph_h(grid: Grid):
    # stem on left tangent line + arch (half circle) to the right, ending at the right tangent line
    x_stem = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_base = grid.circle(1, 1).cy
    stem = SvgPath().M((x_stem, y_top)).L((x_stem, y_base)).d()

    arch_c = grid.circle(1, 1)  # bowl circle for the arch
    # arch: from left tangent (matches stem) -> top -> right tangent
    arch = SvgPath().M(arch_c.L()).A(arch_c.r, 0, 1, arch_c.T()).A(arch_c.r, 0, 1, arch_c.R()).d()
    # right downstroke stops at right tangent (same as arch end)
    return [stem, arch], []


def glyph_I(grid: Grid):
    x = grid.x_tan(0)  # use a tangent line for consistent thin stem placement
    y_top = grid.circle(0, 0).T()[1]
    y_bot = grid.circle(0, 2).T()[1]
    return [SvgPath().M((x, y_top)).L((x, y_bot)).d()], []


def glyph_J(grid: Grid):
    # stem + bottom hook on a circle
    x = grid.x_tan(1)
    y_top = grid.circle(2, 0).T()[1]
    c = grid.circle(1, 1)
    stem = SvgPath().M((x, y_top)).L((x, c.B()[1])).d()
    hook = SvgPath().M((x, c.B()[1])).A(c.r, 0, 1, c.L()).d()
    return [stem, hook], []


def glyph_k(grid: Grid):
    # stem + two diagonals
    x = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_mid = grid.circle(1, 1).cy
    y_bot = grid.circle(0, 2).T()[1]
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    arm1 = SvgPath().M((x, y_mid)).L((grid.x_tan(1), grid.circle(1, 0).cy)).d()
    arm2 = SvgPath().M((x, y_mid)).L((grid.x_tan(1), grid.circle(1, 2).cy)).d()
    return [stem, arm1, arm2], []


def glyph_L(grid: Grid):
    # stem + base
    x = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_bot = grid.circle(0, 2).T()[1]
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    base = SvgPath().M((x, y_bot)).L((grid.x_tan(1), y_bot)).d()
    return [stem, base], []


def glyph_m(grid: Grid):
    # stem + two arches
    x = grid.x_tan(0)
    y_top = grid.circle(0, 1).T()[1]
    y_base = grid.circle(0, 1).cy
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()

    c1 = grid.circle(1, 1)
    c2 = grid.circle(2, 1)
    arch1 = SvgPath().M(c1.L()).A(c1.r, 0, 1, c1.T()).A(c1.r, 0, 1, c1.R()).d()
    arch2 = SvgPath().M(c2.L()).A(c2.r, 0, 1, c2.T()).A(c2.r, 0, 1, c2.R()).d()
    return [stem, arch1, arch2], []


def glyph_n(grid: Grid):
    # stem + one arch
    x = grid.x_tan(0)
    y_top = grid.circle(0, 1).T()[1]
    y_base = grid.circle(0, 1).cy
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()

    c = grid.circle(1, 1)
    arch = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    return [stem, arch], []


def glyph_o(grid: Grid):
    return [circle_full(grid.circle(1, 1))], []


def glyph_p(grid: Grid):
    # stem down + bowl on top (first pass)
    x = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_bot = grid.circle(0, 2).B()[1]
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    bowl = circle_full(grid.circle(1, 1))
    return [stem, bowl], []


def glyph_q(grid: Grid):
    # bowl + diagonal tail
    bowl = circle_full(grid.circle(1, 1))
    tail = SvgPath().M(grid.circle(1, 1).B()).L((grid.x_tan(1), grid.circle(2, 2).cy)).d()
    return [bowl, tail], []


def glyph_r(grid: Grid):
    # stem + small shoulder
    x = grid.x_tan(0)
    y_top = grid.circle(0, 1).T()[1]
    y_base = grid.circle(0, 1).cy
    stem = SvgPath().M((x, y_top)).L((x, y_base)).d()
    c = grid.circle(1, 1)
    shoulder = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).d()
    return [stem, shoulder], []


def glyph_S(grid: Grid):
    # two opposing arcs (first pass)
    cu = grid.circle(1, 0)
    cl = grid.circle(1, 1)
    top = SvgPath().M(cu.R()).A(cu.r, 0, 0, cu.L()).d()
    bot = SvgPath().M(cl.L()).A(cl.r, 0, 0, cl.R()).d()
    return [top, bot], []


def glyph_t(grid: Grid):
    x = grid.x_tan(0)
    y_top = grid.circle(0, 0).T()[1]
    y_bot = grid.circle(0, 2).T()[1]
    stem = SvgPath().M((x, y_top)).L((x, y_bot)).d()
    y_bar = grid.circle(1, 1).cy
    bar = SvgPath().M((grid.x_tan(0), y_bar)).L((grid.x_tan(1), y_bar)).d()
    return [stem, bar], []


def glyph_U(grid: Grid):
    # two stems + bottom semicircle
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    y_top = grid.circle(0, 0).T()[1]
    c = grid.circle(1, 1)
    y_join = c.T()[1]  # stems stop at top of bowl
    stemL = SvgPath().M((xL, y_top)).L((xL, y_join)).d()
    stemR = SvgPath().M((xR, y_top)).L((xR, y_join)).d()
    bowl = SvgPath().M(c.R()).A(c.r, 0, 1, c.B()).A(c.r, 0, 1, c.L()).d()
    return [stemL, stemR, bowl], []


def glyph_u(grid: Grid):
    return glyph_U(grid)


def glyph_v(grid: Grid):
    topL = (grid.x_tan(0), grid.circle(0, 1).T()[1])
    topR = (grid.x_tan(1), grid.circle(0, 1).T()[1])
    bottom = (grid.circle(1, 1).cx, grid.circle(1, 1).B()[1])
    return [SvgPath().M(topL).L(bottom).L(topR).d()], []


def glyph_w(grid: Grid):
    t0 = (grid.x_tan(0), grid.circle(0, 1).T()[1])
    t1 = (grid.circle(1, 1).cx, grid.circle(1, 1).T()[1])
    t2 = (grid.x_tan(1), grid.circle(0, 1).T()[1])
    b0 = (grid.circle(0, 1).cx, grid.circle(0, 1).B()[1])
    b1 = (grid.circle(2, 1).cx, grid.circle(2, 1).B()[1])
    return [SvgPath().M(t0).L(b0).L(t1).L(b1).L(t2).d()], []


def glyph_x(grid: Grid):
    a1 = (grid.x_tan(0), grid.circle(0, 1).T()[1])
    a2 = (grid.x_tan(1), grid.circle(2, 1).B()[1])
    b1 = (grid.x_tan(1), grid.circle(0, 1).T()[1])
    b2 = (grid.x_tan(0), grid.circle(2, 1).B()[1])
    return [SvgPath().M(a1).L(a2).d(), SvgPath().M(b1).L(b2).d()], []


def glyph_y(grid: Grid):
    topL = (grid.x_tan(0), grid.circle(0, 1).T()[1])
    topR = (grid.x_tan(1), grid.circle(0, 1).T()[1])
    mid = (grid.circle(1, 1).cx, grid.circle(1, 1).cy)
    tail = (mid[0], grid.circle(1, 2).cy)
    return [SvgPath().M(topL).L(mid).L(topR).d(), SvgPath().M(mid).L(tail).d()], []


def glyph_Z(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = grid.circle(1, 0).T()[1]
    yB = grid.circle(1, 1).B()[1]
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
    # „ two dots below
    c1 = grid.circle(1, 2)
    c2 = grid.circle(2, 2)
    rr = grid.r * 0.16
    y = c1.cy + grid.r * 0.35
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []


def glyph_quote_high(grid: Grid):
    # ” two dots above
    c1 = grid.circle(1, 0)
    c2 = grid.circle(2, 0)
    rr = grid.r * 0.16
    y = c1.cy - grid.r * 0.35
    def tiny(cx: float) -> str:
        return SvgPath().M((cx + rr, y)).A(rr, 0, 1, (cx, y + rr)).A(rr, 0, 1, (cx - rr, y)).A(rr, 0, 1, (cx, y - rr)).A(rr, 0, 1, (cx + rr, y)).d()
    return [tiny(c1.cx), tiny(c2.cx)], []


def glyph_qmark(grid: Grid):
    # hook + stem + dot (first pass)
    c = grid.circle(1, 0)
    hook = SvgPath().M(c.T()).A(c.r, 0, 1, c.R()).A(c.r, 0, 1, c.B()).d()
    stem = SvgPath().M(c.B()).L((c.B()[0], grid.circle(1, 1).cy)).d()
    dot = glyph_dot(grid)[0]
    return [hook, stem, dot], []


# Digits (grid-faithful first pass)
def digit_0(grid: Grid): return glyph_o(grid)
def digit_1(grid: Grid):
    x = grid.x_tan(1)
    yT = grid.circle(2, 0).T()[1]
    yB = grid.circle(2, 2).T()[1]
    return [SvgPath().M((x, yT)).L((x, yB)).d()], []
def digit_2(grid: Grid):
    c = grid.circle(1, 1)
    top = SvgPath().M(c.L()).A(c.r, 0, 1, c.T()).A(c.r, 0, 1, c.R()).d()
    diag = SvgPath().M(c.R()).L((grid.x_tan(0), grid.circle(2, 1).B()[1])).d()
    base = SvgPath().M((grid.x_tan(0), grid.circle(2, 1).B()[1])).L((grid.x_tan(1), grid.circle(2, 1).B()[1])).d()
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
    yT = grid.circle(1, 0).T()[1]
    yM = grid.circle(1, 1).cy
    yB = grid.circle(1, 2).T()[1]
    diag = SvgPath().M((xR, yT)).L((xL, yM)).d()
    bar = SvgPath().M((xL, yM)).L((xR, yM)).d()
    stem = SvgPath().M((xR, yT)).L((xR, yB)).d()
    return [diag, bar, stem], []
def digit_5(grid: Grid):
    xL = grid.x_tan(0)
    xR = grid.x_tan(1)
    yT = grid.circle(1, 0).T()[1]
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
    yT = grid.circle(1, 0).T()[1]
    p = SvgPath().M((xL, yT)).L((xR, yT)).L((xL, grid.circle(2, 1).B()[1])).d()
    return [p], []
def digit_8(grid: Grid):
    return [circle_full(grid.circle(1, 0)), circle_full(grid.circle(1, 1))], []
def digit_9(grid: Grid):
    c = grid.circle(1, 0)
    loop = circle_full(c)
    tail = SvgPath().M(c.B()).L((grid.x_tan(1), grid.circle(1, 1).cy)).d()
    return [loop, tail], []


# -----------------------------
# Map requested glyphs
# -----------------------------
BUILDERS: Dict[str, GlyphFn] = {
    # Top row set
    "a": glyph_a,
    "A": glyph_A,
    "b": glyph_b,
    "C": glyph_C,
    "c": glyph_c,
    "d": glyph_d,
    "E": glyph_E,
    "e": glyph_E,   # first pass (your sheet shows E, not sure about lowercase e in this design)
    "f": glyph_f,
    "g": glyph_g,
    "h": glyph_h,
    "I": glyph_I,
    "J": glyph_J,

    # Second/third row set (first-pass, grid-faithful)
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

    # punctuation
    ".": glyph_dot,
    "„": glyph_quote_low,
    "”": glyph_quote_high,
    "?": glyph_qmark,

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

    # extra uppercase set you listed
    "G": glyph_g,     # placeholder-ish but same language
    "H": glyph_h,
    "K": glyph_k,
    "M": glyph_m,
    "N": glyph_n,
    "X": glyph_x,
}


# -----------------------------
# SVG document
# -----------------------------
def svg_doc(paths_d: List[str], debug: bool, anchors: List[Tuple[str, Tuple[float, float]]], grid: Grid) -> str:
    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}" width="{VIEW_W}" height="{VIEW_H}">')
    out.append('<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>')

    if debug:
        out.append(f'<g fill="none" stroke="#bdbdbd" stroke-width="{fmt(GRID_STROKE)}" opacity="{fmt(GRID_OPACITY)}">')
        # draw a patch of the orthogonal tangent grid
        for row in range(0, 4):
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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT), help="Output directory (default: src)")
    ap.add_argument("--debug", action="store_true", help="Include grid overlay (recommended while refining)")
    args = ap.parse_args()

    out_dir = FSPath(args.out_dir)
    grid = Grid(r=R, x0=X0, y0=Y0)

    chars = uniq(REQUESTED)
    missing: List[str] = []
    written = 0

    for ch in chars:
        fn = BUILDERS.get(ch)
        if fn is None:
            missing.append(ch)
            continue

        paths_d, anchors = fn(grid)
        doc = svg_doc(paths_d, debug=args.debug, anchors=anchors, grid=grid)

        # required pattern: src/character-u{codepoint}
        out_name = f"character-u{ord(ch):04x}.svg"
        write_text_lf(out_dir / out_name, doc)
        written += 1
        print(f"✓ {ch} -> {out_name}")

    print(f"\nDone. Wrote {written} glyph(s) into {out_dir.resolve()}")
    if missing:
        print("No builder yet for:", "".join(missing))


if __name__ == "__main__":
    main()
