#!/usr/bin/env python3
"""
generate_glyph_svgs.py

Reconstructs the monoline / arc-based glyph style from the screenshot using a tiny DSL:
  ("line", x0, y0, x1, y1)
  ("arc",  cx, cy, r, a0_deg, a1_deg)   # angles in degrees, y-axis DOWN (SVG-like)
  ("circle", cx, cy, r)

Outputs one SVG per character into ../src (relative to this script).

Run:
  python generate_glyph_svgs.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable


# ---------------------------
# Geometry / style (tweak me)
# ---------------------------
EM_W = 240
EM_H = 240

# vertical metrics (y down)
ASC  = 24     # ascender top
XH   = 60     # x-height line (tops of n/m arches)
BASE = 180    # baseline
DESC = 228    # descender bottom

STROKE_W = 16

# "main bowl" circle used by o/b/d/p/q/g/a-ish
CX = EM_W / 2
CY = (XH + BASE) / 2
BOWL_R = (BASE - XH) / 2  # nice: top hits XH, bottom hits BASE

# typical stem x positions for bowls
BOWL_L = CX - BOWL_R
BOWL_R_X = CX + BOWL_R

# narrower stems for arch letters (n,m,h,u)
ARCH_L = 80
ARCH_R = 160

# punctuation dot radius
DOT_R = 7


# ---------------------------
# Cross-version write helper
# ---------------------------
def write_text_lf(path: Path, text: str) -> None:
    """
    Write UTF-8 text with LF newlines on all platforms and Python versions.
    (Path.write_text(newline=...) is not available on older Python.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------
# Primitive types
# ---------------------------
Prim = Union[
    Tuple[str, float, float, float, float],        # line
    Tuple[str, float, float, float, float, float], # arc
    Tuple[str, float, float, float],               # circle
]


def _pt(cx: float, cy: float, r: float, a_deg: float) -> Tuple[float, float]:
    """Point on circle in SVG-like coords (y down)."""
    a = math.radians(a_deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a))


def _arc_path(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """
    Build an SVG arc command as a standalone path (M ... A ...).
    Angles are in degrees. Positive delta means clockwise in our y-down system.
    """
    delta = a1 - a0
    if delta == 0:
        x0, y0 = _pt(cx, cy, r, a0)
        return f"M {x0:.3f} {y0:.3f}"

    x0, y0 = _pt(cx, cy, r, a0)
    x1, y1 = _pt(cx, cy, r, a1)

    large_arc = 1 if (abs(delta) % 360) > 180 else 0
    sweep = 1 if delta > 0 else 0  # y-down: increasing angle = clockwise

    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {r:.3f} {r:.3f} 0 {large_arc} {sweep} {x1:.3f} {y1:.3f}"
    )


def _full_circle_paths(cx: float, cy: float, r: float) -> List[str]:
    """Draw a full circle as two 180° arcs (avoids SVG 360° arc ambiguity)."""
    return [
        _arc_path(cx, cy, r, 0, 180),
        _arc_path(cx, cy, r, 180, 360),
    ]


def prims_to_svg_paths(prims: Iterable[Prim]) -> List[str]:
    paths: List[str] = []
    for p in prims:
        if not p:
            continue
        if p[0] == "line":
            _, x0, y0, x1, y1 = p
            paths.append(f"M {x0:.3f} {y0:.3f} L {x1:.3f} {y1:.3f}")
        elif p[0] == "arc":
            _, cx, cy, r, a0, a1 = p
            # full circle (or any 360 multiple) -> split
            if (a1 - a0) % 360 == 0:
                paths.extend(_full_circle_paths(cx, cy, r))
            else:
                paths.append(_arc_path(cx, cy, r, a0, a1))
        elif p[0] == "circle":
            _, cx, cy, r = p
            paths.extend(_full_circle_paths(cx, cy, r))
        else:
            raise ValueError(f"Unknown primitive: {p[0]}")
    return paths


def svg_doc(paths_d: List[str], *, label: str) -> str:
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{EM_W}" height="{EM_H}" '
        f'viewBox="0 0 {EM_W} {EM_H}">\n'
        f'  <title>{label}</title>\n'
        f'  <g fill="none" stroke="#000" stroke-width="{STROKE_W}" '
        f'stroke-linecap="round" stroke-linejoin="round">\n'
    )
    body = "".join([f'    <path d="{d}"/>\n' for d in paths_d])
    footer = "  </g>\n</svg>\n"
    return header + body + footer


# ---------------------------
# Glyph builders (reconstructed)
# ---------------------------
def glyph_o() -> List[Prim]:
    return [("circle", CX, CY, BOWL_R)]


def glyph_c() -> List[Prim]:
    # open on the right: draw the long arc from 45° -> 315° (270° sweep)
    return [("arc", CX, CY, BOWL_R, 45, 315)]


def glyph_e() -> List[Prim]:
    return [
        ("arc", CX, CY, BOWL_R, 45, 315),
        ("line", CX - BOWL_R, CY, CX, CY),
    ]


def glyph_b() -> List[Prim]:
    return [
        ("line", BOWL_L, ASC, BOWL_L, BASE),
        ("circle", CX, CY, BOWL_R),
    ]


def glyph_d() -> List[Prim]:
    return [
        ("line", BOWL_R_X, ASC, BOWL_R_X, BASE),
        ("circle", CX, CY, BOWL_R),
    ]


def glyph_p() -> List[Prim]:
    return [
        ("line", BOWL_L, XH, BOWL_L, DESC),
        ("circle", CX, CY, BOWL_R),
    ]


def glyph_q() -> List[Prim]:
    return [
        ("line", BOWL_R_X, XH, BOWL_R_X, DESC),
        ("circle", CX, CY, BOWL_R),
    ]


def glyph_a() -> List[Prim]:
    return [
        ("circle", CX, CY, BOWL_R),
        ("line", BOWL_R_X, XH, BOWL_R_X, BASE),
        ("line", CX, XH, BOWL_R_X, XH),
    ]


def glyph_g() -> List[Prim]:
    return [
        ("circle", CX, CY, BOWL_R),
        ("line", BOWL_R_X, XH, BOWL_R_X, DESC),
        ("arc", BOWL_R_X - 18, DESC - 10, 18, 300, 120),
    ]


def top_arch(x1: float, x2: float) -> List[Prim]:
    r = (x2 - x1) / 2
    cx = (x1 + x2) / 2
    # bulge downward: 180 -> 360
    return [("arc", cx, XH, r, 180, 360)]


def bottom_arch(x1: float, x2: float) -> List[Prim]:
    r = (x2 - x1) / 2
    cx = (x1 + x2) / 2
    # bulge upward: 180 -> 0 (negative sweep)
    return [("arc", cx, BASE, r, 180, 0)]


def glyph_n() -> List[Prim]:
    return [
        ("line", ARCH_L, BASE, ARCH_L, XH),
        *top_arch(ARCH_L, ARCH_R),
        ("line", ARCH_R, XH, ARCH_R, BASE),
    ]


def glyph_m() -> List[Prim]:
    x1, x2, x3 = 70, 120, 170
    return [
        ("line", x1, BASE, x1, XH),
        *top_arch(x1, x2),
        ("line", x2, XH, x2, BASE),
        ("line", x2, BASE, x2, XH),
        *top_arch(x2, x3),
        ("line", x3, XH, x3, BASE),
    ]


def glyph_h() -> List[Prim]:
    return [
        ("line", ARCH_L, ASC, ARCH_L, BASE),
        *top_arch(ARCH_L, ARCH_R),
        ("line", ARCH_R, XH, ARCH_R, BASE),
    ]


def glyph_u() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, ARCH_L, BASE),
        *bottom_arch(ARCH_L, ARCH_R),
        ("line", ARCH_R, BASE, ARCH_R, XH),
    ]


def glyph_l() -> List[Prim]:
    return [
        ("line", CX, ASC, CX, BASE),
        ("arc", CX + 16, BASE, 16, 180, 270),  # tiny foot to the right
    ]


def glyph_i() -> List[Prim]:
    return [
        ("line", CX, XH, CX, BASE),
        ("circle", CX, XH - 18, DOT_R),
    ]


def glyph_t() -> List[Prim]:
    return [
        ("line", CX, ASC, CX, BASE),
        ("arc", CX + 22, ASC + 4, 22, 180, 315),  # top hook to the right
    ]


def glyph_k() -> List[Prim]:
    midy = (XH + BASE) / 2
    return [
        ("line", ARCH_L, ASC, ARCH_L, BASE),
        ("line", ARCH_L, midy, ARCH_R, XH),
        ("line", ARCH_L, midy, ARCH_R, BASE),
    ]


def glyph_r() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, ARCH_L, BASE),
        ("arc", ARCH_L + 26, XH, 26, 180, 330),
    ]


def glyph_s() -> List[Prim]:
    r = 36
    return [
        ("arc", CX, XH + r, r, 220, 20),
        ("arc", CX, BASE - r, r, 200, 360 + 20),
    ]


def glyph_v() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, CX, BASE),
        ("line", CX, BASE, ARCH_R, XH),
    ]


def glyph_w() -> List[Prim]:
    x1, x2, x3, x4 = 70, 110, 150, 190
    return [
        ("line", x1, XH, x2, BASE),
        ("line", x2, BASE, x3, XH),
        ("line", x3, XH, x4, BASE),
    ]


def glyph_x() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, ARCH_R, BASE),
        ("line", ARCH_L, BASE, ARCH_R, XH),
    ]


def glyph_y() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, CX, BASE),
        ("line", CX, BASE, ARCH_R, XH),
        ("line", CX, BASE, CX, DESC),
        ("arc", CX - 18, DESC - 6, 18, 330, 150),
    ]


def glyph_z() -> List[Prim]:
    return [
        ("line", ARCH_L, XH, ARCH_R, XH),
        ("line", ARCH_R, XH, ARCH_L, BASE),
        ("line", ARCH_L, BASE, ARCH_R, BASE),
    ]


def glyph_period() -> List[Prim]:
    return [("circle", CX, BASE, DOT_R)]


def glyph_comma() -> List[Prim]:
    return [
        ("circle", CX, BASE, DOT_R),
        ("arc", CX - 6, BASE + 18, 16, 300, 140),
    ]


def glyph_question() -> List[Prim]:
    r = 46
    return [
        ("arc", CX, XH + 20, r, 200, 360 + 20),
        ("line", CX, XH + 20 + r, CX, BASE - 18),
        ("circle", CX, BASE, DOT_R),
    ]


def glyph_2() -> List[Prim]:
    r = 46
    return [
        ("arc", CX, XH + 20, r, 200, 360 + 20),
        ("line", CX + r, XH + 20, ARCH_L, BASE),
        ("line", ARCH_L, BASE, ARCH_R, BASE),
    ]


def glyph_4() -> List[Prim]:
    return [
        ("line", ARCH_R, ASC, ARCH_R, BASE),
        ("line", ARCH_L, (XH + BASE) / 2, ARCH_R, (XH + BASE) / 2),
        ("line", ARCH_L, ASC + 20, ARCH_L, (XH + BASE) / 2),
    ]


# ---------------------------
# Glyph map
# ---------------------------
GLYPHS: Dict[str, List[Prim]] = {
    # letters (lowercase)
    "a": glyph_a(),
    "b": glyph_b(),
    "c": glyph_c(),
    "d": glyph_d(),
    "e": glyph_e(),
    "f": [("line", CX, ASC, CX, BASE), ("arc", CX + 28, XH + 10, 28, 180, 330)],
    "g": glyph_g(),
    "h": glyph_h(),
    "i": glyph_i(),
    "j": [("line", CX, XH, CX, DESC), ("arc", CX - 18, DESC - 6, 18, 330, 150), ("circle", CX, XH - 18, DOT_R)],
    "k": glyph_k(),
    "l": glyph_l(),
    "m": glyph_m(),
    "n": glyph_n(),
    "o": glyph_o(),
    "p": glyph_p(),
    "q": glyph_q(),
    "r": glyph_r(),
    "s": glyph_s(),
    "t": glyph_t(),
    "u": glyph_u(),
    "v": glyph_v(),
    "w": glyph_w(),
    "x": glyph_x(),
    "y": glyph_y(),
    "z": glyph_z(),

    # digits visible in your screenshot
    "2": glyph_2(),
    "4": glyph_4(),

    # punctuation visible
    ".": glyph_period(),
    ",": glyph_comma(),
    "?": glyph_question(),
}

# Uppercase aliases (your screenshot shows C E I J S)
UPPER_ALIASES = {"C": "c", "E": "e", "I": "l", "J": "j", "S": "s"}
for up, lo in UPPER_ALIASES.items():
    GLYPHS[up] = GLYPHS[lo]


def slug_for_filename(ch: str) -> str:
    special = {
        ".": "period",
        ",": "comma",
        "?": "question",
        " ": "space",
        "/": "slash",
        "\\": "backslash",
        ":": "colon",
        ";": "semicolon",
        "'": "apostrophe",
        '"': "quote",
        "-": "hyphen",
        "_": "underscore",
    }
    if ch in special:
        return special[ch]
    if ch.isalnum():
        return ch
    return f"U{ord(ch):04X}"


def main() -> None:
    out_dir = (Path(__file__).resolve().parent / "../src").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = list("abCdefghIJ") + list("klmnopqrSt") + list("uvwxyz") + list("42") + [",", ".", "?"]

    # de-dup while keeping order
    seen = set()
    chars = [c for c in ordered if not (c in seen or seen.add(c))]

    manifest_lines: List[str] = []
    for ch in chars:
        if ch not in GLYPHS:
            continue

        prims = GLYPHS[ch]
        paths = prims_to_svg_paths(prims)
        svg = svg_doc(paths, label=f"glyph {repr(ch)}")

        cp = ord(ch)
        cp_name = f"character-U{cp:04X}.svg"
        friendly = f"character-{slug_for_filename(ch)}.svg"

        write_text_lf(out_dir / cp_name, svg)
        write_text_lf(out_dir / friendly, svg)

        manifest_lines.append(f"{repr(ch)}\t{cp_name}\t{friendly}")

    write_text_lf(
        out_dir / "glyph-manifest.tsv",
        "char\tcodepoint_file\tfriendly_file\n" + "\n".join(manifest_lines) + "\n",
    )

    print(f"Wrote {len(manifest_lines)} glyphs to: {out_dir}")
    print(f"Manifest: {out_dir / 'glyph-manifest.tsv'}")


if __name__ == "__main__":
    main()
