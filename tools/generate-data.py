#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Distribution glyph data files from SVG glyphs in ./src.

Reads:
  src/character-uXXXX.svg

Writes:
  data/glyphs.py
  data/glyphs.php
  data/glyphs.json

Assumptions:
- Glyph SVGs use:
    * <path d="M ... L ..."> for lines
    * <path d="M ... A 40 40 0 0 sweep ..."> for quarter circles
    * <circle cx="..." cy="..." r="1"> for dots
- Background guide circles (r=40), white rects, etc. are ignored.
- Script lives in ./tools, so project root is script_dir/..
"""

from __future__ import annotations

import json
import math
import pprint
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STEP = 40
GUIDE_STEP = 80
HEIGHT = 320

RE_FILENAME = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")
RE_TOKEN = re.compile(r"[MLAmla]|-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def num(v: str) -> float:
    return float(v.strip())


def clean_number(v: float) -> int | float:
    if math.isclose(v, round(v), abs_tol=1e-9):
        return int(round(v))
    return round(v, 6)


def point_to_dict(p: Point) -> Dict[str, int | float]:
    return {
        "x": clean_number(p.x),
        "y": clean_number(p.y),
    }


def normalize_angle_delta(delta: float) -> float:
    while delta <= -math.pi:
        delta += math.pi * 2
    while delta > math.pi:
        delta -= math.pi * 2
    return delta


def arc_sweep_for_variant(p1: Point, p2: Point, variant: int) -> int:
    center = Point(p1.x, p2.y) if variant == 0 else Point(p2.x, p1.y)
    a1 = math.atan2(p1.y - center.y, p1.x - center.x)
    a2 = math.atan2(p2.y - center.y, p2.x - center.x)
    delta = normalize_angle_delta(a2 - a1)
    return 1 if delta > 0 else 0


def infer_arc_variant_from_sweep(p1: Point, p2: Point, sweep: int) -> int:
    v0 = arc_sweep_for_variant(p1, p2, 0)
    v1 = arc_sweep_for_variant(p1, p2, 1)

    if v0 == sweep and v1 != sweep:
        return 0
    if v1 == sweep and v0 != sweep:
        return 1
    return 0


def serialize_shape_key(shape: Dict[str, Any]) -> str:
    if shape["type"] == "dot":
        p = shape["p"]
        return f"dot:{p['x']},{p['y']}"
    if shape["type"] == "line":
        a = f"{shape['p1']['x']},{shape['p1']['y']}"
        b = f"{shape['p2']['x']},{shape['p2']['y']}"
        return f"line:{'|'.join(sorted([a, b]))}"
    a = f"{shape['p1']['x']},{shape['p1']['y']}"
    b = f"{shape['p2']['x']},{shape['p2']['y']}"
    return f"arc:{shape['variant']}:{'|'.join(sorted([a, b]))}"


def parse_viewbox(root: ET.Element) -> tuple[int | float, int | float]:
    viewbox = root.get("viewBox")
    if not viewbox:
        raise ValueError("SVG missing viewBox")
    parts = viewbox.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"Unexpected viewBox: {viewbox}")
    width = clean_number(float(parts[2]))
    height = clean_number(float(parts[3]))
    return width, height


def parse_path_d(d: str) -> List[Dict[str, Any]]:
    tokens = RE_TOKEN.findall(d)
    shapes: List[Dict[str, Any]] = []
    i = 0
    current: Optional[Point] = None

    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        cmd_u = cmd.upper()

        if cmd_u == "M":
            if i + 1 >= len(tokens):
                raise ValueError(f"Incomplete M command in path: {d}")
            current = Point(num(tokens[i]), num(tokens[i + 1]))
            i += 2
            continue

        if cmd_u == "L":
            if current is None:
                raise ValueError(f"L command without current point in path: {d}")
            if i + 1 >= len(tokens):
                raise ValueError(f"Incomplete L command in path: {d}")
            nxt = Point(num(tokens[i]), num(tokens[i + 1]))
            i += 2
            shapes.append({
                "type": "line",
                "p1": point_to_dict(current),
                "p2": point_to_dict(nxt),
            })
            current = nxt
            continue

        if cmd_u == "A":
            if current is None:
                raise ValueError(f"A command without current point in path: {d}")
            if i + 6 >= len(tokens):
                raise ValueError(f"Incomplete A command in path: {d}")

            rx = num(tokens[i]); ry = num(tokens[i + 1])
            x_axis_rotation = num(tokens[i + 2])
            large_arc_flag = int(float(tokens[i + 3]))
            sweep_flag = int(float(tokens[i + 4]))
            x = num(tokens[i + 5]); y = num(tokens[i + 6])
            i += 7

            nxt = Point(x, y)

            # Distribution quarter arcs
            if (
                math.isclose(rx, STEP, abs_tol=1e-9)
                and math.isclose(ry, STEP, abs_tol=1e-9)
                and math.isclose(x_axis_rotation, 0, abs_tol=1e-9)
                and large_arc_flag == 0
            ):
                shapes.append({
                    "type": "arc",
                    "p1": point_to_dict(current),
                    "p2": point_to_dict(nxt),
                    "variant": infer_arc_variant_from_sweep(current, nxt, sweep_flag),
                })
            else:
                raise ValueError(f"Unsupported arc in path: {d}")

            current = nxt
            continue

        raise ValueError(f"Unsupported path command {cmd!r} in path: {d}")

    return shapes


def parse_svg_file(path: Path) -> Dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()

    width, height = parse_viewbox(root)
    if height != HEIGHT:
        raise ValueError(f"{path.name}: expected height {HEIGHT}, got {height}")

    filename_match = RE_FILENAME.match(path.name)
    if not filename_match:
        raise ValueError(f"Unexpected filename: {path.name}")

    hex_code = filename_match.group(1).upper().zfill(4)
    codepoint = int(hex_code, 16)
    char = chr(codepoint)

    shapes: List[Dict[str, Any]] = []

    for elem in root.iter():
        tag = local_name(elem.tag)

        if tag == "path":
            d = elem.get("d")
            if d:
                shapes.extend(parse_path_d(d))

        elif tag == "circle":
            r_attr = elem.get("r")
            cx_attr = elem.get("cx")
            cy_attr = elem.get("cy")
            if not r_attr or not cx_attr or not cy_attr:
                continue

            r = num(r_attr)
            cx = num(cx_attr)
            cy = num(cy_attr)

            # Only actual dots, not guide circles
            if math.isclose(r, 1, abs_tol=1e-9):
                shapes.append({
                    "type": "dot",
                    "p": {
                        "x": clean_number(cx),
                        "y": clean_number(cy),
                    },
                })

    # Deduplicate while preserving order
    unique: Dict[str, Dict[str, Any]] = {}
    for shape in shapes:
        unique.setdefault(serialize_shape_key(shape), shape)
    shapes = list(unique.values())

    glyph = {
        "char": char,
        "codepoint": codepoint,
        "unicode": f"U+{hex_code}",
        "hex": hex_code,
        "filename": path.name,
        "width": width,
        "height": height,
        "guide_cols": int(round(float(width) / GUIDE_STEP)),
        "shapes": shapes,
    }
    return glyph


def build_payload(glyphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_codepoint = {glyph["unicode"]: glyph for glyph in glyphs}
    by_char = {glyph["char"]: glyph for glyph in glyphs}
    return {
        "glyphs": glyphs,
        "by_codepoint": by_codepoint,
        "by_char": by_char,
    }


def py_literal(obj: Any) -> str:
    return pprint.pformat(obj, sort_dicts=False, width=100)


def php_escape_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def php_dump(obj: Any, indent: int = 0) -> str:
    pad = " " * indent
    next_pad = " " * (indent + 4)

    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and math.isfinite(obj) and math.isclose(obj, round(obj), abs_tol=1e-9):
            return str(int(round(obj)))
        return repr(obj)
    if isinstance(obj, str):
        return "'" + php_escape_string(obj) + "'"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        parts = ["["]
        for item in obj:
            parts.append(f"{next_pad}{php_dump(item, indent + 4)},")
        parts.append(f"{pad}]")
        return "\n".join(parts)

    if isinstance(obj, dict):
        if not obj:
            return "[]"
        parts = ["["]
        for key, value in obj.items():
            if isinstance(key, int):
                php_key = str(key)
            else:
                php_key = "'" + php_escape_string(str(key)) + "'"
            parts.append(f"{next_pad}{php_key} => {php_dump(value, indent + 4)},")
        parts.append(f"{pad}]")
        return "\n".join(parts)

    raise TypeError(f"Unsupported type for PHP dump: {type(obj)!r}")


def write_python(path: Path, payload: Dict[str, Any]) -> None:
    content = (
        "# -*- coding: utf-8 -*-\n"
        '"""\n'
        "Auto-generated Distribution glyph data.\n"
        "Do not edit manually; regenerate from SVGs in ./src.\n"
        '"""\n\n'
        f"GLYPHS = {py_literal(payload['glyphs'])}\n\n"
        f"BY_CODEPOINT = {py_literal(payload['by_codepoint'])}\n\n"
        f"BY_CHAR = {py_literal(payload['by_char'])}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_php(path: Path, payload: Dict[str, Any]) -> None:
    content = (
        "<?php\n"
        "declare(strict_types=1);\n\n"
        "// Auto-generated Distribution glyph data.\n"
        "// Do not edit manually; regenerate from SVGs in ./src.\n\n"
        "return " + php_dump(payload, 0) + ";\n"
    )
    path.write_text(content, encoding="utf-8")


def find_svg_glyphs(src_dir: Path) -> List[Path]:
    files = []
    for path in sorted(src_dir.glob("character-u*.svg")):
        if RE_FILENAME.match(path.name):
            files.append(path)
    return files


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    data_dir = project_root / "data"

    if not src_dir.is_dir():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)

    svg_files = find_svg_glyphs(src_dir)
    if not svg_files:
        print(f"No glyph SVGs found in: {src_dir}", file=sys.stderr)
        return 1

    glyphs = [parse_svg_file(path) for path in svg_files]
    glyphs.sort(key=lambda g: g["codepoint"])

    payload = build_payload(glyphs)

    write_python(data_dir / "glyphs.py", payload)
    write_php(data_dir / "glyphs.php", payload)
    write_json(data_dir / "glyphs.json", payload)

    print(f"Generated {len(glyphs)} glyphs:")
    print(f"  {data_dir / 'glyphs.py'}")
    print(f"  {data_dir / 'glyphs.php'}")
    print(f"  {data_dir / 'glyphs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())