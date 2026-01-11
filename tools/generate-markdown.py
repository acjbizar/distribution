#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


CODEPOINT_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def list_glyph_files(svg_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    if not svg_dir.exists():
        return items

    for p in svg_dir.iterdir():
        if not p.is_file():
            continue
        m = CODEPOINT_RE.match(p.name)
        if not m:
            continue
        cp = int(m.group(1), 16)
        items.append((cp, p))

    items.sort(key=lambda t: t[0])
    return items


def safe_label(ch: str) -> str:
    # Markdown image alt text can be almost anything, but keep it readable.
    # Replace newlines/tabs just in case.
    return ch.replace("\n", "\\n").replace("\t", "\\t")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate templates/_letter-sample.md.twig from available src/character-u*.svg files."
    )
    ap.add_argument("--svg-dir", default="src", help="Directory containing character-u*.svg (default: src)")
    ap.add_argument(
        "--out",
        default="templates/_letter-sample.md.twig",
        help="Output twig markdown path (default: templates/_letter-sample.md.twig)",
    )
    ap.add_argument("--newline", action="store_true", help="Insert a newline after each image (default: no)")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    out_path = Path(args.out)

    glyphs = list_glyph_files(svg_dir)
    if not glyphs:
        raise SystemExit(f"No glyph SVGs found in {svg_dir.resolve()} matching character-uXXXX.svg")

    parts: List[str] = []
    for cp, _ in glyphs:
        ch = chr(cp) if 0 <= cp <= 0x10FFFF else "?"
        label = safe_label(ch)
        cp_hex = f"{cp:04x}"
        parts.append(f"![{label}](src/character-u{cp_hex}.svg)")

    sep = "\n" if args.newline else ""
    content = sep.join(parts) + "\n"

    write_text_lf(out_path, content)
    print(f"✓ Wrote {out_path.as_posix()} ({len(parts)} glyph(s))")


if __name__ == "__main__":
    main()
