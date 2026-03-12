#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SVG_DIR = ROOT / "src"
DEFAULT_SYMFONY_TEMPLATES_DIR = ROOT / "templates"
DEFAULT_CAKE_ELEMENTS_DIR = ROOT / "templates" / "element"
DEFAULT_CAKE_CONFIG_OUT = ROOT / "config" / "distribution.php"


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def strip_xml_prolog(svg: str) -> str:
    lines = svg.splitlines()
    if lines and lines[0].lstrip().startswith("<?xml"):
        lines = lines[1:]
        while lines and lines[0].strip() == "":
            lines = lines[1:]
    return "\n".join(lines).rstrip() + "\n"


SVG_OPEN_RE = re.compile(r"<svg\b([^>]*)>", re.IGNORECASE | re.DOTALL)
CODEPOINT_RE = re.compile(r"^character-u([0-9a-fA-F]{4,6})\.svg$")


def inject_twig_attrs_into_svg_open(svg: str) -> str:
    """
    Turn:
      <svg ...>
    into:
      <svg ... {% if attrs is defined and attrs %} {{ attrs|raw }}{% endif %}>
    so templates can pass attributes like class/width/height/aria-label.
    """
    def repl(m: re.Match) -> str:
        before = m.group(0)
        attrs = m.group(1) or ""

        # If already contains a Twig interpolation, don't double-inject.
        if "{{" in before or "{%" in before:
            return before

        return f"<svg{attrs} {{% if attrs is defined and attrs %}} {{ attrs|raw }}{{% endif %}}>"

    return SVG_OPEN_RE.sub(repl, svg, count=1)


def twig_wrapper(svg_inner: str, label: str, codepoint_hex: str) -> str:
    lines: List[str] = []
    lines.append("{# Auto-generated. Glyph: %s (U+%s) #}" % (label, codepoint_hex.upper()))
    lines.append("{% set attrs = attrs|default('') %}")
    lines.append("{% set title = title|default('') %}")
    lines.append("{% set aria_label = aria_label|default('') %}")
    lines.append("{% set role = role|default('') %}")
    lines.append("")
    lines.append("{% if (aria_label or title) and not role %}{% set role = 'img' %}{% endif %}")
    lines.append("{% if role %}{% set attrs = attrs ~ ' role=\"' ~ role ~ '\"' %}{% endif %}")
    lines.append("{% if aria_label %}")
    lines.append("  {% set attrs = attrs ~ ' aria-label=\"' ~ aria_label|e('html_attr') ~ '\"' %}")
    lines.append("{% elseif title %}")
    lines.append("  {% set attrs = attrs ~ ' aria-label=\"' ~ title|e('html_attr') ~ '\"' %}")
    lines.append("{% else %}")
    lines.append("  {# If you want it accessible, pass aria_label or title. #}")
    lines.append("{% endif %}")
    lines.append("")
    lines.append("{% set __svg %}")
    lines.append(svg_inner.rstrip())
    lines.append("{% endset %}")
    lines.append("{% if title %}")
    lines.append("{{ __svg|replace({'>':'>' ~ '<title>' ~ title|e ~ '</title>'})|raw }}")
    lines.append("{% else %}")
    lines.append("{{ __svg|raw }}")
    lines.append("{% endif %}")
    lines.append("")
    return "\n".join(lines)


def php_wrapper(svg_inner: str, label: str, codepoint_hex: str) -> str:
    php: List[str] = []
    php.append("<?php")
    php.append("/**")
    php.append(f" * Auto-generated. Glyph: {label} (U+{codepoint_hex.upper()})")
    php.append(" *")
    php.append(" * Available variables:")
    php.append(" *   @var string $attrs      Raw attributes appended to <svg ...> (e.g. ' class=\"x\" width=\"24\"')")
    php.append(" *   @var string $title      Optional <title> (will be escaped)")
    php.append(" *   @var string $ariaLabel  Optional aria-label (will be escaped)")
    php.append(" *   @var string $role       Optional role (default 'img' when aria/title provided)")
    php.append(" */")
    php.append("$attrs = isset($attrs) ? (string)$attrs : '';")
    php.append("$title = isset($title) ? (string)$title : '';")
    php.append("$ariaLabel = isset($ariaLabel) ? (string)$ariaLabel : '';")
    php.append("$role = isset($role) ? (string)$role : '';")
    php.append("")
    php.append("$hasRole = stripos($attrs, ' role=') !== false;")
    php.append("$hasAria = stripos($attrs, ' aria-label=') !== false;")
    php.append("")
    php.append("if (($ariaLabel !== '' || $title !== '') && $role === '' && !$hasRole) {")
    php.append("    $role = 'img';")
    php.append("}")
    php.append("if ($role !== '' && !$hasRole) {")
    php.append("    $attrs .= ' role=\"' . h($role) . '\"';")
    php.append("}")
    php.append("if (!$hasAria) {")
    php.append("    if ($ariaLabel !== '') {")
    php.append("        $attrs .= ' aria-label=\"' . h($ariaLabel) . '\"';")
    php.append("    } elseif ($title !== '') {")
    php.append("        $attrs .= ' aria-label=\"' . h($title) . '\"';")
    php.append("    }")
    php.append("}")
    php.append("?>")
    php.append(svg_inner.rstrip())
    php.append("")
    php.append("<?php if ($title !== ''): ?>")
    php.append("<?php")
    php.append("$__svg = ob_get_clean();")
    php.append("$__svg = preg_replace('/(<svg\\b[^>]*>)/i', '$1<title>' . h($title) . '</title>', $__svg, 1);")
    php.append("echo $__svg;")
    php.append("ob_start();")
    php.append("?>")
    php.append("<?php endif; ?>")
    return "\n".join(php)


def prepare_svg_for_embedding(svg_text: str, keep_xml_prolog: bool) -> str:
    if not keep_xml_prolog:
        svg_text = strip_xml_prolog(svg_text)
    else:
        svg_text = svg_text.rstrip() + "\n"

    svg_text = inject_twig_attrs_into_svg_open(svg_text)
    return svg_text


def list_glyph_files(svg_dir: Path) -> List[Tuple[int, Path]]:
    """
    Discover src/character-u{codepoint}.svg files.
    Returns sorted list of (codepoint_int, path).
    """
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


def _is_printable_char(ch: str) -> bool:
    return ch.isprintable() and ch not in {"\n", "\r", "\t", "\x0b", "\x0c"}


def _char_for_codepoint(cp: int) -> str:
    try:
        ch = chr(cp)
    except ValueError:
        return ""
    return ch if _is_printable_char(ch) else ""


def _group_glyph_chars(glyph_files: List[Tuple[int, Path]]) -> Dict[str, Any]:
    upper: List[str] = []
    lower: List[str] = []
    digits: List[str] = []
    punct: List[str] = []
    other: List[str] = []
    codepoints: List[int] = []

    for cp, _path in glyph_files:
        codepoints.append(cp)
        ch = _char_for_codepoint(cp)
        if not ch:
            continue

        if "A" <= ch <= "Z":
            upper.append(ch)
        elif "a" <= ch <= "z":
            lower.append(ch)
        elif "0" <= ch <= "9":
            digits.append(ch)
        elif ch.isprintable() and not ch.isalnum() and not ch.isspace():
            punct.append(ch)
        else:
            other.append(ch)

    all_chars = upper + lower + digits + punct + other

    return {
        "all": "".join(all_chars),
        "uppercase": "".join(upper),
        "lowercase": "".join(lower),
        "digits": "".join(digits),
        "punct": "".join(punct),
        "other": "".join(other),
        "codepoints": codepoints,
    }


def _codepoints_to_ranges(codepoints: List[int]) -> List[List[int]]:
    if not codepoints:
        return []

    cps = sorted(set(codepoints))
    ranges: List[List[int]] = []

    start = prev = cps[0]
    for cp in cps[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        ranges.append([start, prev])
        start = prev = cp
    ranges.append([start, prev])

    return ranges


def _php_scalar(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    next_pad = " " * (indent + 4)

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f"'{escaped}'"

    if isinstance(value, list):
        if not value:
            return "[]"
        lines = ["["]
        for item in value:
            lines.append(f"{next_pad}{_php_scalar(item, indent + 4)},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "[]"
        lines = ["["]
        for k, v in value.items():
            lines.append(f"{next_pad}{_php_scalar(str(k))} => {_php_scalar(v, indent + 4)},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    raise TypeError(f"Unsupported value for PHP export: {type(value)!r}")


def build_cake_config_payload(glyph_files: List[Tuple[int, Path]]) -> Dict[str, Any]:
    grouped = _group_glyph_chars(glyph_files)
    codepoints = grouped["codepoints"]
    unicode_ranges = _codepoints_to_ranges(codepoints)

    glyphs: Dict[str, Dict[str, Any]] = {}
    char_to_hex: Dict[str, str] = {}

    for cp, _path in glyph_files:
        cp_hex = f"{cp:04x}"
        ch = _char_for_codepoint(cp)
        key = f"u{cp_hex}"

        glyphs[key] = {
            "codepoint": cp,
            "char": ch,
            "hex": cp_hex,
            "symfonyTemplate": f"_character-u{cp_hex}.svg.twig",
            "cakeElement": f"character-u{cp_hex}",
        }

        if ch:
            char_to_hex[ch] = cp_hex

    return {
        "Distribution": {
            "glyphCount": len(glyph_files),
            "chars": grouped["all"],
            "uppercase": grouped["uppercase"],
            "lowercase": grouped["lowercase"],
            "digits": grouped["digits"],
            "punct": grouped["punct"],
            "other": grouped["other"],
            "codepoints": codepoints,
            "unicodeRanges": unicode_ranges,
            "hasUppercase": bool(grouped["uppercase"]),
            "hasLowercase": bool(grouped["lowercase"]),
            "hasDigits": bool(grouped["digits"]),
            "hasPunct": bool(grouped["punct"]),
            "charToHex": char_to_hex,
            "glyphs": glyphs,
        }
    }


def write_cake_config(glyph_files: List[Tuple[int, Path]], out_path: Path) -> None:
    payload = build_cake_config_payload(glyph_files)

    lines: List[str] = []
    lines.append("<?php")
    lines.append("declare(strict_types=1);")
    lines.append("")
    lines.append("// Auto-generated by tools/generate-templates.py")
    lines.append("// Runtime-friendly config for CakePHP/plugin usage.")
    lines.append("")
    lines.append("return " + _php_scalar(payload, 0) + ";")
    lines.append("")

    write_text_lf(out_path, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Symfony Twig + CakePHP element templates from SVG glyph files discovered on disk."
    )
    ap.add_argument("--svg-dir", default=DEFAULT_SVG_DIR, help=f"Directory containing generated SVGs (default: {DEFAULT_SVG_DIR})")
    ap.add_argument("--symfony-templates-dir", default=DEFAULT_SYMFONY_TEMPLATES_DIR, help=f"Symfony templates directory (default: {DEFAULT_SYMFONY_TEMPLATES_DIR})")
    ap.add_argument("--cake-elements-dir", default=DEFAULT_CAKE_ELEMENTS_DIR, help=f"CakePHP elements directory (default: {DEFAULT_CAKE_ELEMENTS_DIR})")
    ap.add_argument("--cake-config-out", default=DEFAULT_CAKE_CONFIG_OUT, help=f"CakePHP config output path (default: {DEFAULT_CAKE_CONFIG_OUT})")
    ap.add_argument("--no-cake-config", action="store_true", help="Do not write CakePHP config")
    ap.add_argument("--keep-xml-prolog", action="store_true", help="Do not strip the XML prolog from SVGs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written, but don't write files")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    symfony_dir = Path(args.symfony_templates_dir)
    cake_dir = Path(args.cake_elements_dir)
    cake_config_out = Path(args.cake_config_out)

    glyph_files = list_glyph_files(svg_dir)
    if not glyph_files:
        raise SystemExit(f"No glyph SVGs found in {svg_dir.resolve()} matching character-u{'{codepoint}'}.svg")

    written_symfony = 0
    written_cake = 0

    for cp, src_svg in glyph_files:
        cp_hex = f"{cp:04x}"
        label = _char_for_codepoint(cp) or f"U+{cp:04X}"

        raw_svg = src_svg.read_text(encoding="utf-8")
        embedded_svg = prepare_svg_for_embedding(raw_svg, keep_xml_prolog=args.keep_xml_prolog)

        out_twig = symfony_dir / f"_character-u{cp_hex}.svg.twig"
        twig_text = twig_wrapper(embedded_svg, label=label, codepoint_hex=cp_hex)

        out_php = cake_dir / f"character-u{cp_hex}.php"

        php_svg = re.sub(
            r"\{\% if attrs is defined and attrs \%\}\s*\{\{ attrs\|raw \}\}\{\% endif \%\}",
            r"<?php if (!empty($attrs)) echo $attrs; ?>",
            embedded_svg,
            count=1,
        )
        php_text = php_wrapper(php_svg, label=label, codepoint_hex=cp_hex)

        if args.dry_run:
            print(f"[DRY] {label} U+{cp:04X}")
            print(f"  -> {out_twig.as_posix()}")
            print(f"  -> {out_php.as_posix()}")
        else:
            write_text_lf(out_twig, twig_text)
            write_text_lf(out_php, php_text)
            written_symfony += 1
            written_cake += 1
            print(f"✓ {label} U+{cp:04X} -> {out_twig.as_posix()} + {out_php.as_posix()}")

    if args.dry_run:
        if not args.no_cake_config:
            print(f"[DRY] config -> {cake_config_out.as_posix()}")
    else:
        if not args.no_cake_config:
            write_cake_config(glyph_files, cake_config_out)
            print(f"✓ Cake config -> {cake_config_out.as_posix()}")

        print("\nDone.")
        print(f"  Symfony templates written: {written_symfony}")
        print(f"  Cake elements written:     {written_cake}")
        if not args.no_cake_config:
            print("  Cake config written:       1")


if __name__ == "__main__":
    main()