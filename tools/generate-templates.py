#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


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
    php.append(" *   @var string $attrs      Raw attributes appended to <svg ...> (e.g. ' class=\"x\" width=\"24\"')")  # noqa
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Symfony Twig + CakePHP element templates from SVG glyph files discovered on disk."
    )
    ap.add_argument("--svg-dir", default="src", help="Directory containing generated SVGs (default: src)")
    ap.add_argument("--symfony-templates-dir", default="templates", help="Symfony templates directory (default: templates)")
    ap.add_argument("--cake-elements-dir", default="templates/element", help="CakePHP elements directory (default: templates/element)")
    ap.add_argument("--keep-xml-prolog", action="store_true", help="Do not strip the XML prolog from SVGs")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be written, but don't write files")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    symfony_dir = Path(args.symfony_templates_dir)
    cake_dir = Path(args.cake_elements_dir)

    glyph_files = list_glyph_files(svg_dir)
    if not glyph_files:
        raise SystemExit(f"No glyph SVGs found in {svg_dir.resolve()} matching character-u{'{codepoint}'}.svg")

    written_symfony = 0
    written_cake = 0

    for cp, src_svg in glyph_files:
        cp_hex = f"{cp:04x}"
        label = chr(cp) if 0 <= cp <= 0x10FFFF else f"U+{cp:04X}"

        raw_svg = src_svg.read_text(encoding="utf-8")
        embedded_svg = prepare_svg_for_embedding(raw_svg, keep_xml_prolog=args.keep_xml_prolog)

        out_twig = symfony_dir / f"_character-u{cp_hex}.svg.twig"
        twig_text = twig_wrapper(embedded_svg, label=label, codepoint_hex=cp_hex)

        out_php = cake_dir / f"character-u{cp_hex}.php"

        # Convert Twig attrs-injection to PHP for Cake element output.
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

    if not args.dry_run:
        print("\nDone.")
        print(f"  Symfony templates written: {written_symfony}")
        print(f"  Cake elements written:     {written_cake}")


if __name__ == "__main__":
    main()
