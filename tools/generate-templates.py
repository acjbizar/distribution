#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional


# -----------------------------
# Character set (must match glyph generator)
# -----------------------------
SHEET_ROWS = [
    "AbCdEfghIJ",
    "kLmnopqrStUvwxyZ.„”?",
    "0123456789",
    "AcGHKMNX",
]
REQUESTED = "".join(SHEET_ROWS)


def uniq(s: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ch in s:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


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


def inject_twig_attrs_into_svg_open(svg: str) -> str:
    """
    Turn:
      <svg ...>
    into:
      <svg ... {{ attrs|raw }}>
    so templates can pass attributes like class/width/height/aria-label.

    We also set a default of empty string for attrs in Twig/PHP wrappers
    to avoid undefined variable notices.
    """
    def repl(m: re.Match) -> str:
        before = m.group(0)
        attrs = m.group(1) or ""

        # If already contains a Twig interpolation, don't double-inject.
        if "{{" in before:
            return before

        return f"<svg{attrs} {{% if attrs is defined and attrs %}} {{ attrs|raw }}{{% endif %}}>"

    return SVG_OPEN_RE.sub(repl, svg, count=1)


def twig_wrapper(svg_inner: str, label: str, codepoint_hex: str) -> str:
    """
    Expose:
      - attrs: raw attribute string appended to <svg ...>
      - title: optional <title> for accessibility
      - aria_label: optional aria-label (fallback to title/label)
      - role: optional, default 'img' if any label/title provided
    """
    # Add a11y defaults without touching the SVG paths.
    # We'll append `attrs` into the <svg ...> tag, and optionally inject <title>.
    lines: List[str] = []
    lines.append("{# Auto-generated. Glyph: %s (U+%s) #}" % (label, codepoint_hex.upper()))
    lines.append("{% set attrs = attrs|default('') %}")
    lines.append("{% set title = title|default('') %}")
    lines.append("{% set aria_label = aria_label|default('') %}")
    lines.append("{% set role = role|default('') %}")
    lines.append("")
    # If user provided aria_label/title, we want role=img unless user overrides.
    lines.append("{% if (aria_label or title) and not role %}{% set role = 'img' %}{% endif %}")
    lines.append("{% if role %}{% set attrs = attrs ~ ' role=\"' ~ role ~ '\"' %}{% endif %}")
    lines.append("{% if aria_label %}")
    lines.append("  {% set attrs = attrs ~ ' aria-label=\"' ~ aria_label|e('html_attr') ~ '\"' %}")
    lines.append("{% elseif title %}")
    lines.append("  {% set attrs = attrs ~ ' aria-label=\"' ~ title|e('html_attr') ~ '\"' %}")
    lines.append("{% else %}")
    # If no label/title supplied, hide from AT unless user overrides via attrs.
    lines.append("  {# If you want it accessible, pass aria_label or title. #}")
    lines.append("{% endif %}")
    lines.append("")
    # Insert <title> as the first child of svg if title provided.
    # We'll do this by splitting on the first '>' after <svg ...>.
    # But easiest: if title is set, prepend a Twig block right after opening tag:
    # We'll rely on svg_inner already having attrs injection.
    # We'll replace first occurrence of '>' of the opening svg tag with a title block.
    lines.append("{% set __svg %}")
    lines.append(svg_inner.rstrip())
    lines.append("{% endset %}")
    lines.append("{% if title %}")
    lines.append(
        "{{ __svg|replace({'>':'>' ~ '<title>' ~ title|e ~ '</title>'})|raw }}"
    )
    lines.append("{% else %}")
    lines.append("{{ __svg|raw }}")
    lines.append("{% endif %}")
    lines.append("")
    return "\n".join(lines)


def php_wrapper(svg_inner: str, label: str, codepoint_hex: str) -> str:
    """
    Cake element wrapper:
      - $attrs: string of raw attributes appended into <svg ...>
      - $title: optional <title> (escaped)
      - $ariaLabel: optional aria-label (escaped)
      - $role: optional role
    """
    # We will:
    # - default missing vars
    # - append role/aria-label unless caller already provided them in $attrs
    # - optionally inject <title>
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
    # If title present, inject it after opening svg tag (same trick as Twig but with PHP).
    # We'll do it by printing a small replace if needed.
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Symfony Twig + CakePHP element templates from generated SVG glyphs."
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

    missing: List[str] = []
    written_symfony = 0
    written_cake = 0

    for ch in uniq(REQUESTED):
        cp = ord(ch)
        cp_hex = f"{cp:04x}"
        src_svg = svg_dir / f"character-u{cp_hex}.svg"

        if not src_svg.exists():
            missing.append(f"{ch} (expected {src_svg.as_posix()})")
            continue

        raw_svg = src_svg.read_text(encoding="utf-8")
        embedded_svg = prepare_svg_for_embedding(raw_svg, keep_xml_prolog=args.keep_xml_prolog)

        # Symfony output
        out_twig = symfony_dir / f"_character-u{cp_hex}.svg.twig"
        twig_text = twig_wrapper(embedded_svg, label=ch, codepoint_hex=cp_hex)

        # Cake output
        out_php = cake_dir / f"character-u{cp_hex}.php"

        # For Cake we want attrs injection too, but Twig tags are now present.
        # So we create a separate version where the injected block is PHP-aware instead of Twig.
        php_svg = re.sub(
            r"\{\% if attrs is defined and attrs \%\}\s*\{\{ attrs\|raw \}\}\{\% endif \%\}",
            r"<?php if (!empty($attrs)) echo $attrs; ?>",
            embedded_svg,
            count=1,
        )
        php_text = php_wrapper(php_svg, label=ch, codepoint_hex=cp_hex)

        if args.dry_run:
            print(f"[DRY] {ch} U+{cp:04X}")
            print(f"  -> {out_twig.as_posix()}")
            print(f"  -> {out_php.as_posix()}")
        else:
            write_text_lf(out_twig, twig_text)
            write_text_lf(out_php, php_text)
            written_symfony += 1
            written_cake += 1
            print(f"✓ {ch} U+{cp:04X} -> {out_twig.as_posix()} + {out_php.as_posix()}")

    if not args.dry_run:
        print("\nDone.")
        print(f"  Symfony templates written: {written_symfony}")
        print(f"  Cake elements written:     {written_cake}")

    if missing:
        print("\nMissing source SVGs for:")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
