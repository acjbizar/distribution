<?php
declare(strict_types=1);

/**
 * Variables:
 *   @var string $char       Character to render
 *   @var string $hex        Optional hex codepoint without "u", e.g. "0041"
 *   @var string $attrs      Optional raw SVG attributes, without leading space
 *   @var string $title      Optional <title>
 *   @var string $ariaLabel  Optional aria-label
 *   @var string $role       Optional role; defaults to "img" when title/ariaLabel is given
 */

$char = isset($char) ? (string)$char : '';
$hex = isset($hex) ? strtolower((string)$hex) : '';
$attrs = isset($attrs) ? trim((string)$attrs) : '';
$title = isset($title) ? (string)$title : '';
$ariaLabel = isset($ariaLabel) ? (string)$ariaLabel : '';
$role = isset($role) ? (string)$role : '';

if ($hex === '' && $char !== '') {
    if (function_exists('mb_substr')) {
        $char = mb_substr($char, 0, 1, 'UTF-8');
    } else {
        $char = substr($char, 0, 1);
    }

    if (function_exists('mb_ord')) {
        $codepoint = mb_ord($char, 'UTF-8');
    } else {
        $codepoint = ord($char);
    }

    $hex = strtolower(sprintf('%04x', $codepoint));
}

if ($hex === '') {
    return;
}

$path = dirname(__DIR__, 2) . DS . 'src' . DS . 'character-u' . $hex . '.svg';

if (!is_file($path)) {
    return;
}

$svg = (string)file_get_contents($path);
if ($svg === '') {
    return;
}

/* Strip XML prolog for HTML embedding */
$svg = preg_replace('/^\s*<\?xml[^>]*\?>\s*/i', '', $svg, 1) ?? $svg;

$attrsProbe = ' ' . $attrs;

if (($ariaLabel !== '' || $title !== '') && $role === '' && stripos($attrsProbe, ' role=') === false) {
    $role = 'img';
}

if ($role !== '' && stripos($attrsProbe, ' role=') === false) {
    $attrs .= ($attrs !== '' ? ' ' : '') . 'role="' . h($role) . '"';
    $attrsProbe = ' ' . $attrs;
}

if (stripos($attrsProbe, ' aria-label=') === false) {
    if ($ariaLabel !== '') {
        $attrs .= ($attrs !== '' ? ' ' : '') . 'aria-label="' . h($ariaLabel) . '"';
    } elseif ($title !== '') {
        $attrs .= ($attrs !== '' ? ' ' : '') . 'aria-label="' . h($title) . '"';
    }
}

$svgAttrs = $attrs !== '' ? ' ' . $attrs : '';
$titleMarkup = $title !== '' ? '<title>' . h($title) . '</title>' : '';

$svg = preg_replace_callback(
        '/<svg\b([^>]*)>/i',
        static function (array $m) use ($svgAttrs, $titleMarkup): string {
            return '<svg' . $m[1] . $svgAttrs . '>' . $titleMarkup;
        },
        $svg,
        1
) ?? $svg;

echo $svg;