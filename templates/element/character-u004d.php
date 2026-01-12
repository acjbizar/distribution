<?php
/**
 * Auto-generated. Glyph: M (U+004D)
 *
 * Available variables:
 *   @var string $attrs      Raw attributes appended to <svg ...> (e.g. ' class="x" width="24"')
 *   @var string $title      Optional <title> (will be escaped)
 *   @var string $ariaLabel  Optional aria-label (will be escaped)
 *   @var string $role       Optional role (default 'img' when aria/title provided)
 */
$attrs = isset($attrs) ? (string)$attrs : '';
$title = isset($title) ? (string)$title : '';
$ariaLabel = isset($ariaLabel) ? (string)$ariaLabel : '';
$role = isset($role) ? (string)$role : '';

$hasRole = stripos($attrs, ' role=') !== false;
$hasAria = stripos($attrs, ' aria-label=') !== false;

if (($ariaLabel !== '' || $title !== '') && $role === '' && !$hasRole) {
    $role = 'img';
}
if ($role !== '' && !$hasRole) {
    $attrs .= ' role="' . h($role) . '"';
}
if (!$hasAria) {
    if ($ariaLabel !== '') {
        $attrs .= ' aria-label="' . h($ariaLabel) . '"';
    } elseif ($title !== '') {
        $attrs .= ' aria-label="' . h($title) . '"';
    }
}
?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320" {% if attrs is defined and attrs %} { attrs|raw }{% endif %}>
<desc>glyph: M U+004D</desc>
<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>
<g fill="none" stroke="#bdbdbd" stroke-width="1" opacity="0.35">
<circle cx="40" cy="40" r="40" />
<circle cx="120" cy="40" r="40" />
<circle cx="200" cy="40" r="40" />
<circle cx="280" cy="40" r="40" />
<circle cx="40" cy="120" r="40" />
<circle cx="120" cy="120" r="40" />
<circle cx="200" cy="120" r="40" />
<circle cx="280" cy="120" r="40" />
<circle cx="40" cy="200" r="40" />
<circle cx="120" cy="200" r="40" />
<circle cx="200" cy="200" r="40" />
<circle cx="280" cy="200" r="40" />
<circle cx="40" cy="280" r="40" />
<circle cx="120" cy="280" r="40" />
<circle cx="200" cy="280" r="40" />
<circle cx="280" cy="280" r="40" />
</g>
<g fill="none" stroke="#000" stroke-width="9" stroke-linecap="butt" stroke-linejoin="round">
<path d="M 80 240 L 80 120" />
<path d="M 80 120 A 40 40 0 0 1 120 80 A 40 40 0 0 1 160 120" />
<path d="M 160 240 L 160 120" />
<path d="M 160 120 A 40 40 0 0 1 200 80 A 40 40 0 0 1 240 120" />
<path d="M 240 240 L 240 120" />
</g>
</svg>

<?php if ($title !== ''): ?>
<?php
$__svg = ob_get_clean();
$__svg = preg_replace('/(<svg\b[^>]*>)/i', '$1<title>' . h($title) . '</title>', $__svg, 1);
echo $__svg;
ob_start();
?>
<?php endif; ?>