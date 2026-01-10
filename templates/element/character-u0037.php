<?php
/**
 * Auto-generated. Glyph: 7 (U+0037)
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
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 320" width="240" height="320" {% if attrs is defined and attrs %} { attrs|raw }{% endif %}>
<desc>glyph: 7 U+0037</desc>
<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>
<g fill="none" stroke="#bdbdbd" stroke-width="1" opacity="0.35">
<circle cx="40" cy="40" r="40" />
<circle cx="120" cy="40" r="40" />
<circle cx="200" cy="40" r="40" />
<circle cx="40" cy="120" r="40" />
<circle cx="120" cy="120" r="40" />
<circle cx="200" cy="120" r="40" />
<circle cx="40" cy="200" r="40" />
<circle cx="120" cy="200" r="40" />
<circle cx="200" cy="200" r="40" />
<circle cx="40" cy="280" r="40" />
<circle cx="120" cy="280" r="40" />
<circle cx="200" cy="280" r="40" />
</g>
<g fill="none" stroke="#000" stroke-width="9" stroke-linecap="round" stroke-linejoin="round">
<path d="M 80 160 L 160 160 L 80 240" />
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