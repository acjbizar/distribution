<?php
/**
 * Auto-generated. Glyph: Π (U+03A0)
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
<desc>glyph: Π U+03A0</desc>
<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>
<g fill="none" stroke="#000" stroke-width="9" stroke-linecap="butt" stroke-linejoin="round">
<path d="M 80 160 L 80 200" />
<path d="M 160 200 L 160 240" />
<path d="M 80 200 L 80 240" />
<path d="M 160 160 L 160 200" />
<path d="M 80 80 L 120 80" />
<path d="M 120 80 L 160 80" />
<path d="M 80 120 L 80 160" />
<path d="M 80 80 L 80 120" />
<path d="M 160 80 L 160 120" />
<path d="M 160 120 L 160 160" />
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