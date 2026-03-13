(() => {
  const STEP = 40;
  const GUIDE_STEP = 80;
  const HEIGHT = 320;
  const GUIDE_ROWS = 4;
  const DEFAULT_STROKE_WIDTH = 9;
  const STORAGE_THEME_KEY = 'distribution-glyph-designer-theme';
  const MANIFEST_PATHS = ['../data/manifest.json', './data/manifest.json', 'data/manifest.json'];

  const state = {
    theme: 'device',
    char: 'A',
    guideCols: 3,
    autoWidth: true,
    shapes: [],
    hoverCandidate: null,
    manifest: {
      loaded: false,
      path: null,
      glyphs: [],
    },
  };

  const el = {
    editorStage: document.getElementById('editorStage'),
    charInput: document.getElementById('charInput'),
    guideColsSelect: document.getElementById('guideColsSelect'),
    autoWidthCheckbox: document.getElementById('autoWidthCheckbox'),
    themeSelect: document.getElementById('themeSelect'),
    codepointField: document.getElementById('codepointField'),
    filenameField: document.getElementById('filenameField'),
    glyphInfo: document.getElementById('glyphInfo'),
    sizeInfo: document.getElementById('sizeInfo'),
    statusPill: document.getElementById('statusPill'),
    pointCountLabel: document.getElementById('pointCountLabel'),
    footerHint: document.getElementById('footerHint'),
    primitiveList: document.getElementById('primitiveList'),
    svgOutput: document.getElementById('svgOutput'),
    undoBtn: document.getElementById('undoBtn'),
    clearBtn: document.getElementById('clearBtn'),
    copySvgBtn: document.getElementById('copySvgBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    manifestStatus: document.getElementById('manifestStatus'),
    manifestHint: document.getElementById('manifestHint'),
    manifestGlyphGrid: document.getElementById('manifestGlyphGrid'),
  };

  function escapeXml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  function getDisplayChar(input) {
    const trimmed = (input || '').trim();
    if (!trimmed) return '?';
    return Array.from(trimmed)[0] || '?';
  }

  function getCodepointHex(ch) {
    const cp = ch.codePointAt(0);
    return cp.toString(16).toUpperCase().padStart(4, '0');
  }

  function getFilename(ch) {
    return `character-u${getCodepointHex(ch).toLowerCase()}.svg`;
  }

  function suggestGuideCols(ch) {
    if (!ch) return 3;
    const narrow = new Set(['.', ',', ':', ';', '!', 'I', 'i', 'l', '│', '|']);
    const wide = new Set(['M', 'W', 'm', 'w']);
    if (narrow.has(ch)) return 2;
    if (wide.has(ch)) return 4;
    return 3;
  }

  function getWidth() {
    return state.guideCols * GUIDE_STEP;
  }

  function getGuideCentersX() {
    const width = getWidth();
    const arr = [];
    for (let x = GUIDE_STEP / 2; x <= width - GUIDE_STEP / 2; x += GUIDE_STEP) {
      arr.push(x);
    }
    return arr;
  }

  function getGuideCentersY() {
    const arr = [];
    for (let y = GUIDE_STEP / 2; y <= (GUIDE_ROWS * GUIDE_STEP) - (GUIDE_STEP / 2); y += GUIDE_STEP) {
      arr.push(y);
    }
    return arr;
  }

  function getGridPoints() {
    const width = getWidth();
    const points = [];
    for (let y = STEP; y <= HEIGHT; y += STEP) {
      for (let x = STEP; x <= width; x += STEP) {
        points.push({ x, y, id: `${x},${y}` });
      }
    }
    return points;
  }

  function samePoint(a, b) {
    return !!a && !!b && a.x === b.x && a.y === b.y;
  }

  function pointKey(p) {
    return `${p.x},${p.y}`;
  }

  function setStatus(text) {
    el.statusPill.textContent = text;
  }

  function setFooterHint(text) {
    el.footerHint.textContent = text || 'Hover a valid geometry and click once to place it.';
  }

  function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_THEME_KEY, theme);
  }

  function loadTheme() {
    const saved = localStorage.getItem(STORAGE_THEME_KEY);
    const theme = saved || 'device';
    el.themeSelect.value = theme;
    setTheme(theme);
  }

  function normalizeAngleDelta(delta) {
    while (delta <= -Math.PI) delta += Math.PI * 2;
    while (delta > Math.PI) delta -= Math.PI * 2;
    return delta;
  }

  function normalizeAnglePositive(angle) {
    while (angle < 0) angle += Math.PI * 2;
    while (angle >= Math.PI * 2) angle -= Math.PI * 2;
    return angle;
  }

  function arcSweepForVariant(p1, p2, variant) {
    const center = variant === 0 ? { x: p1.x, y: p2.y } : { x: p2.x, y: p1.y };
    const a1 = Math.atan2(p1.y - center.y, p1.x - center.x);
    const a2 = Math.atan2(p2.y - center.y, p2.x - center.x);
    const delta = normalizeAngleDelta(a2 - a1);
    return delta > 0 ? 1 : 0;
  }

  function linePath(shape) {
    return `M ${shape.p1.x} ${shape.p1.y} L ${shape.p2.x} ${shape.p2.y}`;
  }

  function arcPath(shape) {
    const sweep = arcSweepForVariant(shape.p1, shape.p2, shape.variant);
    return `M ${shape.p1.x} ${shape.p1.y} A 40 40 0 0 ${sweep} ${shape.p2.x} ${shape.p2.y}`;
  }

  function shapeToLabel(shape) {
    if (shape.type === 'line') {
      return `Line · (${shape.p1.x},${shape.p1.y}) → (${shape.p2.x},${shape.p2.y})`;
    }
    if (shape.type === 'arc') {
      return `Arc ${shape.variant === 0 ? 'A' : 'B'} · (${shape.p1.x},${shape.p1.y}) → (${shape.p2.x},${shape.p2.y})`;
    }
    return `Dot · (${shape.p.x},${shape.p.y})`;
  }

  function serializeShapeKey(shape) {
    if (shape.type === 'dot') {
      return `dot:${shape.p.x},${shape.p.y}`;
    }
    if (shape.type === 'line') {
      const a = pointKey(shape.p1);
      const b = pointKey(shape.p2);
      return `line:${[a, b].sort().join('|')}`;
    }
    const a = pointKey(shape.p1);
    const b = pointKey(shape.p2);
    return `arc:${shape.variant}:${[a, b].sort().join('|')}`;
  }

  function serializeShapes() {
    const pathLines = [];
    const dotLines = [];

    for (const shape of state.shapes) {
      if (shape.type === 'line') {
        pathLines.push(`<path d="${linePath(shape)}" />`);
      } else if (shape.type === 'arc') {
        pathLines.push(`<path d="${arcPath(shape)}" />`);
      } else if (shape.type === 'dot') {
        dotLines.push(`<circle cx="${shape.p.x}" cy="${shape.p.y}" r="1" />`);
      }
    }

    return { pathLines, dotLines };
  }

  function buildExportSvgString() {
    const ch = getDisplayChar(state.char);
    const width = getWidth();
    const hex = getCodepointHex(ch);
    const glyphDesc = `${escapeXml(ch)} U+${hex}`;
    const { pathLines, dotLines } = serializeShapes();

    const guideCircles = [];
    for (const y of getGuideCentersY()) {
      for (const x of getGuideCentersX()) {
        guideCircles.push(`  <circle cx="${x}" cy="${y}" r="40" />`);
      }
    }

    const parts = [
      `<?xml version="1.0" encoding="UTF-8"?>`,
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${HEIGHT}">`,
      `  <desc>glyph: ${glyphDesc}</desc>`,
      `  <rect x="0" y="0" width="100%" height="100%" fill="#fff"/>`,
      `  <g fill="none" stroke="#bdbdbd" stroke-width="1" opacity="0.35">`,
      ...guideCircles,
      `  </g>`,
      `  <g fill="none" stroke="#000" stroke-width="${DEFAULT_STROKE_WIDTH}" stroke-linecap="butt" stroke-linejoin="round">`,
      ...pathLines.map(line => `  ${line}`),
      `  </g>`,
    ];

    if (dotLines.length) {
      parts.push(`  <g fill="#000" stroke="#000" stroke-width="${DEFAULT_STROKE_WIDTH}">`);
      parts.push(...dotLines.map(line => `  ${line}`));
      parts.push(`  </g>`);
    }

    parts.push(`</svg>`);
    return parts.join('\n');
  }

  function pointDistance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function distancePointToSegment(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (!lenSq) return pointDistance(p, a);
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const proj = { x: a.x + t * dx, y: a.y + t * dy };
    return pointDistance(p, proj);
  }

  function arcCenter(candidate) {
    return candidate.variant === 0
      ? { x: candidate.p1.x, y: candidate.p2.y }
      : { x: candidate.p2.x, y: candidate.p1.y };
  }

  function isAngleOnArc(test, start, end, sweep) {
    const a = normalizeAnglePositive(start);
    const b = normalizeAnglePositive(end);
    const t = normalizeAnglePositive(test);

    if (sweep === 1) {
      const span = normalizeAnglePositive(b - a);
      const rel = normalizeAnglePositive(t - a);
      return rel <= span + 1e-6;
    }

    const span = normalizeAnglePositive(a - b);
    const rel = normalizeAnglePositive(a - t);
    return rel <= span + 1e-6;
  }

  function distancePointToArc(p, candidate) {
    const center = arcCenter(candidate);
    const startAngle = Math.atan2(candidate.p1.y - center.y, candidate.p1.x - center.x);
    const endAngle = Math.atan2(candidate.p2.y - center.y, candidate.p2.x - center.x);
    const sweep = arcSweepForVariant(candidate.p1, candidate.p2, candidate.variant);
    const testAngle = Math.atan2(p.y - center.y, p.x - center.x);
    const radial = Math.abs(pointDistance(p, center) - STEP);

    if (isAngleOnArc(testAngle, startAngle, endAngle, sweep)) {
      return radial;
    }

    return Math.min(pointDistance(p, candidate.p1), pointDistance(p, candidate.p2));
  }

  function getLineCandidates() {
    const width = getWidth();
    const candidates = [];
    const seen = new Set();

    for (let y = STEP; y <= HEIGHT; y += STEP) {
      for (let x = STEP; x <= width; x += STEP) {
        const p = { x, y };
        const options = [
          { x: x + STEP, y },
          { x, y: y + STEP },
          { x: x + STEP, y: y + STEP },
          { x: x - STEP, y: y + STEP },
        ];

        for (const q of options) {
          if (q.x < STEP || q.x > width || q.y < STEP || q.y > HEIGHT) continue;
          const key = ['line', pointKey(p), pointKey(q)].sort().join('|');
          if (seen.has(key)) continue;
          seen.add(key);
          candidates.push({ type: 'line', p1: p, p2: q });
        }
      }
    }

    return candidates;
  }

  function getArcCandidates() {
    const width = getWidth();
    const candidates = [];

    for (let y = STEP; y < HEIGHT; y += STEP) {
      for (let x = STEP; x < width; x += STEP) {
        const tl = { x, y };
        const tr = { x: x + STEP, y };
        const bl = { x, y: y + STEP };
        const br = { x: x + STEP, y: y + STEP };

        candidates.push({ type: 'arc', p1: bl, p2: tr, variant: 0 });
        candidates.push({ type: 'arc', p1: bl, p2: tr, variant: 1 });
        candidates.push({ type: 'arc', p1: tl, p2: br, variant: 0 });
        candidates.push({ type: 'arc', p1: tl, p2: br, variant: 1 });
      }
    }

    return candidates;
  }

  function getDotCandidates() {
    return getGridPoints().map(p => ({ type: 'dot', p }));
  }

  function getAllCandidates() {
    return [
      ...getDotCandidates(),
      ...getLineCandidates(),
      ...getArcCandidates(),
    ];
  }

  function candidateDistance(pointer, candidate) {
    if (candidate.type === 'dot') {
      return pointDistance(pointer, candidate.p);
    }
    if (candidate.type === 'line') {
      return distancePointToSegment(pointer, candidate.p1, candidate.p2);
    }
    return distancePointToArc(pointer, candidate);
  }

  function candidateThreshold(candidate) {
    if (candidate.type === 'dot') return 9;
    if (candidate.type === 'line') return 8;
    return 8;
  }

  function getBestCandidate(pointer) {
    let best = null;
    let bestScore = Infinity;

    for (const candidate of getAllCandidates()) {
      const distance = candidateDistance(pointer, candidate);
      const threshold = candidateThreshold(candidate);
      const score = distance / threshold;
      if (distance <= threshold && score < bestScore) {
        bestScore = score;
        best = candidate;
      }
    }

    return best;
  }

  function getCurrentGlyphDefined() {
    const current = getDisplayChar(state.char);
    return state.manifest.glyphs.some(item => item.char === current);
  }

  function getPreviewMarkup() {
    const candidate = state.hoverCandidate;
    if (!candidate) return '';

    if (candidate.type === 'dot') {
      return `<circle class="preview-dot" cx="${candidate.p.x}" cy="${candidate.p.y}" r="5"></circle>`;
    }

    if (candidate.type === 'line') {
      return `<path class="preview-path" d="${linePath(candidate)}"></path>`;
    }

    return `<path class="preview-path" d="${arcPath(candidate)}"></path>`;
  }

  function renderEditor() {
    const width = getWidth();
    const gridPoints = getGridPoints();
    const previewMarkup = getPreviewMarkup();

    const guideCircles = [];
    for (const y of getGuideCentersY()) {
      for (const x of getGuideCentersX()) {
        guideCircles.push(`<circle class="guide-circle" cx="${x}" cy="${y}" r="40"></circle>`);
      }
    }

    const strokePaths = [];
    const dotShapes = [];
    for (const shape of state.shapes) {
      if (shape.type === 'line') {
        strokePaths.push(`<path class="stroke-path" d="${linePath(shape)}"></path>`);
      } else if (shape.type === 'arc') {
        strokePaths.push(`<path class="stroke-path" d="${arcPath(shape)}"></path>`);
      } else if (shape.type === 'dot') {
        dotShapes.push(`<circle class="dot-shape" cx="${shape.p.x}" cy="${shape.p.y}" r="1"></circle>`);
      }
    }

    const pointMarkup = gridPoints.map((p) => {
      const isHover = state.hoverCandidate && state.hoverCandidate.type === 'dot' && samePoint(state.hoverCandidate.p, p);
      return `<circle class="snap-core${isHover ? ' is-hover' : ''}" cx="${p.x}" cy="${p.y}" r="${isHover ? 4.5 : 2.5}"></circle>`;
    }).join('');

    el.editorStage.innerHTML = `
      <svg id="editorSvg" viewBox="0 0 ${width} ${HEIGHT}" xmlns="http://www.w3.org/2000/svg" aria-label="Distribution glyph editor canvas">
        <rect x="0" y="0" width="100%" height="100%" fill="#fff"></rect>
        <g>${guideCircles.join('')}</g>
        <g>${strokePaths.join('')}</g>
        <g>${dotShapes.join('')}</g>
        <g>${previewMarkup}</g>
        <g>${pointMarkup}</g>
      </svg>
    `;

    const svg = document.getElementById('editorSvg');
    svg.addEventListener('pointermove', handleSvgPointerMove);
    svg.addEventListener('pointerleave', handleSvgPointerLeave);
    svg.addEventListener('click', handleSvgClick);
    svg.addEventListener('contextmenu', handleSvgContextMenu);

    updateInfoPanels();
  }

  function updateInfoPanels() {
    const ch = getDisplayChar(state.char);
    const hex = getCodepointHex(ch);
    const width = getWidth();
    const primitiveCount = state.shapes.length;
    const isDefined = getCurrentGlyphDefined();

    el.codepointField.value = `U+${hex}`;
    el.filenameField.value = getFilename(ch);
    el.glyphInfo.textContent = `${ch} · U+${hex}${state.manifest.loaded ? (isDefined ? ' · in manifest' : ' · not in manifest') : ''}`;
    el.sizeInfo.textContent = `${state.guideCols} guide column${state.guideCols === 1 ? '' : 's'} · ${width}×${HEIGHT}`;
    el.pointCountLabel.textContent = `${primitiveCount} primitive${primitiveCount === 1 ? '' : 's'}`;
    el.svgOutput.textContent = buildExportSvgString();

    if (state.hoverCandidate) {
      setFooterHint(`Preview: ${shapeToLabel(state.hoverCandidate)}. Click to place.`);
    } else {
      setFooterHint('Hover a valid geometry and click once to place it.');
    }

    renderPrimitiveList();
    renderManifestGlyphs();
  }

  function renderPrimitiveList() {
    if (!state.shapes.length) {
      el.primitiveList.innerHTML = `<li><span class="muted" style="grid-column:1 / -1;">No primitives yet.</span></li>`;
      return;
    }

    el.primitiveList.innerHTML = state.shapes.map((shape, index) => `
      <li>
        <span class="pill">${index + 1}</span>
        <span>${escapeXml(shapeToLabel(shape))}</span>
        <button class="btn" type="button" data-remove-index="${index}">Remove</button>
      </li>
    `).join('');

    el.primitiveList.querySelectorAll('[data-remove-index]').forEach(btn => {
      btn.addEventListener('click', () => {
        const index = Number(btn.getAttribute('data-remove-index'));
        state.shapes.splice(index, 1);
        setStatus('Primitive removed');
        renderEditor();
      });
    });
  }

  function renderManifestGlyphs() {
    if (!state.manifest.loaded) {
      if (!el.manifestGlyphGrid.childElementCount) {
        el.manifestGlyphGrid.innerHTML = '';
      }
      return;
    }

    if (!state.manifest.glyphs.length) {
      el.manifestGlyphGrid.innerHTML = '<div class="hint" style="grid-column:1 / -1;">Manifest loaded, but no character-uXXXX entries were found.</div>';
      return;
    }

    const current = getDisplayChar(state.char);
    el.manifestGlyphGrid.innerHTML = state.manifest.glyphs.map(item => `
      <button type="button" class="glyph-chip ${item.char === current ? 'is-active' : ''}" data-manifest-char="${escapeXml(item.char)}" title="${escapeXml(item.label)}">
        ${escapeXml(item.char)}
        <small>${escapeXml(item.hex)}</small>
      </button>
    `).join('');

    el.manifestGlyphGrid.querySelectorAll('[data-manifest-char]').forEach(btn => {
      btn.addEventListener('click', () => {
        const char = btn.getAttribute('data-manifest-char');
        state.char = char;
        el.charInput.value = char;
        if (state.autoWidth) {
          state.guideCols = suggestGuideCols(char);
          el.guideColsSelect.value = String(state.guideCols);
        }
        setStatus('Character selected from manifest');
        renderEditor();
      });
    });
  }

  function addShape(shape) {
    const key = serializeShapeKey(shape);
    const hasAlready = state.shapes.some(item => serializeShapeKey(item) === key);
    if (hasAlready) {
      setStatus('That primitive already exists');
      return;
    }
    state.shapes.push(shape);
    setStatus(`${shape.type} added`);
    renderEditor();
  }

  function removeNearestPlacedPrimitive(pointer) {
    let bestIndex = -1;
    let bestDistance = Infinity;

    state.shapes.forEach((shape, index) => {
      const distance = candidateDistance(pointer, shape);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });

    if (bestIndex >= 0 && bestDistance <= 10) {
      state.shapes.splice(bestIndex, 1);
      setStatus('Primitive removed');
      renderEditor();
      return true;
    }

    setStatus('No placed primitive close enough to remove');
    return false;
  }

  function svgPointerToCoords(event) {
    const svg = event.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  function handleSvgPointerMove(event) {
    const pointer = svgPointerToCoords(event);
    const candidate = getBestCandidate(pointer);
    const currentKey = state.hoverCandidate ? serializeShapeKey(state.hoverCandidate) : null;
    const nextKey = candidate ? serializeShapeKey(candidate) : null;
    if (currentKey !== nextKey) {
      state.hoverCandidate = candidate;
      renderEditor();
    }
  }

  function handleSvgPointerLeave() {
    if (state.hoverCandidate) {
      state.hoverCandidate = null;
      renderEditor();
    }
  }

  function handleSvgClick() {
    if (!state.hoverCandidate) {
      setStatus('Hover a valid primitive before clicking');
      return;
    }
    addShape(structuredClone(state.hoverCandidate));
  }

  function handleSvgContextMenu(event) {
    event.preventDefault();
    const pointer = svgPointerToCoords(event);
    removeNearestPlacedPrimitive(pointer);
  }

  function updateCharacterFromInput() {
    const ch = getDisplayChar(el.charInput.value);
    state.char = ch;
    el.charInput.value = ch;
    if (state.autoWidth) {
      state.guideCols = suggestGuideCols(ch);
      el.guideColsSelect.value = String(state.guideCols);
    }
    setStatus('Character updated');
    renderEditor();
  }

  function downloadSvg() {
    const svg = buildExportSvgString();
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = getFilename(getDisplayChar(state.char));
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus('SVG downloaded');
  }

  async function copySvg() {
    try {
      await navigator.clipboard.writeText(buildExportSvgString());
      setStatus('SVG copied');
    } catch (error) {
      setStatus('Clipboard copy failed');
    }
  }

  function extractGlyphsFromManifest(value, map = new Map()) {
    if (value == null) return map;

    if (typeof value === 'string') {
      const fileMatches = value.match(/character-u([0-9a-f]{4,6})\.svg/ig) || [];
      for (const match of fileMatches) {
        const hex = match.match(/u([0-9a-f]{4,6})/i)?.[1];
        if (hex) {
          const cp = parseInt(hex, 16);
          if (Number.isFinite(cp)) {
            const char = String.fromCodePoint(cp);
            map.set(char, {
              char,
              hex: `U+${hex.toUpperCase().padStart(4, '0')}`,
              label: `${char} U+${hex.toUpperCase().padStart(4, '0')}`,
            });
          }
        }
      }

      const soloHex = value.match(/^U\+?([0-9A-F]{4,6})$/i);
      if (soloHex) {
        const hex = soloHex[1].toUpperCase();
        const char = String.fromCodePoint(parseInt(hex, 16));
        map.set(char, {
          char,
          hex: `U+${hex.padStart(4, '0')}`,
          label: `${char} U+${hex.padStart(4, '0')}`,
        });
      }

      if (Array.from(value).length === 1 && !/^[\x00-\x1F]$/.test(value)) {
        const char = value;
        const hex = getCodepointHex(char);
        map.set(char, { char, hex: `U+${hex}`, label: `${char} U+${hex}` });
      }
      return map;
    }

    if (typeof value === 'number' && Number.isInteger(value) && value >= 32 && value <= 0x10ffff) {
      try {
        const char = String.fromCodePoint(value);
        const hex = value.toString(16).toUpperCase().padStart(4, '0');
        map.set(char, { char, hex: `U+${hex}`, label: `${char} U+${hex}` });
      } catch (error) {
        // ignore invalid codepoints
      }
      return map;
    }

    if (Array.isArray(value)) {
      value.forEach(item => extractGlyphsFromManifest(item, map));
      return map;
    }

    if (typeof value === 'object') {
      const maybeChar = typeof value.char === 'string' && Array.from(value.char).length === 1 ? value.char : null;
      const maybeCodepoint = typeof value.codepoint === 'number' ? value.codepoint : null;
      const maybeFilename = typeof value.filename === 'string'
        ? value.filename
        : (typeof value.file === 'string' ? value.file : null);

      if (maybeChar) {
        const hex = getCodepointHex(maybeChar);
        map.set(maybeChar, { char: maybeChar, hex: `U+${hex}`, label: `${maybeChar} U+${hex}` });
      }
      if (maybeCodepoint != null) {
        try {
          const char = String.fromCodePoint(maybeCodepoint);
          const hex = maybeCodepoint.toString(16).toUpperCase().padStart(4, '0');
          map.set(char, { char, hex: `U+${hex}`, label: `${char} U+${hex}` });
        } catch (error) {
          // ignore invalid codepoints
        }
      }
      if (maybeFilename) {
        extractGlyphsFromManifest(maybeFilename, map);
      }

      Object.values(value).forEach(item => extractGlyphsFromManifest(item, map));
      return map;
    }

    return map;
  }

  async function loadManifest() {
    for (const path of MANIFEST_PATHS) {
      try {
        const response = await fetch(path, { cache: 'no-store' });
        if (!response.ok) continue;
        const data = await response.json();
        const glyphMap = extractGlyphsFromManifest(data);
        state.manifest.loaded = true;
        state.manifest.path = path;
        state.manifest.glyphs = Array.from(glyphMap.values()).sort((a, b) => a.hex.localeCompare(b.hex));
        el.manifestStatus.textContent = `${state.manifest.glyphs.length} glyphs loaded`;
        el.manifestStatus.classList.add('ok');
        el.manifestHint.innerHTML = `Loaded manifest from <code>${escapeXml(path)}</code>. Click a defined character to jump to it.`;
        renderEditor();
        return;
      } catch (error) {
        // try next path
      }
    }

    state.manifest.loaded = true;
    state.manifest.path = null;
    state.manifest.glyphs = [];
    el.manifestStatus.textContent = 'Manifest not found';
    el.manifestHint.innerHTML = 'Could not load a manifest from the expected relative paths. The editor still works normally.';
    renderEditor();
  }

  function bindEvents() {
    el.charInput.addEventListener('input', updateCharacterFromInput);

    el.guideColsSelect.addEventListener('change', () => {
      state.guideCols = Number(el.guideColsSelect.value);
      state.hoverCandidate = null;
      setStatus('Glyph width updated');
      renderEditor();
    });

    el.autoWidthCheckbox.addEventListener('change', () => {
      state.autoWidth = el.autoWidthCheckbox.checked;
      if (state.autoWidth) {
        state.guideCols = suggestGuideCols(getDisplayChar(state.char));
        el.guideColsSelect.value = String(state.guideCols);
        renderEditor();
      }
    });

    el.themeSelect.addEventListener('change', () => setTheme(el.themeSelect.value));

    el.undoBtn.addEventListener('click', () => {
      if (!state.shapes.length) return;
      state.shapes.pop();
      setStatus('Last primitive removed');
      renderEditor();
    });

    el.clearBtn.addEventListener('click', () => {
      state.shapes = [];
      state.hoverCandidate = null;
      setStatus('Canvas cleared');
      renderEditor();
    });

    el.downloadBtn.addEventListener('click', downloadSvg);
    el.copySvgBtn.addEventListener('click', copySvg);

    document.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (state.shapes.length) {
          state.shapes.pop();
          setStatus('Last primitive removed');
          renderEditor();
        }
      }
    });
  }

  async function init() {
    loadTheme();
    state.char = getDisplayChar(el.charInput.value);
    state.guideCols = suggestGuideCols(state.char);
    el.guideColsSelect.value = String(state.guideCols);
    el.autoWidthCheckbox.checked = true;
    state.autoWidth = true;
    bindEvents();
    renderEditor();
    await loadManifest();
  }

  init();
})();
