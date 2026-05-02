// ── Section collapse ──
function toggleSection(id) {
  document.getElementById(id).classList.toggle('collapsed');
}

// ── State ──
let currentSample = null;
let sampleData = null;
let indexData = null;
let colorMode = 'subclass';
let activeTypes = new Set();
let cellTypeSearchFilter = '';
let soloMode = false;
let soloType = null;
let pointSize = 2;
let pointOpacity = 0.8;
let showLayerOverlay = false;
let layerOverlayOpacity = 0.15;
let hideQcFail = true;
let baseScale = 1;

// View transform
let viewX = 0, viewY = 0, viewScale = 1;
let isDragging = false;
let dragStartX, dragStartY, dragViewX, dragViewY;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let logicalWidth = 0, logicalHeight = 0;

// ── Cell boundary state ──
let boundaryData = null;  // decoded boundary polygon data for current sample
let nucleusData = null;   // decoded nucleus boundary polygon data
let cellToBoundaryIdx = null; // mapping from cell index to boundary polygon index
let showBoundaries = true;
let showNucleus = true;
let showQcDetails = false; // show HANN conf, margin, doublet info in tooltips
let showDeselectedCells = true; // dim-render filtered-out cells + allow hover (default ON for spatial context)
const BOUNDARY_ZOOM_THRESHOLD = 3.0; // only show when zoomed in this much

// ── Transcript molecule state ──
let transcriptIndex = null;
let transcriptGenes = {};
let activeGenes = new Set();
let customGeneColors = {};   // gene -> custom hex color set by user
// Per-mode user color overrides for cell types: { mode: { typeName: '#hex' } }
// Persists across in-session mode switches; reset on page reload.
let customCellTypeColors = {};
let moleculeSize = 1;
let moleculeOpacity = 0.6;
const MAX_ACTIVE_GENES = 5;

const GENE_COLORS = [
  '#FF4444', '#44FF44', '#FFFF44', '#FF44FF', '#44FFFF',
];

// ── Depth colormap (viridis-like) ──
function depthColor(d) {
  d = Math.max(0, Math.min(1, d));
  const r = Math.round(68 + d * (253 - 68));
  const g = Math.round(1 + d * (231 - 1));
  const b = Math.round(84 + d * (37 - 84));
  return `rgb(${r},${g},${b})`;
}
const DEPTH_LUT = [];
for (let i = 0; i <= 256; i++) DEPTH_LUT.push(depthColor(i / 256));

// ── Confidence colormap (red → yellow → green) ──
let confidenceLevel = 'subclass'; // 'class', 'subclass', 'supertype'
function confColor(c) {
  // c in [0, 1]: 0=low confidence (red), 0.5=medium (yellow), 1=high (green)
  c = Math.max(0, Math.min(1, c));
  let r, g, b;
  if (c < 0.5) {
    const t = c * 2; // 0→1 within first half
    r = Math.round(220 - t * 40);  // 220→180
    g = Math.round(30 + t * 190);  // 30→220
    b = Math.round(30 + t * 10);   // 30→40
  } else {
    const t = (c - 0.5) * 2; // 0→1 within second half
    r = Math.round(180 - t * 140); // 180→40
    g = Math.round(220 + t * 20);  // 220→240
    b = Math.round(40 + t * 20);   // 40→60
  }
  return `rgb(${r},${g},${b})`;
}
const CONF_LUT = [];
for (let i = 0; i <= 200; i++) CONF_LUT.push(confColor(i / 200));

// ── Neuron/glia classification ──
const GLIA_TYPES = new Set(['Astrocyte', 'Oligodendrocyte', 'OPC', 'Microglia-PVM', 'Endothelial', 'VLMC']);
const NEURON_TYPES = new Set(['L2/3 IT','L4 IT','L5 IT','L5 ET','L5/6 NP','L6 IT','L6 IT Car3','L6 CT','L6b',
  'Sst','Sst Chodl','Pvalb','Vip','Lamp5','Lamp5 Lhx6','Sncg','Pax6','Chandelier']);

// ── Init ──
async function init() {
  const resp = await fetch('index.json');
  indexData = await resp.json();
  buildSampleList();
  setupEvents();
  resizeCanvas();
  // Default to Br8667 (has transcript data) or first sample
  const defaultSample = indexData.samples.find(s => s.sample_id === 'Br8667') || indexData.samples[0];
  loadSample(defaultSample.sample_id);
}

function buildSampleList() {
  const el = document.getElementById('sample-list');
  const controls = indexData.samples.filter(s => s.diagnosis === 'Control').sort((a,b) => a.sample_id.localeCompare(b.sample_id));
  const scz = indexData.samples.filter(s => s.diagnosis === 'SCZ').sort((a,b) => a.sample_id.localeCompare(b.sample_id));
  [...controls, ...scz].forEach(s => {
    const btn = document.createElement('button');
    btn.className = `sample-btn ${s.diagnosis.toLowerCase()}`;
    btn.textContent = s.sample_id;
    btn.dataset.sample = s.sample_id;
    btn.onclick = () => loadSample(s.sample_id);
    el.appendChild(btn);
  });
}

async function loadSample(sampleId) {
  const loading = document.getElementById('loading');
  loading.classList.add('show');
  document.querySelectorAll('.sample-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.sample === sampleId);
  });
  currentSample = sampleId;
  // Clear previous transcript state immediately to prevent stale rendering
  activeGenes.clear();
  transcriptGenes = {};
  transcriptIndex = null;
  // Reset cell-type search + solo mode (type set is sample-scoped)
  cellTypeSearchFilter = '';
  soloMode = false; soloType = null;
  const ctSearch = document.getElementById('celltype-search');
  if (ctSearch) ctSearch.value = '';
  document.getElementById('solo-btn')?.classList.remove('solo-active');
  const resp = await fetch(`${sampleId}.json`);
  sampleData = await resp.json();
  precomputeColors();
  buildLayerGrid();
  buildCellTypeFilter();
  fitView();
  document.getElementById('info-sample').textContent = sampleId;
  document.getElementById('info-dx').textContent = sampleData.diagnosis;
  document.getElementById('info-dx').style.color = sampleData.diagnosis === 'SCZ' ? '#e94560' : '#4ecdc4';
  document.getElementById('info-total').textContent = sampleData.n_cells.toLocaleString();

  // Load boundary and transcript data in parallel
  await Promise.all([
    loadBoundaryData(sampleId),
    checkTranscriptData(sampleId)
  ]);

  loading.classList.remove('show');
  render();
}

// ── Boundary data loading ──
function decodeBoundaryJson(raw) {
  const n = raw.n_cells;
  const vpc = raw.verts_per_cell;
  const xOff = raw.x_offset;
  const yOff = raw.y_offset;
  const xScale = raw.x_scale;
  const yScale = raw.y_scale;
  const totalVerts = n * vpc;
  const bx = new Float32Array(totalVerts);
  const by = new Float32Array(totalVerts);
  for (let i = 0; i < totalVerts; i++) {
    bx[i] = raw.bx[i] * xScale + xOff;
    by[i] = raw.by[i] * yScale + yOff;
  }
  return { n, vpc, bx, by };
}

// Build mapping from cell indices to boundary polygon indices.
// Boundary data may have fewer entries than cell data (e.g. QC-filtered cells
// were removed from boundaries but kept in the cell table).  The two arrays
// are in the same spatial order, so we walk them in lockstep, skipping cell
// entries that have no matching boundary polygon.
function buildCellToBoundaryMap(bd) {
  if (!bd || !sampleData) { cellToBoundaryIdx = null; return; }
  const nCells = sampleData.n_cells;
  const nBound = bd.n;
  const vpc = bd.vpc;
  const map = new Int32Array(nCells).fill(-1); // -1 = no boundary

  if (nCells === nBound) {
    // Perfect 1:1 — no mapping needed, but fill identity for uniformity
    for (let i = 0; i < nCells; i++) map[i] = i;
  } else {
    // Precompute boundary centroids
    const bcx = new Float32Array(nBound);
    const bcy = new Float32Array(nBound);
    for (let bi = 0; bi < nBound; bi++) {
      const base = bi * vpc;
      let sx = 0, sy = 0;
      for (let v = 0; v < vpc; v++) { sx += bd.bx[base + v]; sy += bd.by[base + v]; }
      bcx[bi] = sx / vpc;
      bcy[bi] = sy / vpc;
    }
    // Walk both arrays in lockstep
    let bi = 0;
    const tol = 5; // µm tolerance for centroid matching
    for (let ci = 0; ci < nCells && bi < nBound; ci++) {
      const dx = sampleData.x[ci] - bcx[bi];
      const dy = sampleData.y[ci] - bcy[bi];
      if (dx * dx + dy * dy < tol * tol) {
        map[ci] = bi;
        bi++;
      }
      // else: this cell has no boundary polygon, skip it
    }
    console.log(`Boundary mapping: ${bi}/${nBound} boundaries matched to ${nCells} cells (${nCells - bi} cells without boundaries)`);
  }
  cellToBoundaryIdx = map;
}

async function loadBoundaryData(sampleId) {
  // Load cell and nucleus boundaries in parallel
  const [cellResp, nucResp] = await Promise.allSettled([
    fetch(`boundaries/${sampleId}.json`).then(r => r.ok ? r.json() : null),
    fetch(`boundaries/${sampleId}_nucleus.json`).then(r => r.ok ? r.json() : null),
  ]);

  if (cellResp.status === 'fulfilled' && cellResp.value) {
    boundaryData = decodeBoundaryJson(cellResp.value);
    console.log(`Loaded cell boundaries for ${sampleId}: ${boundaryData.n} cells`);
  } else {
    boundaryData = null;
  }

  if (nucResp.status === 'fulfilled' && nucResp.value) {
    nucleusData = decodeBoundaryJson(nucResp.value);
    console.log(`Loaded nucleus boundaries for ${sampleId}: ${nucleusData.n} cells`);
  } else {
    nucleusData = null;
  }

  // Build cell→boundary index mapping (handles mismatched counts)
  buildCellToBoundaryMap(boundaryData);
}

// ── Transcript data loading ──
async function checkTranscriptData(sampleId) {
  const section = document.getElementById('transcript-section');
  try {
    const resp = await fetch(`transcripts/${sampleId}/gene_index.json`);
    if (!resp.ok) throw new Error('not found');
    const idx = await resp.json();
    transcriptIndex = idx;
    section.classList.add('visible');
    clearAllGenes();
    buildGeneList();
  } catch (e) {
    transcriptIndex = null;
    activeGenes.clear();
    transcriptGenes = {};
    section.classList.remove('visible');
  }
}

function buildGeneList(filter = '') {
  const el = document.getElementById('gene-list');
  el.innerHTML = '';
  if (!transcriptIndex) return;
  const filterLower = filter.toLowerCase();
  const genes = transcriptIndex.genes.filter(g => g.gene.toLowerCase().includes(filterLower));
  genes.sort((a, b) => {
    const aA = activeGenes.has(a.gene) ? 0 : 1;
    const bA = activeGenes.has(b.gene) ? 0 : 1;
    if (aA !== bA) return aA - bA;
    return a.gene.localeCompare(b.gene);
  });
  const shown = genes.slice(0, 50);
  shown.forEach(g => {
    const row = document.createElement('div');
    row.className = 'gene-row' + (activeGenes.has(g.gene) ? ' active' : '');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = activeGenes.has(g.gene);
    cb.onchange = () => toggleGene(g.gene, cb.checked, row);
    const isActive = activeGenes.has(g.gene);
    const liveColor = transcriptGenes[g.gene]?.color
      || customGeneColors[g.gene]
      || GENE_COLORS[getGeneColorIndex(g.gene)];
    let swatch;
    if (isActive) {
      // Color-picker swatch — click opens native color picker
      swatch = document.createElement('input');
      swatch.type = 'color';
      swatch.className = 'gene-swatch';
      swatch.value = liveColor;
      swatch.title = "Click to change this gene's overlay color";
      swatch.onclick = (e) => e.stopPropagation();   // don't toggle the row
      swatch.oninput = (e) => {
        const newColor = e.target.value;
        customGeneColors[g.gene] = newColor;
        if (transcriptGenes[g.gene]) transcriptGenes[g.gene].color = newColor;
        render();
        updateLegend();
      };
    } else {
      swatch = document.createElement('div');
      swatch.className = 'gene-swatch';
      swatch.style.background = '#444';
    }
    const label = document.createElement('span');
    label.className = 'gene-label';
    label.textContent = g.gene;
    const count = document.createElement('span');
    count.className = 'gene-count';
    count.textContent = g.n.toLocaleString();
    row.appendChild(cb);
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(count);
    row.onclick = (e) => {
      if (e.target === cb || e.target === swatch) return;
      cb.checked = !cb.checked; cb.onchange();
    };
    el.appendChild(row);
  });
  if (genes.length > 50) {
    const more = document.createElement('div');
    more.style.cssText = 'font-size:10px; color:#666; padding:4px 0;';
    more.textContent = `... ${genes.length - 50} more (type to filter)`;
    el.appendChild(more);
  }
  updateTranscriptInfo();
}

function getGeneColorIndex(gene) {
  const activeList = [...activeGenes];
  const idx = activeList.indexOf(gene);
  return idx >= 0 ? idx % GENE_COLORS.length : activeGenes.size % GENE_COLORS.length;
}

async function toggleGene(gene, checked, row) {
  if (checked) {
    if (activeGenes.size >= MAX_ACTIVE_GENES) {
      alert(`Maximum ${MAX_ACTIVE_GENES} genes can be displayed simultaneously.`);
      const cb = row.querySelector('input[type="checkbox"]');
      if (cb) cb.checked = false;
      return;
    }
    await loadGene(gene, row);
  } else {
    unloadGene(gene);
    buildGeneList(document.getElementById('gene-search').value);
    render();
  }
}

async function loadGene(gene, row) {
  if (transcriptGenes[gene]) {
    activeGenes.add(gene);
    buildGeneList(document.getElementById('gene-search').value);
    render();
    return;
  }
  if (row) row.classList.add('loading');
  try {
    const geneInfo = transcriptIndex.genes.find(g => g.gene === gene);
    if (!geneInfo) return;
    const resp = await fetch(`transcripts/${currentSample}/${geneInfo.file}`);
    if (!resp.ok) throw new Error(`Failed to load ${gene}`);
    const data = await resp.json();
    const n = data.n;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const xOff = transcriptIndex.x_offset, yOff = transcriptIndex.y_offset;
    const xSc = transcriptIndex.x_scale, ySc = transcriptIndex.y_scale;
    for (let i = 0; i < n; i++) {
      x[i] = data.x[i] * xSc + xOff;
      y[i] = data.y[i] * ySc + yOff;
    }
    const colorIdx = activeGenes.size % GENE_COLORS.length;
    const color = customGeneColors[gene] || GENE_COLORS[colorIdx];
    transcriptGenes[gene] = { x, y, n, color };
    activeGenes.add(gene);
    buildGeneList(document.getElementById('gene-search').value);
    render();
  } catch (e) { console.error(`Error loading gene ${gene}:`, e); }
  if (row) row.classList.remove('loading');
}

function unloadGene(gene) { activeGenes.delete(gene); delete transcriptGenes[gene]; }

function clearAllGenes() {
  activeGenes.clear();
  transcriptGenes = {};
  buildGeneList(document.getElementById('gene-search')?.value || '');
  render();
}

function updateTranscriptInfo() {
  const el = document.getElementById('transcript-info');
  const ctrl = document.getElementById('transcript-controls');
  if (!el) return;
  if (activeGenes.size === 0) {
    el.textContent = 'Select genes to overlay transcript molecules';
    if (ctrl) ctrl.classList.remove('visible');
  } else {
    const totalMols = [...activeGenes].reduce((s, g) => s + (transcriptGenes[g]?.n || 0), 0);
    el.textContent = `${activeGenes.size} gene(s) | ${totalMols.toLocaleString()} molecules`;
    if (ctrl) ctrl.classList.add('visible');
  }
}

function precomputeColors() {
  if (!sampleData) return;
  const n = sampleData.n_cells;
  sampleData._colors = new Array(n);
  if (colorMode === 'subclass') {
    const cats = sampleData.subclass_cats;
    const palette = indexData.subclass_colors;
    const cc = cats.map(c => palette[c] || '#666');
    for (let i = 0; i < n; i++) sampleData._colors[i] = cc[sampleData.subclass[i]];
  } else if (colorMode === 'supertype') {
    const cats = sampleData.supertype_cats;
    const palette = indexData.supertype_colors || {};
    const cc = cats.map(c => palette[c] || '#666');
    for (let i = 0; i < n; i++) sampleData._colors[i] = cc[sampleData.supertype[i]];
  } else if (colorMode === 'class') {
    const cats = sampleData.class_cats;
    const palette = indexData.class_colors;
    const cc = cats.map(c => palette[c] || '#666');
    for (let i = 0; i < n; i++) sampleData._colors[i] = cc[sampleData.class[i]];
  } else if (colorMode === 'layer') {
    const cats = sampleData.layer_cats;
    const palette = indexData.layer_colors;
    const cc = cats.map(c => palette[c] || '#444');
    for (let i = 0; i < n; i++) {
      const li = sampleData.layer[i];
      sampleData._colors[i] = li < cc.length ? cc[li] : '#444';
    }
  } else if (colorMode === 'depth') {
    for (let i = 0; i < n; i++) {
      const d = Math.max(0, Math.min(1, sampleData.depth[i]));
      sampleData._colors[i] = DEPTH_LUT[Math.round(d * 256)];
    }
  } else if (colorMode === 'confidence') {
    const confKey = 'conf_' + confidenceLevel;
    const confArr = sampleData[confKey];
    if (confArr) {
      for (let i = 0; i < n; i++) {
        const c = Math.max(0, Math.min(200, confArr[i]));
        sampleData._colors[i] = CONF_LUT[c];
      }
    } else {
      for (let i = 0; i < n; i++) sampleData._colors[i] = '#666';
    }
  } else if (colorMode === 'margin') {
    const marginArr = sampleData.corr_margin;
    if (marginArr) {
      for (let i = 0; i < n; i++) {
        // margin is quantized: 0-255 (value * 1000, clamped)
        // Map to CONF_LUT (0-200 range)
        const m = Math.max(0, Math.min(255, marginArr[i]));
        sampleData._colors[i] = CONF_LUT[Math.round(m * 200 / 255)];
      }
    } else {
      for (let i = 0; i < n; i++) sampleData._colors[i] = '#666';
    }
  }
}

// Apply a user-chosen color override for a cell type in a given color mode.
// Mutates the live palette so all downstream readers see it, records the
// override, then re-renders. Sidebar + legend swatches re-sync via rebuilds.
function applyCellTypeColor(mode, name, color) {
  if (!customCellTypeColors[mode]) customCellTypeColors[mode] = {};
  customCellTypeColors[mode][name] = color;
  // SCZ palette key convention: <mode>_colors (subclass_colors, etc.)
  const paletteKey = mode + '_colors';
  if (indexData[paletteKey]) {
    indexData[paletteKey][name] = color;
  }
  if (colorMode === mode) precomputeColors();
  buildCellTypeFilter();
  render();
  updateLegend();
}

// Normalize CSS color literal to 7-char #rrggbb (input type=color requirement)
function _normalizeHex(c) {
  if (!c || typeof c !== 'string') return '#666666';
  if (c[0] === '#') {
    if (c.length === 4) return ('#' + c[1]+c[1] + c[2]+c[2] + c[3]+c[3]).toLowerCase();
    if (c.length >= 7) return c.toLowerCase().slice(0, 7);
  }
  try {
    const ctxTmp = document.createElement('canvas').getContext('2d');
    ctxTmp.fillStyle = c;
    return ctxTmp.fillStyle.toLowerCase().slice(0, 7);
  } catch (e) { return '#666666'; }
}

function buildLayerGrid() {
  if (!sampleData) return;
  const BIN = 50;
  const x = sampleData.x, y = sampleData.y;
  const layerIndices = sampleData.layer;
  const nLayers = sampleData.layer_cats.length;
  const n = sampleData.n_cells;
  const xMin = sampleData.x_range[0] - BIN, yMin = sampleData.y_range[0] - BIN;
  const xMax = sampleData.x_range[1] + BIN, yMax = sampleData.y_range[1] + BIN;
  const nx = Math.ceil((xMax - xMin) / BIN), ny = Math.ceil((yMax - yMin) / BIN);
  const counts = new Uint16Array(ny * nx * nLayers);
  for (let i = 0; i < n; i++) {
    const xi = Math.min(nx-1, Math.max(0, Math.floor((x[i]-xMin)/BIN)));
    const yi = Math.min(ny-1, Math.max(0, Math.floor((y[i]-yMin)/BIN)));
    const li = layerIndices[i];
    if (li < nLayers) counts[(yi*nx+xi)*nLayers+li]++;
  }
  const grid = new Uint8Array(ny * nx);
  grid.fill(255);
  for (let gi = 0; gi < ny*nx; gi++) {
    let bc = 0, bl = 255;
    const base = gi * nLayers;
    for (let li = 0; li < nLayers; li++) { if (counts[base+li] > bc) { bc = counts[base+li]; bl = li; } }
    if (bc > 0) grid[gi] = bl;
  }
  const smoothed = new Uint8Array(grid);
  for (let pass = 0; pass < 2; pass++) {
    for (let yi = 1; yi < ny-1; yi++) for (let xi = 1; xi < nx-1; xi++) {
      const gi = yi*nx+xi;
      if (smoothed[gi] !== 255) continue;
      const nc = new Uint8Array(nLayers); let total = 0;
      for (let dy=-1; dy<=1; dy++) for (let dx=-1; dx<=1; dx++) {
        if (dx===0&&dy===0) continue;
        const v = smoothed[(yi+dy)*nx+(xi+dx)];
        if (v < nLayers) { nc[v]++; total++; }
      }
      if (total >= 4) { let best=0, bestL=255; for (let li=0;li<nLayers;li++) { if (nc[li]>best) {best=nc[li]; bestL=li;} } smoothed[gi]=bestL; }
    }
  }
  sampleData._layerGrid = { nx, ny, xMin, yMin, binSize: BIN, data: smoothed };
}

function buildCellTypeFilter() {
  const el = document.getElementById('celltype-filter');
  el.innerHTML = '';
  if (!sampleData) return;
  let cats, indices;
  if (colorMode === 'supertype') { cats = sampleData.supertype_cats; indices = sampleData.supertype; }
  else if (colorMode === 'layer') { cats = sampleData.layer_cats; indices = sampleData.layer; }
  else if (colorMode === 'class') { cats = sampleData.class_cats; indices = sampleData.class; }
  else { cats = sampleData.subclass_cats; indices = sampleData.subclass; } // subclass, depth, confidence all use subclass filter
  const counts = new Array(cats.length).fill(0);
  for (let i = 0; i < indices.length; i++) counts[indices[i]]++;
  if (activeTypes.size === 0 && !activeTypes._explicitEmpty) cats.forEach(c => activeTypes.add(c));
  const sorted = cats.map((c,i) => ({name:c, count:counts[i], idx:i})).sort((a,b) => a.name.localeCompare(b.name));

  // Apply search filter (selection state unchanged)
  const q = cellTypeSearchFilter.toLowerCase();
  const visible = q ? sorted.filter(({name}) => name.toLowerCase().includes(q)) : sorted;

  // Show solo-mode hint banner when waiting for user to pick a target
  if (soloMode && !soloType) {
    const hint = document.createElement('div');
    hint.style.cssText = 'font-size:11px;color:#e94560;padding:4px 6px;'
      + 'border:1px dashed #e94560;border-radius:4px;margin-bottom:6px;';
    hint.textContent = 'Solo mode on — click a type to show only that one.';
    el.appendChild(hint);
  }

  if (visible.length === 0) {
    el.innerHTML += `<div style="font-size:11px;color:#888;padding:6px 0;">`
      + `No types match "${cellTypeSearchFilter}".</div>`;
    return;
  }

  let palette;
  if (colorMode === 'layer') palette = indexData.layer_colors;
  else if (colorMode === 'class') palette = indexData.class_colors;
  else if (colorMode === 'supertype') palette = (indexData.supertype_colors||{});
  else palette = indexData.subclass_colors; // subclass, depth, confidence

  visible.forEach(({name, count}) => {
    const row = document.createElement('div');
    const isSoloTarget = soloMode && soloType === name;
    row.className = 'ct-row'
      + (activeTypes.has(name) ? '' : ' dimmed')
      + (isSoloTarget ? ' solo-target' : '');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = activeTypes.has(name);
    cb.onchange = () => {
      if (soloMode) {
        // In solo mode, any click = make THAT row the only active
        soloRow(name);
        return;
      }
      if (cb.checked) activeTypes.add(name); else activeTypes.delete(name);
      row.classList.toggle('dimmed',!cb.checked); render();
    };
    // Color-picker swatch — click opens native picker without toggling row
    const swatch = document.createElement('input');
    swatch.type = 'color';
    swatch.className = 'ct-swatch';
    swatch.value = _normalizeHex(palette[name] || '#666666');
    swatch.title = `Click to change ${name}'s color`;
    swatch.onclick = (e) => e.stopPropagation();
    swatch.oninput = (e) => applyCellTypeColor(colorMode, name, e.target.value);
    const label = document.createElement('span'); label.className='ct-label'; label.textContent=name;
    const countEl = document.createElement('span'); countEl.className='ct-count'; countEl.textContent=count.toLocaleString();
    row.appendChild(cb); row.appendChild(swatch); row.appendChild(label); row.appendChild(countEl);
    row.onclick = (e) => {
      if (e.target===cb || e.target===swatch) return;
      if (soloMode) { soloRow(name); return; }
      cb.checked=!cb.checked; cb.onchange();
    };
    el.appendChild(row);
  });
}

function getFilterCats() {
  if (colorMode==='supertype') return sampleData.supertype_cats;
  if (colorMode==='layer') return sampleData.layer_cats;
  if (colorMode==='class') return sampleData.class_cats;
  return sampleData.subclass_cats; // subclass, depth, confidence
}
function _exitSoloMode() {
  soloMode = false; soloType = null;
  document.getElementById('solo-btn')?.classList.remove('solo-active');
}
function selectAllTypes() { _exitSoloMode(); activeTypes = new Set(getFilterCats()); buildCellTypeFilter(); render(); }
function selectNoneTypes() { _exitSoloMode(); activeTypes = new Set(); activeTypes._explicitEmpty = true; buildCellTypeFilter(); render(); }
function selectNeurons() {
  _exitSoloMode();
  activeTypes = new Set(); activeTypes._explicitEmpty = true;
  getFilterCats().forEach(c => { const p=c.replace(/_\d+$/,''); if (NEURON_TYPES.has(p)||NEURON_TYPES.has(c)) activeTypes.add(c); });
  if (activeTypes.size>0) delete activeTypes._explicitEmpty; buildCellTypeFilter(); render();
}
function selectGlia() {
  _exitSoloMode();
  activeTypes = new Set(); activeTypes._explicitEmpty = true;
  getFilterCats().forEach(c => { const p=c.replace(/_\d+$/,''); if (GLIA_TYPES.has(p)||GLIA_TYPES.has(c)) activeTypes.add(c); });
  if (activeTypes.size>0) delete activeTypes._explicitEmpty; buildCellTypeFilter(); render();
}

function soloRow(name) {
  // Set the solo target (only this type active). soloMode stays on.
  activeTypes = new Set([name]);
  delete activeTypes._explicitEmpty;
  soloType = name;
  buildCellTypeFilter(); render();
}

function toggleSoloMode() {
  if (!sampleData) return;
  const btn = document.getElementById('solo-btn');
  if (soloMode) {
    // Exit solo mode: restore All
    soloMode = false; soloType = null;
    activeTypes = new Set(getFilterCats());
    delete activeTypes._explicitEmpty;
    btn?.classList.remove('solo-active');
  } else {
    // Enter solo mode. If exactly one type is currently active, lock that as
    // the solo target; otherwise wait for the user to click a row.
    soloMode = true;
    if (activeTypes.size === 1) soloType = [...activeTypes][0];
    else soloType = null;
    btn?.classList.add('solo-active');
  }
  buildCellTypeFilter(); render();
}

// ── View ──
function fitView() {
  if (!sampleData) return;
  const w = logicalWidth, h = logicalHeight - 30;
  const dx = sampleData.x_range[1] - sampleData.x_range[0];
  const dy = sampleData.y_range[1] - sampleData.y_range[0];
  viewScale = Math.min(w/dx, h/dy) * 0.9;
  baseScale = viewScale;
  viewX = (w - dx*viewScale)/2 - sampleData.x_range[0]*viewScale;
  viewY = (h - dy*viewScale)/2 - sampleData.y_range[0]*viewScale;
}

function resizeCanvas() {
  const main = document.getElementById('main');
  const dpr = window.devicePixelRatio || 1;
  logicalWidth = main.clientWidth;
  logicalHeight = main.clientHeight;
  canvas.width = logicalWidth * dpr;
  canvas.height = logicalHeight * dpr;
  canvas.style.width = logicalWidth + 'px';
  canvas.style.height = logicalHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (sampleData) { fitView(); render(); }
}

// ── Render ──
function render() {
  if (!sampleData) return;
  const w = logicalWidth, h = logicalHeight;
  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, w, h);

  const n = sampleData.n_cells;
  const x = sampleData.x, y = sampleData.y;
  const colors = sampleData._colors;

  let filterCats, filterIndices;
  if (colorMode==='supertype') { filterCats=sampleData.supertype_cats; filterIndices=sampleData.supertype; }
  else if (colorMode==='layer') { filterCats=sampleData.layer_cats; filterIndices=sampleData.layer; }
  else if (colorMode==='class') { filterCats=sampleData.class_cats; filterIndices=sampleData.class; }
  else { filterCats=sampleData.subclass_cats; filterIndices=sampleData.subclass; }
  const activeSet = new Set();
  filterCats.forEach((c,i) => { if (activeTypes.has(c)) activeSet.add(i); });

  const zoomRatio = baseScale > 0 ? viewScale / baseScale : 1;
  const r = pointSize * Math.max(1, Math.sqrt(zoomRatio));
  let shown = 0;

  // Layer overlay
  if (showLayerOverlay && colorMode !== 'layer' && sampleData._layerGrid) {
    const grid = sampleData._layerGrid;
    const layerPalette = indexData.layer_colors;
    const layerCats = sampleData.layer_cats;
    const binPx = grid.binSize * viewScale;
    const xi0 = Math.max(0, Math.floor((-viewX/viewScale-grid.xMin)/grid.binSize));
    const yi0 = Math.max(0, Math.floor((-viewY/viewScale-grid.yMin)/grid.binSize));
    const xi1 = Math.min(grid.nx-1, Math.floor(((w-viewX)/viewScale-grid.xMin)/grid.binSize));
    const yi1 = Math.min(grid.ny-1, Math.floor(((h-viewY)/viewScale-grid.yMin)/grid.binSize));
    ctx.globalAlpha = layerOverlayOpacity;
    const layerColors = layerCats.map(c => layerPalette[c] || null);
    for (let li=0; li<layerCats.length; li++) {
      const lc = layerColors[li]; if (!lc) continue; ctx.fillStyle = lc;
      for (let yi=yi0; yi<=yi1; yi++) for (let xi=xi0; xi<=xi1; xi++) {
        if (grid.data[yi*grid.nx+xi] !== li) continue;
        ctx.fillRect((grid.xMin+xi*grid.binSize)*viewScale+viewX, (grid.yMin+yi*grid.binSize)*viewScale+viewY, binPx+1, binPx+1);
      }
    }
    ctx.globalAlpha = 1;
  }

  // Determine if we should render boundaries or centroids
  const renderBoundaries = showBoundaries && boundaryData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD;
  const bMap = cellToBoundaryIdx; // cell index → boundary index (-1 if none)

  // Pre-pass: dim-render deselected cells so the user has tissue context
  // (and can hover over them — see handleHover). Mirrors active-layer behavior:
  // polygon outline at high zoom, scatter points at low zoom. ALSO respects
  // hideQcFail — QC-failed cells stay hidden in both layers when filter is on.
  let dimShown = 0;
  if (showDeselectedCells) {
    if (renderBoundaries) {
      // High zoom: per-polygon stroke (matches active-layer pattern; batched
      // Path2D was empirically 60× slower for many small subpaths)
      const bd = boundaryData; const vpc = bd.vpc;
      ctx.globalAlpha = 0.45;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.0;
      for (let i = 0; i < n; i++) {
        if (activeSet.has(filterIndices[i])) continue;
        if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) continue;
        const cx = x[i] * viewScale + viewX;
        const cy = y[i] * viewScale + viewY;
        if (cx < -50 || cx > w+50 || cy < -50 || cy > h+50) continue;
        const bi = bMap ? bMap[i] : i;
        if (bi < 0 || bi >= bd.n) continue;
        const base = bi * vpc;
        ctx.beginPath();
        ctx.moveTo(bd.bx[base]*viewScale+viewX, bd.by[base]*viewScale+viewY);
        for (let v = 1; v < vpc; v++) {
          ctx.lineTo(bd.bx[base+v]*viewScale+viewX, bd.by[base+v]*viewScale+viewY);
        }
        ctx.closePath();
        ctx.stroke();
        dimShown++;
      }
      ctx.globalAlpha = 1;
    } else {
      // Low/medium zoom: per-rect fillRect scatter. A single batched Path2D
      // with 64K+ subpaths is faster on warm path (~5ms vs ~18ms) but
      // cold-path browser tessellation can take 4+ seconds, freezing the UI
      // on the first None click after page load. Per-rect fillRect has no
      // such cold/warm cliff.
      const dr = pointSize * Math.max(1, Math.sqrt(zoomRatio));
      ctx.globalAlpha = 0.30;
      ctx.fillStyle = '#ffffff';
      for (let i = 0; i < n; i++) {
        if (activeSet.has(filterIndices[i])) continue;
        if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) continue;
        const sx = x[i]*viewScale+viewX, sy = y[i]*viewScale+viewY;
        if (sx < -dr || sx > w+dr || sy < -dr || sy > h+dr) continue;
        ctx.fillRect(sx-dr/2, sy-dr/2, dr, dr);
        dimShown++;
      }
      ctx.globalAlpha = 1;
    }
  }

  if (renderBoundaries) {
    // ── Render cells as filled polygons (with point fallback) ──
    const bd = boundaryData;
    const vpc = bd.vpc;
    ctx.globalAlpha = pointOpacity;

    // Batch by color, separating cells with/without boundary polygons
    const colorPolygons = {};  // cells that have boundary data
    const colorPoints = {};    // cells without boundary data (fallback to dots)

    for (let i = 0; i < n; i++) {
      if (!activeSet.has(filterIndices[i])) continue;
      if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) continue;

      // Quick check: is centroid on screen?
      const cx = x[i] * viewScale + viewX;
      const cy = y[i] * viewScale + viewY;
      if (cx < -50 || cx > w + 50 || cy < -50 || cy > h + 50) continue;

      const c = colors[i];
      const bi = bMap ? bMap[i] : i;
      if (bi >= 0 && bi < bd.n) {
        if (!colorPolygons[c]) colorPolygons[c] = [];
        colorPolygons[c].push(bi);
      } else {
        // No boundary polygon for this cell — render as point
        if (!colorPoints[c]) colorPoints[c] = [];
        colorPoints[c].push(cx, cy);
      }
      shown++;
    }

    // Draw filled boundary polygons
    for (const [color, boundaryIndices] of Object.entries(colorPolygons)) {
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.5;

      for (const bi of boundaryIndices) {
        const base = bi * vpc;
        ctx.beginPath();
        const sx0 = bd.bx[base] * viewScale + viewX;
        const sy0 = bd.by[base] * viewScale + viewY;
        ctx.moveTo(sx0, sy0);
        for (let v = 1; v < vpc; v++) {
          ctx.lineTo(bd.bx[base+v] * viewScale + viewX, bd.by[base+v] * viewScale + viewY);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    // Draw fallback points for cells without boundaries
    for (const [color, pts] of Object.entries(colorPoints)) {
      ctx.fillStyle = color;
      for (let j = 0; j < pts.length; j += 2) ctx.fillRect(pts[j]-r/2, pts[j+1]-r/2, r, r);
    }
    ctx.globalAlpha = 1;

    // ── Render nucleus outlines (stroke only, no fill) ──
    if (showNucleus && nucleusData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD) {
      const nd = nucleusData;
      const nvpc = nd.vpc;
      ctx.globalAlpha = pointOpacity * 0.8;
      ctx.lineWidth = 1.0;

      for (const [color, boundaryIndices] of Object.entries(colorPolygons)) {
        ctx.strokeStyle = color;
        for (const bi of boundaryIndices) {
          if (bi >= nd.n) continue;
          const base = bi * nvpc;
          ctx.beginPath();
          ctx.moveTo(nd.bx[base] * viewScale + viewX, nd.by[base] * viewScale + viewY);
          for (let v = 1; v < nvpc; v++) {
            ctx.lineTo(nd.bx[base+v] * viewScale + viewX, nd.by[base+v] * viewScale + viewY);
          }
          ctx.closePath();
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }

  } else if (showNucleus && nucleusData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD) {
    // ── Render nucleus outlines without cell boundaries ──
    const nd = nucleusData;
    const nvpc = nd.vpc;
    ctx.globalAlpha = pointOpacity * 0.8;
    ctx.lineWidth = 1.0;

    const colorPolygons = {};
    const colorPoints = {};
    for (let i = 0; i < n; i++) {
      if (!activeSet.has(filterIndices[i])) continue;
      if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) continue;
      const cx = x[i] * viewScale + viewX;
      const cy = y[i] * viewScale + viewY;
      if (cx < -50 || cx > w + 50 || cy < -50 || cy > h + 50) continue;
      const c = colors[i];
      const bi = bMap ? bMap[i] : i;
      if (bi >= 0 && bi < nd.n) {
        if (!colorPolygons[c]) colorPolygons[c] = [];
        colorPolygons[c].push({ci: i, bi: bi});
      } else {
        if (!colorPoints[c]) colorPoints[c] = [];
        colorPoints[c].push(cx, cy);
      }
      shown++;
    }

    // Draw point centroids first (for all visible cells)
    for (const [color, items] of Object.entries(colorPolygons)) {
      ctx.fillStyle = color;
      for (const {ci} of items) {
        const sx = x[ci] * viewScale + viewX;
        const sy = y[ci] * viewScale + viewY;
        ctx.fillRect(sx - r/2, sy - r/2, r, r);
      }
    }
    for (const [color, pts] of Object.entries(colorPoints)) {
      ctx.fillStyle = color;
      for (let j = 0; j < pts.length; j += 2) ctx.fillRect(pts[j]-r/2, pts[j+1]-r/2, r, r);
    }

    // Draw nucleus outlines on top
    for (const [color, items] of Object.entries(colorPolygons)) {
      ctx.strokeStyle = color;
      for (const {bi} of items) {
        const base = bi * nvpc;
        ctx.beginPath();
        ctx.moveTo(nd.bx[base] * viewScale + viewX, nd.by[base] * viewScale + viewY);
        for (let v = 1; v < nvpc; v++) {
          ctx.lineTo(nd.bx[base+v] * viewScale + viewX, nd.by[base+v] * viewScale + viewY);
        }
        ctx.closePath();
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

  } else {
    // ── Render cells as point centroids (original behavior) ──
    const colorBuckets = {};
    for (let i = 0; i < n; i++) {
      if (!activeSet.has(filterIndices[i])) continue;
      if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) continue;
      const sx = x[i]*viewScale+viewX, sy = y[i]*viewScale+viewY;
      if (sx < -r || sx > w+r || sy < -r || sy > h+r) continue;
      const c = colors[i];
      if (!colorBuckets[c]) colorBuckets[c] = [];
      colorBuckets[c].push(sx, sy);
      shown++;
    }
    ctx.globalAlpha = pointOpacity;
    for (const [color, pts] of Object.entries(colorBuckets)) {
      ctx.fillStyle = color;
      for (let i = 0; i < pts.length; i += 2) ctx.fillRect(pts[i]-r/2, pts[i+1]-r/2, r, r);
    }
    ctx.globalAlpha = 1;
  }

  // ── Render transcript molecules on top ──
  let molsShown = 0;
  if (activeGenes.size > 0) {
    const mr = moleculeSize * Math.max(0.5, Math.sqrt(zoomRatio) * 0.5);
    ctx.globalAlpha = moleculeOpacity;
    for (const gene of activeGenes) {
      const gd = transcriptGenes[gene]; if (!gd) continue;
      ctx.fillStyle = gd.color;
      const gx = gd.x, gy = gd.y, gn = gd.n;
      for (let i = 0; i < gn; i++) {
        const sx = gx[i]*viewScale+viewX, sy = gy[i]*viewScale+viewY;
        if (sx < -mr || sx > w+mr || sy < -mr || sy > h+mr) continue;
        ctx.fillRect(sx-mr/2, sy-mr/2, mr, mr);
        molsShown++;
      }
    }
    ctx.globalAlpha = 1;
  }

  // Draw persistent scale bar
  drawScaleBar();

  document.getElementById('info-shown').textContent = shown.toLocaleString();
  let statusText = `Zoom: ${viewScale.toFixed(1)}x | ${shown.toLocaleString()} cells`;
  if (dimShown > 0) statusText += ` (+ ${dimShown.toLocaleString()} dimmed)`;
  if (renderBoundaries) statusText += ' (boundaries)';
  if (showNucleus && nucleusData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD) statusText += ' (nuclei)';
  if (molsShown > 0) statusText += ` | ${molsShown.toLocaleString()} molecules`;
  document.getElementById('status-right').textContent = statusText;
  updateLegend();
}

function updateLegend() {
  const el = document.getElementById('legend-overlay');
  let html = '';

  // Cell-type legend — only when in a categorical mode and a small number
  // of types are active, otherwise the overlay would be unreadable.
  const CELL_TYPE_LEGEND_MAX = 10;
  const CONTINUOUS_MODES = new Set(['depth', 'confidence', 'margin']);
  if (!CONTINUOUS_MODES.has(colorMode)
      && sampleData
      && activeTypes.size > 0
      && activeTypes.size <= CELL_TYPE_LEGEND_MAX) {
    let cats, indices;
    if (colorMode === 'supertype') { cats = sampleData.supertype_cats; indices = sampleData.supertype; }
    else if (colorMode === 'layer') { cats = sampleData.layer_cats; indices = sampleData.layer; }
    else if (colorMode === 'class') { cats = sampleData.class_cats; indices = sampleData.class; }
    else { cats = sampleData.subclass_cats; indices = sampleData.subclass; }
    if (cats && indices) {
      let palette;
      if (colorMode === 'layer') palette = indexData.layer_colors;
      else if (colorMode === 'class') palette = indexData.class_colors;
      else if (colorMode === 'supertype') palette = (indexData.supertype_colors || {});
      else palette = indexData.subclass_colors;
      // Per-active-type cell counts on this sample
      const counts = {};
      for (let i = 0; i < indices.length; i++) {
        const name = cats[indices[i]];
        if (activeTypes.has(name)) counts[name] = (counts[name] || 0) + 1;
      }
      const sorted = [...activeTypes].sort((a, b) => a.localeCompare(b));
      html += '<div class="leg-title">Active cell types</div>';
      for (const name of sorted) {
        const swatch = _normalizeHex(palette[name] || '#666666');
        const count = counts[name] || 0;
        const safeName = name.replace(/"/g, '&quot;');
        html += `<div class="leg-row">`
          + `<input type="color" class="leg-swatch" value="${swatch}" `
          +   `data-celltype="${safeName}" title="Click to change ${safeName}'s color">`
          + `<span>${name} (${count.toLocaleString()})</span></div>`;
      }
      html += '<div style="height:6px;"></div>';
    }
  }

  if (activeGenes.size > 0) {
    html += '<div class="leg-title">Transcripts</div>';
    for (const gene of activeGenes) {
      const gd = transcriptGenes[gene]; if (!gd) continue;
      // Use a color input as the swatch so clicking opens the native picker.
      const safeGene = gene.replace(/"/g, '&quot;');
      html += `<div class="leg-row">`
        + `<input type="color" class="leg-swatch" value="${gd.color}" `
        +   `data-gene="${safeGene}" title="Click to change ${safeGene}'s color">`
        + `<span>${gene} (${gd.n.toLocaleString()})</span></div>`;
    }
    html += '<div style="height:6px;"></div>';
  }
  if (colorMode === 'depth') {
    html += `<div class="leg-title">Cortical Depth</div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:10px;">Pia (0)</span>
        <div style="width:100px;height:12px;background:linear-gradient(to right,rgb(68,1,84),rgb(59,82,139),rgb(33,145,140),rgb(94,201,98),rgb(253,231,37));border-radius:2px;"></div>
        <span style="font-size:10px;">WM (1)</span></div>`;
  } else if (colorMode === 'layer') {
    html += '<div class="leg-title">Cortical Layer</div>' + Object.entries(indexData.layer_colors).map(([k,v]) =>
      `<div class="leg-row"><div class="leg-swatch" style="background:${v}"></div><span>${k}</span></div>`).join('');
  } else if (colorMode === 'class') {
    html += '<div class="leg-title">Cell Class</div>' + Object.entries(indexData.class_colors).map(([k,v]) =>
      `<div class="leg-row"><div class="leg-swatch" style="background:${v}"></div><span>${k}</span></div>`).join('');
  } else if (colorMode === 'margin') {
    html += `<div class="leg-title">Correlation Margin</div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:10px;color:#dc3030;">Low (0)</span>
        <div style="width:100px;height:12px;background:linear-gradient(to right,rgb(220,30,30),rgb(180,220,40),rgb(40,240,60));border-radius:2px;"></div>
        <span style="font-size:10px;color:#28f03c;">High</span></div>`;
    if (sampleData.qc_status) {
      const qcLabels = {1: 'Spatial QC', 2: 'Low Margin', 3: 'Doublet Suspect'};
      let nFail = 0;
      const failCounts = {1: 0, 2: 0, 3: 0};
      for (let i = 0; i < sampleData.qc_status.length; i++) {
        const s = sampleData.qc_status[i];
        if (s > 0) { nFail++; if (failCounts[s] !== undefined) failCounts[s]++; }
      }
      const pctFail = (nFail / sampleData.n_cells * 100).toFixed(1);
      html += `<div style="margin-top:4px;font-size:10px;color:#aaa;">QC-fail: ${nFail.toLocaleString()} (${pctFail}%)</div>`;
      for (const [code, label] of Object.entries(qcLabels)) {
        if (failCounts[code] > 0) html += `<div style="font-size:9px;color:#777;padding-left:8px;">${label}: ${failCounts[code].toLocaleString()}</div>`;
      }
    }
  } else if (colorMode === 'confidence') {
    const levelName = confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1);
    html += `<div class="leg-title">HANN ${levelName} Confidence</div>
      <div style="display:flex;align-items:center;gap:6px;">
        <span style="font-size:10px;color:#dc3030;">Low (0)</span>
        <div style="width:100px;height:12px;background:linear-gradient(to right,rgb(220,30,30),rgb(180,220,40),rgb(40,240,60));border-radius:2px;"></div>
        <span style="font-size:10px;color:#28f03c;">High (1)</span></div>`;
    // Show distribution stats
    const confKey = 'conf_' + confidenceLevel;
    const confArr = sampleData[confKey];
    if (confArr) {
      const n = confArr.length;
      let sum = 0, below50 = 0, below28 = 0;
      for (let i = 0; i < n; i++) {
        sum += confArr[i];
        if (confArr[i] < 100) below50++;
        if (confArr[i] < 56) below28++;
      }
      const mean = (sum / n / 200).toFixed(2);
      html += `<div style="margin-top:4px;font-size:10px;color:#aaa;">Mean: ${mean} | <0.50: ${(below50/n*100).toFixed(1)}% | <0.28: ${(below28/n*100).toFixed(1)}%</div>`;
    }
  }
  el.innerHTML = html;
  // Wire up legend color pickers — keeps sidebar + legend swatches in sync
  el.querySelectorAll('input.leg-swatch[data-gene]').forEach(inp => {
    inp.oninput = (e) => {
      const gene = e.target.dataset.gene;
      const newColor = e.target.value;
      customGeneColors[gene] = newColor;
      if (transcriptGenes[gene]) transcriptGenes[gene].color = newColor;
      render();
      buildGeneList(document.getElementById('gene-search').value);
      updateLegend();
    };
  });
  // Cell-type legend swatches — reuse the same applyCellTypeColor pipeline
  el.querySelectorAll('input.leg-swatch[data-celltype]').forEach(inp => {
    inp.oninput = (e) => {
      applyCellTypeColor(colorMode, e.target.dataset.celltype, e.target.value);
    };
  });
}

// ── Scale bar ──
function drawScaleBar() {
  if (!sampleData) return;
  const padding = 20;
  const statusBarH = 30;
  const barY = logicalHeight - statusBarH - 20;  // above the status bar
  const barX = padding;

  // viewScale = pixels per µm
  // Pick a nice round µm value that gives a bar ~100-200px wide
  const targetPx = 150;
  const targetUm = targetPx / viewScale;
  // Find nice round number: 1, 2, 5 × 10^n
  const pow = Math.pow(10, Math.floor(Math.log10(targetUm)));
  const d = targetUm / pow;
  let niceUm;
  if (d < 1.5) niceUm = pow;
  else if (d < 3.5) niceUm = 2 * pow;
  else if (d < 7.5) niceUm = 5 * pow;
  else niceUm = 10 * pow;
  // Clamp to at least 1 µm
  niceUm = Math.max(1, Math.round(niceUm));

  const barPx = niceUm * viewScale;

  // Format label
  let label;
  if (niceUm >= 1000) label = `${niceUm / 1000} mm`;
  else label = `${niceUm} µm`;

  // Draw with semi-transparent background for contrast
  ctx.save();
  ctx.globalAlpha = 0.85;

  // Background pill behind the scale bar + label
  const bgPad = 6;
  ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
  const textW = ctx.measureText(label).width;
  const bgW = Math.max(barPx, textW) + bgPad * 2;
  const bgH = 32;
  const bgX = barX - bgPad;
  const bgY = barY - 22;
  ctx.fillStyle = 'rgba(13,13,26,0.7)';
  ctx.beginPath();
  ctx.roundRect(bgX, bgY, bgW, bgH, 4);
  ctx.fill();

  // Scale bar line (thin white)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(barX, barY);
  ctx.lineTo(barX + barPx, barY);
  ctx.stroke();

  // Small end ticks
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(barX, barY - 3);
  ctx.lineTo(barX, barY + 3);
  ctx.moveTo(barX + barPx, barY - 3);
  ctx.lineTo(barX + barPx, barY + 3);
  ctx.stroke();

  // Label text above the bar
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(label, barX + barPx / 2, barY - 5);

  ctx.restore();
}

// ── Events ──
function setupEvents() {
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      colorMode = btn.dataset.mode;
      activeTypes = new Set();
      // Reset cell-type search + solo mode (type set differs per mode)
      cellTypeSearchFilter = '';
      soloMode = false; soloType = null;
      const ctSearch = document.getElementById('celltype-search');
      if (ctSearch) ctSearch.value = '';
      document.getElementById('solo-btn')?.classList.remove('solo-active');
      // Show/hide confidence level selector
      document.getElementById('confidence-level-row').style.display = colorMode === 'confidence' ? 'block' : 'none';
      precomputeColors(); buildCellTypeFilter(); render();
    };
  });
  // Confidence level buttons
  document.querySelectorAll('.conf-level-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.conf-level-btn').forEach(b => {
        b.style.background = '#1a1a2e'; b.style.color = '#aaa';
      });
      btn.style.background = '#533483'; btn.style.color = 'white';
      confidenceLevel = btn.dataset.level;
      if (colorMode === 'confidence') { precomputeColors(); render(); }
    };
  });
  document.getElementById('point-size').oninput = (e) => { pointSize=parseFloat(e.target.value); document.getElementById('size-val').textContent=pointSize; render(); };
  document.getElementById('point-opacity').oninput = (e) => { pointOpacity=parseFloat(e.target.value); document.getElementById('opacity-val').textContent=pointOpacity; render(); };
  document.getElementById('layer-overlay-toggle').onchange = (e) => { showLayerOverlay=e.target.checked; document.getElementById('layer-overlay-opacity-row').style.display=showLayerOverlay?'block':'none'; render(); };
  document.getElementById('layer-overlay-opacity').oninput = (e) => { layerOverlayOpacity=parseFloat(e.target.value); document.getElementById('overlay-opacity-val').textContent=layerOverlayOpacity; render(); };
  document.getElementById('show-boundaries-toggle').onchange = (e) => { showBoundaries=e.target.checked; render(); };
  document.getElementById('show-nucleus-toggle').onchange = (e) => { showNucleus=e.target.checked; render(); };
  document.getElementById('show-deselected-toggle').onchange = (e) => { showDeselectedCells=e.target.checked; render(); };
  document.getElementById('hide-qc-fail-toggle').onchange = (e) => { hideQcFail=e.target.checked; render(); };
  document.getElementById('show-qc-details-toggle').onchange = (e) => {
    showQcDetails = e.target.checked;
    document.querySelectorAll('.mode-btn.qc-only').forEach(b => b.style.display = showQcDetails ? '' : 'none');
    // If hiding QC modes while one is active, switch back to subclass
    if (!showQcDetails && (colorMode === 'confidence' || colorMode === 'margin')) {
      colorMode = 'subclass';
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === 'subclass'));
      document.getElementById('confidence-level-row').style.display = 'none';
      activeTypes = new Set();
      precomputeColors(); buildCellTypeFilter(); render();
    }
  };
  document.getElementById('mol-size').oninput = (e) => { moleculeSize=parseFloat(e.target.value); document.getElementById('mol-size-val').textContent=moleculeSize; render(); };
  document.getElementById('mol-opacity').oninput = (e) => { moleculeOpacity=parseFloat(e.target.value); document.getElementById('mol-opacity-val').textContent=moleculeOpacity; render(); };
  document.getElementById('gene-search').oninput = (e) => { buildGeneList(e.target.value); };
  document.getElementById('celltype-search').oninput = (e) => {
    cellTypeSearchFilter = e.target.value;
    buildCellTypeFilter();
  };

  canvas.addEventListener('mousedown', (e) => { isDragging=true; dragStartX=e.clientX; dragStartY=e.clientY; dragViewX=viewX; dragViewY=viewY; canvas.style.cursor='grabbing'; });
  window.addEventListener('mousemove', (e) => { if (isDragging) { viewX=dragViewX+(e.clientX-dragStartX); viewY=dragViewY+(e.clientY-dragStartY); render(); } else { handleHover(e); } });
  window.addEventListener('mouseup', () => { isDragging=false; canvas.style.cursor='crosshair'; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect=canvas.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
    viewX = mx-(mx-viewX)*factor; viewY = my-(my-viewY)*factor; viewScale *= factor;
    render();
  }, { passive: false });
  canvas.addEventListener('dblclick', () => { fitView(); render(); });
  window.addEventListener('resize', resizeCanvas);

  window.addEventListener('keydown', (e) => {
    if (e.target.id === 'gene-search' || e.target.id === 'celltype-search') return;
    if (e.key==='r') { fitView(); render(); }
    if (e.key==='a') selectAllTypes();
    if (e.key==='n') selectNeurons();
    if (e.key==='g') selectGlia();
    if (e.key==='l') { showLayerOverlay=!showLayerOverlay; document.getElementById('layer-overlay-toggle').checked=showLayerOverlay; document.getElementById('layer-overlay-opacity-row').style.display=showLayerOverlay?'block':'none'; render(); }
    if (e.key==='b') { showBoundaries=!showBoundaries; document.getElementById('show-boundaries-toggle').checked=showBoundaries; render(); }
    if (e.key==='n') { showNucleus=!showNucleus; document.getElementById('show-nucleus-toggle').checked=showNucleus; render(); }
    if (e.key==='x') { showDeselectedCells=!showDeselectedCells; document.getElementById('show-deselected-toggle').checked=showDeselectedCells; render(); }
    if (e.key==='f') { hideQcFail=!hideQcFail; document.getElementById('hide-qc-fail-toggle').checked=hideQcFail; render(); }
    if (e.key==='d') { const el=document.getElementById('show-qc-details-toggle'); el.checked=!el.checked; el.onchange({target:el}); }
    if (e.key==='t') {
      const section = document.getElementById('transcript-section');
      if (section.classList.contains('visible')) { clearAllGenes(); section.classList.remove('visible'); }
      else if (transcriptIndex) { section.classList.add('visible'); buildGeneList(); }
    }
    if (e.key==='c') {
      // Toggle confidence mode
      if (colorMode !== 'confidence') {
        colorMode = 'confidence';
      } else {
        colorMode = 'subclass';
      }
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === colorMode));
      document.getElementById('confidence-level-row').style.display = colorMode === 'confidence' ? 'block' : 'none';
      activeTypes = new Set();
      precomputeColors(); buildCellTypeFilter(); render();
    }
    if (e.key==='q' && colorMode === 'confidence') {
      // Cycle confidence level: subclass -> supertype -> class -> subclass
      const levels = ['subclass', 'supertype', 'class'];
      const ci = levels.indexOf(confidenceLevel);
      confidenceLevel = levels[(ci + 1) % levels.length];
      document.querySelectorAll('.conf-level-btn').forEach(b => {
        const isActive = b.dataset.level === confidenceLevel;
        b.style.background = isActive ? '#533483' : '#1a1a2e';
        b.style.color = isActive ? 'white' : '#aaa';
      });
      precomputeColors(); render();
    }
    if (e.key==='ArrowLeft'||e.key==='ArrowRight') {
      const samples=indexData.samples.map(s=>s.sample_id);
      const idx=samples.indexOf(currentSample);
      const ni = e.key==='ArrowRight' ? Math.min(idx+1,samples.length-1) : Math.max(idx-1,0);
      if (ni!==idx) loadSample(samples[ni]);
    }
  });
}

// ── Hover tooltip ──
let hoverTimeout;
function handleHover(e) {
  if (!sampleData || isDragging) return;
  clearTimeout(hoverTimeout);
  hoverTimeout = setTimeout(() => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX-rect.left, my = e.clientY-rect.top;
    const tooltip = document.getElementById('tooltip');
    const sidebarWidth = document.getElementById('sidebar').offsetWidth;

    // ── Check transcript molecules first (they render on top) ──
    // Work in data coordinates to avoid per-molecule screen transform
    let bestMolDist = Infinity, bestMolGene = null, bestMolIdx = -1;
    if (activeGenes.size > 0) {
      const molThreshPx = Math.max(15, moleculeSize * 5);
      const molThreshData = molThreshPx / viewScale; // threshold in data coords
      const molThreshData2 = molThreshData * molThreshData;
      // Mouse position in data coordinates
      const mxData = (mx - viewX) / viewScale;
      const myData = (my - viewY) / viewScale;
      // Visible data range for frustum culling
      const dataXMin = -viewX / viewScale - 20/viewScale;
      const dataXMax = (logicalWidth - viewX) / viewScale + 20/viewScale;
      const dataYMin = -viewY / viewScale - 20/viewScale;
      const dataYMax = (logicalHeight - viewY) / viewScale + 20/viewScale;

      for (const gene of activeGenes) {
        const gd = transcriptGenes[gene]; if (!gd) continue;
        const gx = gd.x, gy = gd.y, gn = gd.n;
        for (let i = 0; i < gn; i++) {
          const px = gx[i], py = gy[i];
          // Skip off-screen molecules (frustum cull in data space)
          if (px < dataXMin || px > dataXMax || py < dataYMin || py > dataYMax) continue;
          const dx = mxData-px, dy = myData-py, d2 = dx*dx+dy*dy;
          if (d2 < molThreshData2 && d2 < bestMolDist) {
            bestMolDist = d2;
            bestMolGene = gene;
            bestMolIdx = i;
          }
        }
      }
    }

    // ── Check cells ──
    // Strategy: if boundaries are loaded and we're zoomed in enough, use point-in-polygon
    // hit testing. Otherwise fall back to centroid distance.
    let bestCellDist = Infinity, bestCellIdx = -1;
    const n=sampleData.n_cells, x=sampleData.x, y=sampleData.y;
    let filterCats, filterIndices;
    if (colorMode==='supertype') { filterCats=sampleData.supertype_cats; filterIndices=sampleData.supertype; }
    else if (colorMode==='layer') { filterCats=sampleData.layer_cats; filterIndices=sampleData.layer; }
    else if (colorMode==='class') { filterCats=sampleData.class_cats; filterIndices=sampleData.class; }
    else { filterCats=sampleData.subclass_cats; filterIndices=sampleData.subclass; }
    const activeSet = new Set();
    filterCats.forEach((c,i) => { if (activeTypes.has(c)) activeSet.add(i); });
    // Per-cell flag: was this cell hovered while filtered-out (deselected)?
    // Used to add a "deselected" badge in the tooltip when showDeselectedCells
    // lets users hover dim cells.
    const isDeselected = new Uint8Array(n);

    // Allow hover on filtered-out cells when toggle on. Still always respect
    // hideQcFail — QC-failed cells stay hidden in both layers when filter is on.
    const passesCellHover = (i) => {
      if (hideQcFail && sampleData.qc_status && sampleData.qc_status[i] > 0) return false;
      if (activeSet.has(filterIndices[i])) return true;
      return showDeselectedCells;
    };

    const zoomRatio = baseScale > 0 ? viewScale / baseScale : 1;
    const useBoundaryHit = boundaryData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD;

    if (useBoundaryHit) {
      // Point-in-polygon test using boundary data
      // Convert mouse to data coordinates
      const mxD = (mx - viewX) / viewScale;
      const myD = (my - viewY) / viewScale;
      const bd = boundaryData;
      const vpc = bd.vpc;
      // Pre-filter: only check cells whose centroid is reasonably close (within ~50µm)
      const searchRadius = 50;
      const sr2 = searchRadius * searchRadius;

      const bMapH = cellToBoundaryIdx;
      for (let i = 0; i < n; i++) {
        if (!passesCellHover(i)) continue;
        const bi = bMapH ? bMapH[i] : i;
        if (bi < 0 || bi >= bd.n) continue; // no boundary for this cell
        // Quick centroid distance check
        const cdx = mxD - x[i], cdy = myD - y[i];
        if (cdx*cdx + cdy*cdy > sr2) continue;
        // Ray-casting point-in-polygon test
        const base = bi * vpc;
        let inside = false;
        for (let v = 0, w = vpc - 1; v < vpc; w = v++) {
          const vx = bd.bx[base+v], vy = bd.by[base+v];
          const wx = bd.bx[base+w], wy = bd.by[base+w];
          if (((vy > myD) !== (wy > myD)) &&
              (mxD < (wx - vx) * (myD - vy) / (wy - vy) + vx)) {
            inside = !inside;
          }
        }
        if (inside) {
          // Found a cell — use centroid distance to pick the best if multiple overlap
          const d2 = cdx*cdx + cdy*cdy;
          if (d2 < bestCellDist) {
            bestCellDist = d2; bestCellIdx = i;
            if (!activeSet.has(filterIndices[i])) isDeselected[i] = 1;
          }
        }
      }
    } else {
      // Fallback: centroid distance
      const cellThreshold = Math.max(20, pointSize*3);
      bestCellDist = cellThreshold*cellThreshold;
      for (let i=0; i<n; i++) {
        if (!passesCellHover(i)) continue;
        const sx=x[i]*viewScale+viewX, sy=y[i]*viewScale+viewY;
        const dx=mx-sx, dy=my-sy, d2=dx*dx+dy*dy;
        if (d2 < bestCellDist) {
          bestCellDist=d2; bestCellIdx=i;
          if (!activeSet.has(filterIndices[i])) isDeselected[i] = 1;
        }
      }
    }

    // ── Determine what to show ──
    // Molecules take priority if one is found (they're rendered on top)
    const showMol = bestMolGene !== null;
    const showCell = bestCellIdx >= 0;

    if (showMol || showCell) {
      let html = '';

      // Molecule info (if found)
      if (showMol) {
        const gd = transcriptGenes[bestMolGene];
        const molX = gd.x[bestMolIdx].toFixed(1);
        const molY = gd.y[bestMolIdx].toFixed(1);
        html += `<div class="tt-label" style="color:${gd.color}">${bestMolGene}</div>`;
        html += `<div style="font-size:11px; color:#aaa;">Transcript molecule</div>`;
        html += `<div style="color:#666;font-size:10px;">x=${molX}, y=${molY}</div>`;
      }

      // Cell info (if found) — show below molecule info, or as primary
      if (showCell) {
        if (showMol) html += '<div style="border-top:1px solid #333; margin:4px 0;"></div>';
        const subclass=sampleData.subclass_cats[sampleData.subclass[bestCellIdx]];
        const supertype=sampleData.supertype_cats[sampleData.supertype[bestCellIdx]];
        const cls=sampleData.class_cats[sampleData.class[bestCellIdx]];
        const depth=sampleData.depth[bestCellIdx];
        const layerIdx=sampleData.layer[bestCellIdx];
        const layer=sampleData.layer_cats[layerIdx]||'Outside';
        const layerColor=indexData.layer_colors[layer]||'#888';
        if (showMol) {
          html += `<div style="font-size:11px; color:#888;">Nearest cell:</div>`;
        }
        const dimBadge = isDeselected[bestCellIdx]
          ? ' <span style="font-size:9px;color:#aaa;background:rgba(255,255,255,0.08);padding:0 4px;border-radius:3px;">deselected</span>'
          : '';
        html += `<div class="tt-label">${supertype}${dimBadge}</div>`;
        html += `<div>Subclass: ${subclass}</div>`;
        html += `<div>Class: ${cls}</div>`;
        html += `<div>Layer: <span style="color:${layerColor};font-weight:700;">${layer}</span></div>`;
        html += `<div>Depth: ${depth.toFixed(3)}</div>`;
        // QC details (gated behind Cell QC mode toggle)
        if (showQcDetails) {
          // Confidence scores (HANN mapping quality)
          const confC = sampleData.conf_class ? (sampleData.conf_class[bestCellIdx] / 200).toFixed(2) : '?';
          const confS = sampleData.conf_subclass ? (sampleData.conf_subclass[bestCellIdx] / 200).toFixed(2) : '?';
          const confT = sampleData.conf_supertype ? (sampleData.conf_supertype[bestCellIdx] / 200).toFixed(2) : '?';
          function confSpan(val) {
            const v = parseFloat(val);
            const color = v >= 0.5 ? '#28f03c' : v >= 0.28 ? '#e0d020' : '#dc3030';
            return `<span style="color:${color};font-weight:600;">${val}</span>`;
          }
          html += `<div style="margin-top:4px;border-top:1px solid #333;padding-top:4px;font-size:10px;color:#e94560;font-weight:600;">QC Details</div>`;
          html += `<div style="font-size:11px;color:#888;">HANN conf: ${confSpan(confC)} / ${confSpan(confS)} / ${confSpan(confT)}</div>`;
          html += `<div style="font-size:9px;color:#555;">class / subclass / supertype</div>`;
          // Correlation margin and QC status
          if (sampleData.corr_margin) {
            const margin = (sampleData.corr_margin[bestCellIdx] / 1000).toFixed(3);
            html += `<div style="font-size:11px;">Corr margin: ${margin}</div>`;
          }
          if (sampleData.qc_status) {
            const qcVal = sampleData.qc_status[bestCellIdx];
            const qcReasons = {0: null, 1: 'Spatial QC Fail', 2: 'Low Margin', 3: 'Doublet Suspect'};
            if (qcVal > 0) {
              html += `<div style="font-size:11px;"><span style="color:#e94560;font-weight:700;">QC-FAIL: ${qcReasons[qcVal] || 'Unknown'}</span></div>`;
            }
          }
          // HANN subclass comparison (if different)
          if (sampleData.hann_subclass_cats && sampleData.hann_subclass) {
            const hannSub = sampleData.hann_subclass_cats[sampleData.hann_subclass[bestCellIdx]];
            if (hannSub !== subclass) {
              html += `<div style="font-size:10px;color:#888;">HANN: ${hannSub}</div>`;
            }
          }
        }
        html += `<div style="color:#666;font-size:10px;">x=${x[bestCellIdx].toFixed(1)}, y=${y[bestCellIdx].toFixed(1)}</div>`;
      }

      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX-sidebarWidth+12)+'px';
      tooltip.style.top = (e.clientY-60)+'px';
    } else { tooltip.style.display = 'none'; }
  }, 50);
}

// ── Start ──
canvas.style.cursor = 'crosshair';
init();
