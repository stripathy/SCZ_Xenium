// ── Shared core (loaded via core/*.js script tags before this file) ──
const {
  normalizeHex,
  decodeBoundaryJson, buildCellToBoundaryMap,
  drawDimLayer, drawBoundaryLayer, drawNucleusOnlyLayer, drawScatterLayer, drawTranscriptOverlay,
  hitTestMolecule, hitTestCell,
  drawScaleBar,
  renderTooltip,
  createApp, features,
} = window.SpatialViewerCore;

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

// ── App + adapter ──
// State that features own (showDeselectedCells, soloMode, soloType,
// cellTypeSearchFilter, geneSearchFilter) lives in app.state. Everything
// else stays as module-scope `let` for now — narrowest possible refactor.
const adapter = createSczAdapter({
  getSampleData: () => sampleData,
  getIndexData: () => indexData,
  getColorMode: () => colorMode,
  getActiveTypes: () => activeTypes,
  setActiveTypes: (s) => { activeTypes = s; },
  getCustomCellTypeColors: () => customCellTypeColors,
  getCustomGeneColors: () => customGeneColors,
  getTranscriptGenes: () => transcriptGenes,
  getShowQcDetails: () => showQcDetails,
  precomputeColors: () => precomputeColors(),
  buildCellTypeFilter: () => buildCellTypeFilter(),
  buildGeneList: (q) => buildGeneList(q),
  updateLegend: () => updateLegend(),
  render: () => render(),
});
const app = createApp({
  adapter,
  initialState: {
    showDeselectedCells: true,
    cellTypeSearchFilter: '',
    geneSearchFilter: '',
    soloMode: false,
    soloType: null,
    colorMode: 'subclass',
  },
});
app.use(features.showDeselected)
   .use(features.cellTypeSearch)
   .use(features.geneSearch)
   .use(features.solo)
   .use(features.colorPicker);
app.on('render', () => render());

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
  app.setState({
    cellTypeSearchFilter: '',
    soloMode: false, soloType: null,
  });
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

  cellToBoundaryIdx = buildCellToBoundaryMap(boundaryData, sampleData);
  if (cellToBoundaryIdx && boundaryData && sampleData.n_cells !== boundaryData.n) {
    let matched = 0;
    for (let i = 0; i < cellToBoundaryIdx.length; i++) if (cellToBoundaryIdx[i] !== -1) matched++;
    const unmatched = sampleData.n_cells - matched;
    console.log(`Boundary mapping: ${matched}/${boundaryData.n} boundaries matched to ${sampleData.n_cells} cells (${unmatched} cells without boundaries)`);
  }
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
  const q = (app.state.cellTypeSearchFilter || '').toLowerCase();
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
    const isSoloTarget = app.state.soloMode && app.state.soloType === name;
    row.className = 'ct-row'
      + (activeTypes.has(name) ? '' : ' dimmed')
      + (isSoloTarget ? ' solo-target' : '');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = activeTypes.has(name);
    cb.onchange = () => {
      if (app.state.soloMode) {
        // In solo mode, any click = make THAT row the only active.
        // The solo feature's enterSolo() updates state + calls adapter.
        app.enterSolo(name);
        return;
      }
      if (cb.checked) activeTypes.add(name); else activeTypes.delete(name);
      row.classList.toggle('dimmed',!cb.checked); render();
    };
    // Color-picker swatch — click opens native picker without toggling row.
    // The colorPicker feature handles input via delegation on #celltype-filter.
    const swatch = document.createElement('input');
    swatch.type = 'color';
    swatch.className = 'ct-swatch';
    swatch.value = normalizeHex(palette[name] || '#666666');
    swatch.title = `Click to change ${name}'s color`;
    swatch.setAttribute('data-color-celltype', name);
    swatch.setAttribute('data-color-mode', colorMode);
    swatch.onclick = (e) => e.stopPropagation();
    const label = document.createElement('span'); label.className='ct-label'; label.textContent=name;
    const countEl = document.createElement('span'); countEl.className='ct-count'; countEl.textContent=count.toLocaleString();
    row.appendChild(cb); row.appendChild(swatch); row.appendChild(label); row.appendChild(countEl);
    row.onclick = (e) => {
      if (e.target===cb || e.target===swatch) return;
      if (app.state.soloMode) { app.enterSolo(name); return; }
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
  if (app.state.soloMode) app.exitSolo();
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
  // Legacy entry point — delegate to the solo feature.
  app.enterSolo(name);
}

function toggleSoloMode() {
  if (!sampleData) return;
  if (app.state.soloMode) {
    app.exitSolo();
  } else {
    const seed = activeTypes.size === 1 ? [...activeTypes][0] : null;
    app.enterSolo(seed);
  }
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

  // Build per-cell `passes` mask from current colorMode + activeTypes.
  let filterCats, filterIndices;
  if (colorMode==='supertype') { filterCats=sampleData.supertype_cats; filterIndices=sampleData.supertype; }
  else if (colorMode==='layer') { filterCats=sampleData.layer_cats; filterIndices=sampleData.layer; }
  else if (colorMode==='class') { filterCats=sampleData.class_cats; filterIndices=sampleData.class; }
  else { filterCats=sampleData.subclass_cats; filterIndices=sampleData.subclass; }
  const activeSet = new Set();
  filterCats.forEach((c,i) => { if (activeTypes.has(c)) activeSet.add(i); });
  const passes = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (activeSet.has(filterIndices[i])) passes[i] = 1;
  }
  const qcMask = (hideQcFail && sampleData.qc_status) ? sampleData.qc_status : null;

  const zoomRatio = baseScale > 0 ? viewScale / baseScale : 1;
  const r = pointSize * Math.max(1, Math.sqrt(zoomRatio));
  const renderBoundaries = showBoundaries && boundaryData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD;
  const showNuc = showNucleus && nucleusData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD;
  const bMap = cellToBoundaryIdx;
  const baseOpts = { x, y, n, viewScale, viewX, viewY, w, h };

  // ── Layer overlay (study-specific: SCZ-only binned grid of layer colors) ──
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

  // Dim layer: deselected cells (passes[i]===0) so the user has tissue context.
  let dimShown = 0;
  if (app.state.showDeselectedCells) {
    dimShown = drawDimLayer(ctx, {
      ...baseOpts, passes, qcMask, useBoundaries: renderBoundaries,
      boundaryData, bMap, r,
    }).shown;
  }

  // Active cells: pick the fastest path that covers what we need to draw.
  let shown = 0;
  if (renderBoundaries) {
    shown = drawBoundaryLayer(ctx, {
      ...baseOpts, colors, passes, qcMask, boundaryData, bMap, r,
      alpha: pointOpacity,
      nucleusData: showNuc ? nucleusData : null,
      nucleusAlpha: pointOpacity * 0.8,
    }).shown;
  } else if (showNuc) {
    shown = drawNucleusOnlyLayer(ctx, {
      ...baseOpts, colors, passes, qcMask, nucleusData, bMap, r, alpha: pointOpacity,
    }).shown;
  } else {
    shown = drawScatterLayer(ctx, {
      ...baseOpts, colors, passes, qcMask, r, alpha: pointOpacity,
    }).shown;
  }

  // Transcript molecules on top.
  let molsShown = 0;
  if (activeGenes.size > 0) {
    molsShown = drawTranscriptOverlay(ctx, {
      transcriptGenes, activeGenes,
      viewScale, viewX, viewY, w, h,
      moleculeSize, moleculeOpacity, zoomRatio,
    }).shown;
  }

  drawScaleBar(ctx, { viewScale, logicalWidth, logicalHeight });

  document.getElementById('info-shown').textContent = shown.toLocaleString();
  let statusText = `Zoom: ${viewScale.toFixed(1)}x | ${shown.toLocaleString()} cells`;
  if (dimShown > 0) statusText += ` (+ ${dimShown.toLocaleString()} dimmed)`;
  if (renderBoundaries) statusText += ' (boundaries)';
  if (showNuc) statusText += ' (nuclei)';
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
        const swatch = normalizeHex(palette[name] || '#666666');
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
  // Tag legend swatches with the data attributes the colorPicker feature
  // reads; the feature handles input via delegation on #legend-overlay.
  el.querySelectorAll('input.leg-swatch[data-gene]').forEach(inp => {
    inp.setAttribute('data-color-gene', inp.dataset.gene);
  });
  el.querySelectorAll('input.leg-swatch[data-celltype]').forEach(inp => {
    inp.setAttribute('data-color-celltype', inp.dataset.celltype);
    inp.setAttribute('data-color-mode', colorMode);
  });
}

// ── Events ──
function setupEvents() {
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      colorMode = btn.dataset.mode;
      activeTypes = new Set();
      // Reset cell-type search + solo mode (type set differs per mode).
      // app.state.colorMode mirrors `colorMode` so the colorPicker feature
      // resolves the right palette via data-color-mode-less fallback.
      app.setState({
        cellTypeSearchFilter: '',
        soloMode: false, soloType: null,
        colorMode: btn.dataset.mode,
      });
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
  // The show-deselected toggle is wired by the show-deselected feature.
  document.getElementById('hide-qc-fail-toggle').onchange = (e) => { hideQcFail=e.target.checked; render(); };
  document.getElementById('show-qc-details-toggle').onchange = (e) => {
    showQcDetails = e.target.checked;
    document.querySelectorAll('.mode-btn.qc-only').forEach(b => b.style.display = showQcDetails ? '' : 'none');
    // If hiding QC modes while one is active, switch back to subclass
    if (!showQcDetails && (colorMode === 'confidence' || colorMode === 'margin')) {
      colorMode = 'subclass';
      app.setState({ colorMode: 'subclass' });
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === 'subclass'));
      document.getElementById('confidence-level-row').style.display = 'none';
      activeTypes = new Set();
      precomputeColors(); buildCellTypeFilter(); render();
    }
  };
  document.getElementById('mol-size').oninput = (e) => { moleculeSize=parseFloat(e.target.value); document.getElementById('mol-size-val').textContent=moleculeSize; render(); };
  document.getElementById('mol-opacity').oninput = (e) => { moleculeOpacity=parseFloat(e.target.value); document.getElementById('mol-opacity-val').textContent=moleculeOpacity; render(); };
  // Gene-search and celltype-search inputs are wired by the search features.

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
    // 'x' (show-deselected toggle) is owned by the show-deselected feature.
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
      app.setState({ colorMode });
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
    const x = sampleData.x, y = sampleData.y, n = sampleData.n_cells;

    // Molecule hover (rendered on top → checked first).
    const molHit = activeGenes.size > 0 ? hitTestMolecule({
      mx, my, transcriptGenes, activeGenes,
      viewScale, viewX, viewY, w: logicalWidth, h: logicalHeight,
      moleculeSize,
    }) : null;

    // Build hover masks. Same colorMode → filterIndices switch as render().
    let filterCats, filterIndices;
    if (colorMode==='supertype') { filterCats=sampleData.supertype_cats; filterIndices=sampleData.supertype; }
    else if (colorMode==='layer') { filterCats=sampleData.layer_cats; filterIndices=sampleData.layer; }
    else if (colorMode==='class') { filterCats=sampleData.class_cats; filterIndices=sampleData.class; }
    else { filterCats=sampleData.subclass_cats; filterIndices=sampleData.subclass; }
    const activeSet = new Set();
    filterCats.forEach((c,i) => { if (activeTypes.has(c)) activeSet.add(i); });

    // passesHover: respect hideQcFail; allow filtered-out cells when showDeselectedCells.
    // isDeselected: flag the ones we let through despite being filtered, so the
    // tooltip can render a badge.
    const passesHover = new Uint8Array(n);
    const isDeselected = new Uint8Array(n);
    const qcFail = sampleData.qc_status;
    for (let i = 0; i < n; i++) {
      if (hideQcFail && qcFail && qcFail[i] > 0) continue;
      if (activeSet.has(filterIndices[i])) {
        passesHover[i] = 1;
      } else if (app.state.showDeselectedCells) {
        passesHover[i] = 1;
        isDeselected[i] = 1;
      }
    }

    const zoomRatio = baseScale > 0 ? viewScale / baseScale : 1;
    const useBoundary = boundaryData && zoomRatio >= BOUNDARY_ZOOM_THRESHOLD;
    const cellHit = hitTestCell({
      mx, my, x, y, n, passes: passesHover, isDeselected,
      viewScale, viewX, viewY,
      useBoundary, boundaryData, bMap: cellToBoundaryIdx,
      pointSize,
    });

    const showMol = molHit !== null;
    const showCell = cellHit !== null;

    if (showMol || showCell) {
      // Build the field-list via the adapter, let features mutate it
      // through the tooltipReady event, then render to HTML via core.
      let html = '';
      if (showMol) {
        const molFields = adapter.getMoleculeTooltip(molHit);
        const molEv = { kind: 'molecule', hit: molHit, fields: molFields };
        app.emit('tooltipReady', molEv);
        // Color the molecule title with the gene's color (replicates pre-3c).
        const gd = transcriptGenes[molHit.gene];
        let molHtml = renderTooltip(molEv.fields);
        if (gd && gd.color) {
          molHtml = molHtml.replace(
            'class="tt-label"',
            `class="tt-label" style="color:${gd.color}"`,
          );
        }
        html += molHtml;
      }
      if (showCell) {
        if (showMol) {
          html += '<div style="border-top:1px solid #333; margin:4px 0;"></div>';
          html += `<div style="font-size:11px; color:#888;">Nearest cell:</div>`;
        }
        const cellFields = adapter.getCellTooltip(cellHit.idx, cellHit);
        const cellEv = {
          kind: 'cell', idx: cellHit.idx, hit: cellHit, fields: cellFields,
        };
        app.emit('tooltipReady', cellEv);
        html += renderTooltip(cellEv.fields);
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
app.start();

// ── Test/debug diagnostic exposure ──
// Top-level `let` in a classic script is script-scoped, not on window —
// invisible to Playwright's page.evaluate. Expose key state via getters
// so tests can introspect without changing runtime behavior. Useful for
// ad-hoc browser-console debugging too.
// NOTE: Top-level function declarations (render, fitView, etc.) are already
// on window via classic-script hoisting — do NOT re-expose them, or the
// inner bare reference in an arrow body would resolve back into the
// arrow and infinitely recurse.
Object.defineProperty(window, 'sampleData', { get: () => sampleData });
Object.defineProperty(window, 'indexData', { get: () => indexData });
Object.defineProperty(window, 'currentSample', { get: () => currentSample });
Object.defineProperty(window, 'colorMode', { get: () => colorMode });
Object.defineProperty(window, 'activeTypes', { get: () => activeTypes });
Object.defineProperty(window, 'soloMode', { get: () => app.state.soloMode });
Object.defineProperty(window, 'soloType', { get: () => app.state.soloType });
Object.defineProperty(window, 'cellTypeSearchFilter', { get: () => app.state.cellTypeSearchFilter || '' });
Object.defineProperty(window, 'showDeselectedCells', { get: () => app.state.showDeselectedCells });
Object.defineProperty(window, 'app', { get: () => app });
Object.defineProperty(window, 'adapter', { get: () => adapter });
Object.defineProperty(window, 'showBoundaries', { get: () => showBoundaries });
Object.defineProperty(window, 'showNucleus', { get: () => showNucleus });
Object.defineProperty(window, 'hideQcFail', { get: () => hideQcFail });
Object.defineProperty(window, 'showQcDetails', { get: () => showQcDetails });
Object.defineProperty(window, 'confidenceLevel', { get: () => confidenceLevel });
Object.defineProperty(window, 'showLayerOverlay', { get: () => showLayerOverlay });
Object.defineProperty(window, 'customGeneColors', { get: () => customGeneColors });
Object.defineProperty(window, 'customCellTypeColors', { get: () => customCellTypeColors });
Object.defineProperty(window, 'activeGenes', { get: () => activeGenes });
Object.defineProperty(window, 'transcriptGenes', { get: () => transcriptGenes });
Object.defineProperty(window, 'transcriptIndex', { get: () => transcriptIndex });
Object.defineProperty(window, 'viewScale', { get: () => viewScale });
Object.defineProperty(window, 'viewX', { get: () => viewX });
Object.defineProperty(window, 'viewY', { get: () => viewY });
Object.defineProperty(window, 'baseScale', { get: () => baseScale });
Object.defineProperty(window, 'logicalWidth', { get: () => logicalWidth });
Object.defineProperty(window, 'logicalHeight', { get: () => logicalHeight });
