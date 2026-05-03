/**
 * SCZ viewer — study-specific data adapter.
 *
 * Mirrors output/viewer/rsc-data.js in the RSC repo with SCZ's data shape
 * swapped in (subclass/supertype/class/layer + qc_status + HANN
 * confidence triplet + corr_margin).
 *
 * Adapter methods consumed by core features:
 *   isCellDeselected(idx)              show-deselected
 *   onCellTypeSearchChange(value)      cellTypeSearch
 *   onGeneSearchChange(value)          geneSearch
 *   onSoloChange(soloMode, soloType)   solo
 *   getCurrentActiveTypes()            solo
 *   applyCellTypeColor(mode, name, c)  colorPicker
 *   applyGeneColor(gene, color)        colorPicker
 *
 * Plus tooltip builders called by viewer.js's handleHover:
 *   getCellTooltip(idx, ctx) → field-list
 *   getMoleculeTooltip(hit)  → field-list
 *
 * The adapter holds references to the (mutable) module-scope viewer state
 * via getters that the viewer wires in at construction.
 */
function createSczAdapter(refs) {
  const { normalizeHex } = window.SpatialViewerCore;

  function getFilterCatsAndIndices() {
    const sample = refs.getSampleData();
    const mode = refs.getColorMode();
    if (!sample) return { cats: null, indices: null };
    if (mode === 'supertype') return { cats: sample.supertype_cats, indices: sample.supertype };
    if (mode === 'layer')     return { cats: sample.layer_cats,     indices: sample.layer };
    if (mode === 'class')     return { cats: sample.class_cats,     indices: sample.class };
    return { cats: sample.subclass_cats, indices: sample.subclass };
  }

  // ── Feature contracts ──────────────────────────────────────────────────

  function isCellDeselected(idx) {
    const sample = refs.getSampleData();
    if (!sample) return false;
    const { cats, indices } = getFilterCatsAndIndices();
    if (!cats || !indices) return false;
    const name = cats[indices[idx]];
    return !refs.getActiveTypes().has(name);
  }

  function onCellTypeSearchChange(/* value */) {
    refs.buildCellTypeFilter();
  }

  function onGeneSearchChange(value) {
    refs.buildGeneList(value);
  }

  function onSoloChange(soloMode, soloType) {
    const { cats } = getFilterCatsAndIndices();
    if (!cats) return;
    if (soloMode && soloType) {
      const set = new Set([soloType]);
      delete set._explicitEmpty;
      refs.setActiveTypes(set);
    } else if (!soloMode) {
      const set = new Set(cats);
      delete set._explicitEmpty;
      refs.setActiveTypes(set);
    }
    // soloMode=true with soloType=null: no activeTypes change yet (user
    // clicks a row next).
    refs.precomputeColors();
    refs.buildCellTypeFilter();
    refs.render();
  }

  function getCurrentActiveTypes() {
    return refs.getActiveTypes();
  }

  function applyCellTypeColor(mode, name, color) {
    const indexData = refs.getIndexData();
    const customCellTypeColors = refs.getCustomCellTypeColors();
    if (!customCellTypeColors[mode]) customCellTypeColors[mode] = {};
    customCellTypeColors[mode][name] = color;
    // SCZ palette key convention: <mode>_colors (subclass_colors, etc.).
    const paletteKey = mode + '_colors';
    if (indexData[paletteKey]) {
      indexData[paletteKey][name] = color;
    }
    if (refs.getColorMode() === mode) refs.precomputeColors();
    refs.buildCellTypeFilter();
    refs.updateLegend();
    // Render is requested by the colorPicker feature itself.
  }

  function applyGeneColor(gene, color) {
    const customGeneColors = refs.getCustomGeneColors();
    const transcriptGenes = refs.getTranscriptGenes();
    customGeneColors[gene] = color;
    if (transcriptGenes[gene]) transcriptGenes[gene].color = color;
    // Rebuild sidebar gene-list so its swatch reflects the new color when
    // the change came from the legend (different element).
    refs.buildGeneList();
    refs.updateLegend();
  }

  // ── Tooltip builders ───────────────────────────────────────────────────

  function getMoleculeTooltip(hit) {
    const transcriptGenes = refs.getTranscriptGenes();
    const gd = transcriptGenes[hit.gene];
    if (!gd) return null;
    return {
      title: hit.gene,
      sections: [{
        rows: [{ value: 'Transcript molecule' }],
      }],
      position: { x: gd.x[hit.idx], y: gd.y[hit.idx] },
    };
  }

  function getCellTooltip(idx /* , ctx */) {
    const sample = refs.getSampleData();
    const indexData = refs.getIndexData();
    if (!sample) return null;

    const subclass = sample.subclass_cats[sample.subclass[idx]];
    const supertype = sample.supertype_cats[sample.supertype[idx]];
    const cls = sample.class_cats[sample.class[idx]];
    const depth = sample.depth ? sample.depth[idx] : null;
    const layerIdx = sample.layer ? sample.layer[idx] : null;
    const layer = (layerIdx != null && sample.layer_cats[layerIdx]) || 'Outside';
    const layerColor = (indexData.layer_colors && indexData.layer_colors[layer]) || '#888';

    // Identity rows (mirrors the original SCZ tooltip layout).
    const identityRows = [
      { raw: true, html: `<div>Subclass: ${esc(subclass)}</div>` },
      { raw: true, html: `<div>Class: ${esc(cls)}</div>` },
      { raw: true, html: `<div>Layer: <span style="color:${layerColor};font-weight:700;">${esc(layer)}</span></div>` },
    ];
    if (depth != null) {
      identityRows.push({ raw: true, html: `<div>Depth: ${depth.toFixed(3)}</div>` });
    }

    // Optional QC details (gated behind showQcDetails toggle).
    const qcRows = [];
    if (refs.getShowQcDetails()) {
      const confC = sample.conf_class     ? (sample.conf_class[idx]     / 200).toFixed(2) : '?';
      const confS = sample.conf_subclass  ? (sample.conf_subclass[idx]  / 200).toFixed(2) : '?';
      const confT = sample.conf_supertype ? (sample.conf_supertype[idx] / 200).toFixed(2) : '?';
      qcRows.push({
        raw: true,
        html: `<div style="font-size:11px;color:#888;">HANN conf: ${confSpan(confC)} / ${confSpan(confS)} / ${confSpan(confT)}</div>`,
      });
      qcRows.push({
        raw: true,
        html: `<div style="font-size:9px;color:#555;">class / subclass / supertype</div>`,
      });
      if (sample.corr_margin) {
        const margin = (sample.corr_margin[idx] / 1000).toFixed(3);
        qcRows.push({ raw: true, html: `<div style="font-size:11px;">Corr margin: ${margin}</div>` });
      }
      if (sample.qc_status) {
        const qcVal = sample.qc_status[idx];
        const qcReasons = { 0: null, 1: 'Spatial QC Fail', 2: 'Low Margin', 3: 'Doublet Suspect' };
        if (qcVal > 0) {
          qcRows.push({
            raw: true,
            html: `<div style="font-size:11px;"><span style="color:#e94560;font-weight:700;">QC-FAIL: ${qcReasons[qcVal] || 'Unknown'}</span></div>`,
          });
        }
      }
      if (sample.hann_subclass_cats && sample.hann_subclass) {
        const hannSub = sample.hann_subclass_cats[sample.hann_subclass[idx]];
        if (hannSub !== subclass) {
          qcRows.push({ raw: true, html: `<div style="font-size:10px;color:#888;">HANN: ${esc(hannSub)}</div>` });
        }
      }
    }

    const sections = [{ rows: identityRows }];
    if (qcRows.length > 0) {
      sections.push({ heading: 'QC Details', rows: qcRows });
    }

    return {
      title: supertype,
      // dim badge gets added by show-deselected feature's tooltipReady hook
      sections,
      position: { x: sample.x[idx], y: sample.y[idx] },
    };
  }

  // Color-coded confidence value: green (≥0.5) / yellow (≥0.28) / red.
  function confSpan(val) {
    const v = parseFloat(val);
    const color = v >= 0.5 ? '#28f03c' : v >= 0.28 ? '#e0d020' : '#dc3030';
    return `<span style="color:${color};font-weight:600;">${val}</span>`;
  }

  function esc(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  return {
    state: {},
    isCellDeselected,
    onCellTypeSearchChange,
    onGeneSearchChange,
    onSoloChange,
    getCurrentActiveTypes,
    applyCellTypeColor,
    applyGeneColor,
    getCellTooltip,
    getMoleculeTooltip,
  };
}

window.createSczAdapter = createSczAdapter;
