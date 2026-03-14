# Cell Type Mapping Pipeline: Methods and Validation

## 1. Overview

We mapped cell types in 24 human DLPFC Xenium sections (12 SCZ, 12 control; 1.3M cells; 300-gene panel) to the SEA-AD MTG taxonomy (24 subclasses, 137 supertypes). The core challenge is cross-platform label transfer: Xenium's 300 genes capture only ~1% of the transcriptome measured by the snRNA-seq reference (137,303 cells x 36,601 genes), and platform-specific artifacts (optical detection limits, probe specificity) make standard integration approaches unreliable.

Our solution is a self-referencing correlation classifier that operates entirely within the Xenium feature space, with a simplified three-stage QC gate: spatial QC, 5th-percentile margin filter, and doublet suspect exclusion. The pipeline produces 1,225,037 QC-pass cells with cell type assignments validated against an independent MERFISH reference.

---

## 2. Pipeline Summary

```
Step 00: Create h5ad from raw Xenium data
Step 01: Spatial QC (negative probes, gene counts, total counts)
    → 1,298,687 cells pass basic QC (24 samples)

Step 02: MapMyCells HANN label transfer
    → class/subclass/supertype labels + confidence scores
    → Used as input for exemplar selection, not as final labels

Step 02b: Two-stage correlation classifier
    → Build centroids from top-100 HANN exemplars per type (from Xenium data)
    → Stage 1: Subclass assignment via Pearson correlation (24 types)
    → Stage 2: Supertype assignment within subclass
    → QC: Flag bottom 5% margin per sample + spatial doublets
    → Save centroids to disk for optional nuclear resolution (see code/nuclear_resolution/)

Step 03: Export transcript coordinates (for viewer)

Step 04: Cortical depth model (trained on MERFISH)
Step 05: Spatial domain annotation + layer assignment

Step 06: Viewer export (per-sample JSON + standalone HTML)
Step 07: Cell + nucleus boundary polygon export

Optional: Nuclear doublet resolution → code/nuclear_resolution/
    → See nuclear_resolution/README.md for details (not part of numbered sequence)

Analysis: Cortical cells → crumblr compositional regression (SCZ vs Control)
```

| QC Step | Cells | Lost | % Lost |
|---------|-------|------|--------|
| Raw (all cells) | 1,339,151 | — | — |
| Step 01: spatial QC (`qc_pass`) | 1,298,687 | 40,464 | 3.0% |
| **Step 02b: `corr_qc_pass` (default gate)** | **1,225,037** | **114,114** | **8.5%** |

*Cell counts reflect all 24 samples. `corr_qc_pass` combines spatial QC (step 01), 5th-percentile margin filter, and doublet suspect exclusion. An optional nuclear doublet resolution step (in `code/nuclear_resolution/`, not part of the numbered pipeline) can produce `hybrid_qc_pass` but was found to have negligible impact on downstream biology. Br2039 is excluded from downstream disease comparisons due to high white matter content (65%).*

---

## 3. Correlation Classifier

### Centroid construction from HANN exemplars

Allen Institute's MapMyCells HANN classifier provides initial cell type labels with bootstrap-based confidence scores (100 iterations, 0.5 subsampling). Rather than using these labels directly — HANN confidence thresholds aggressive enough to be useful removed ~25% of cells, disproportionately deep excitatory neurons and rare interneurons — we use the top 100 highest-confidence HANN exemplars per type (pooled across all Xenium samples) to build per-type centroid expression vectors. Expression is normalized (counts per 10k, log1p) and averaged. Because centroids are built from Xenium data itself, they inherently capture platform-specific expression characteristics, eliminating cross-platform normalization issues.

### Two-stage hierarchical Pearson correlation

**Stage 1 (Subclass):** Each cell's normalized expression is correlated against all 24 subclass centroids. The winning subclass is assigned along with the correlation value and *margin* (best minus second-best correlation). Both cell and centroid vectors are z-scored before correlation.

**Stage 2 (Supertype):** Within the assigned subclass, each cell is correlated against only the supertype centroids belonging to that subclass. This prevents biologically impossible assignments (e.g., an L2/3 IT cell being called an Sst supertype).

The 300-gene panel provides sufficient marker resolution to validate assignments at the single-molecule level:

![Exemplar cell transcript-level classification](output/presentation/exemplar_transcript_classification.png)
*Figure 1. Transcript-level cell type identity for six exemplar subclasses. Each panel shows a single cell with individual marker transcript molecules (e.g., CUX2 for L2/3 IT, PVALB+GAD1+GAD2 for Pvalb). Scale bar = 10 um.*

### QC: margin filtering and spatial doublet detection

**Margin filtering:** The bottom 5th percentile of subclass correlation margins per sample are flagged as low-confidence. Per-sample thresholds account for variation in data quality across sections. This threshold was calibrated against SEA-AD MERFISH ground-truth labels, where cells below the 5th percentile have ~80% subclass accuracy — see `docs/pipeline_qc_audit.md` for the full calibration analysis.

**Spatial doublet detection** identifies cells with biologically implausible marker co-expression:
- *Glut+GABA doublets:* Cells expressing 4+ of 7 GABAergic markers (GAD1, GAD2, SLC32A1, SST, PVALB, VIP, LAMP5) while also expressing glutamatergic markers. False-positive rate validated at 0.098% in snRNA-seq.
- *GABA+GABA doublets:* GABAergic cells co-expressing SST + PVALB + LAMP5 simultaneously (<0.01% in snRNA-seq).

Flagged cells show the expected doublet signatures: mixed marker expression from both neuronal classes and ~1.7-1.9x elevated UMI counts.

![Marker detection rates in doublets vs normal cells](output/presentation/doublet_marker_barplot.png)
*Figure 2. Marker detection rates comparing normal cells and doublets. Glut+GABA doublets express markers from both neuronal classes — a pattern expected from two co-captured cells.*

Transcript-level visualization directly confirms that doublet cells contain interleaved GABAergic and glutamatergic marker molecules within a single segmented cell body:

![Transcript molecules in doublet vs normal cells](output/presentation/doublet_transcript_examples.png)
*Figure 3. Individual transcript molecules within cell boundaries. Red = GABAergic markers; blue = glutamatergic markers. Normal cells show transcripts from one class only; doublets contain both.*

### Why not Harmony?

We benchmarked against Harmony + kNN label transfer (the standard cross-dataset integration approach). Harmony showed systematic failure modes with 300-gene Xenium data: Sst was inflated to 12.1% of cells (vs 2.5% expected), driven by non-neuronal cells (VLMC, Astrocyte) being misassigned into GABAergic space. Overall subclass agreement with both the correlation classifier and HANN was only 69%. The fundamental issue is that Harmony corrects batch effects between same-technology datasets, but cross-modality integration (snRNA-seq vs Xenium) introduces systematic differences — detection budgets, probe specificity, and PCA on 300 curated marker genes — that go beyond batch effects. Self-referencing centroids built from Xenium data avoid these problems entirely.

![Subclass proportions by method](output/presentation/subclass_proportions_by_method.png)
*Figure 4. Subclass proportions across classification methods. Note Sst inflation and non-neuronal distortion in Harmony results.*

---

## 4. Nuclear Doublet Resolution (optional — moved to `code/nuclear_resolution/`)

> **Note:** This step was empirically shown to have negligible impact on downstream compositional analysis (see `docs/pipeline_qc_audit.md`). The simplified QC pipeline (spatial QC + 5th-percentile margin filter + doublet exclusion) is now the default. This section is retained for scientific interest.

### Motivation

Inspection of doublet cells revealed that mixed-type marker transcripts concentrate in the cytoplasmic compartment, suggesting most doublets arise from mRNA spillover between neighboring cells during segmentation rather than true co-expression. Since nuclei are spatially more isolated, nuclear-restricted transcripts should be less affected by spillover.

### Method

For each sample, the pipeline builds a nuclear-only count matrix by intersecting per-molecule transcript coordinates with nucleus boundary polygons (Shapely STRtree spatial indexing). Each doublet-flagged cell is reclassified using nuclear-only counts against the pre-built correlation centroids:
- **Resolved:** Nuclear classification agrees with the non-doublet class identity (cytoplasmic spillover confirmed)
- **Persistent:** Nuclear classification still shows mixed-class identity (true doublet)
- **Nuclear-only:** Not flagged by whole-cell, but nuclear classification reveals a different class
- **Insufficient:** <50 nuclear UMIs

### Results across 24 samples

| Category | Count | % of Flagged |
|----------|-------|-------------|
| Whole-cell doublets (QC-pass) | 10,505 | 100% |
| Resolved (cytoplasmic spillover) | 7,941 | 75.6% |
| Persistent (true doublets) | 2,539 | 24.2% |
| Insufficient evidence | 25 | 0.2% |
| Nuclear-only (new detections) | 2,128 | — |

The overall resolution rate is **75.8%** (range: 68-83% across samples). Resolved doublets show high marker scores in the whole-cell compartment but low scores in the nuclear compartment, directly confirming cytoplasmic spillover as the mechanism.

![Nuclear doublet resolution summary](output/presentation/nuclear_doublet_resolution_summary.png)
*Figure 5. Nuclear doublet resolution outcomes across all 24 samples. The majority of whole-cell doublets (75.8%) are resolved by nuclear-only classification.*

![Nuclear doublet marker evidence](output/presentation/nuclear_doublet_marker_evidence.png)
*Figure 6. Marker expression evidence for doublet resolution. Resolved doublets show high whole-cell but low nuclear marker scores. Persistent doublets maintain high scores in both compartments.*

### Impact on downstream biology

The nuclear resolution machinery does not meaningfully change compositional SCZ vs Control results. Switching from `hybrid_qc_pass` to `corr_qc_pass` changes the snRNAseq meta-analysis correlation from r=0.405 to r=0.397, with identical top FDR-significant cell types. See `docs/pipeline_qc_audit.md` for the full comparison, including margin threshold calibration against MERFISH ground truth.

---

## 5. External Validation: MERFISH Benchmark

We compared Xenium cell type proportions and laminar depth distributions against the SEA-AD MERFISH dataset (341,595 cortical cells, 27 donors, manual layer annotations) as an independent external validation.

### Cell type proportions

| Level | Correlation Classifier r | Harmony r | n types |
|-------|-------------------------|-----------|---------|
| Subclass (Pearson, log-scale) | **0.80** | 0.73 | 24 |
| Supertype (Pearson, log-scale) | **0.73** | 0.60 | 110 |

The correlation classifier tracks the MERFISH reference more closely than Harmony at both resolution levels, particularly for non-neuronal types where Harmony shows the largest distortions (Sst inflated ~4x, VLMC nearly absent).

![Subclass proportion benchmark](output/presentation/cell_typing_benchmark_subclass_proportions.png)
*Figure 9. Subclass proportions: Xenium vs MERFISH reference. Left: Correlation Classifier (r = 0.80). Right: Harmony (r = 0.73). Dashed line = perfect agreement.*

### Laminar depth distributions

Median cortical depth per subclass correlates at r = 0.92 with the MERFISH reference, confirming spatially coherent laminar assignments. At supertype resolution, paired depth distributions show close agreement between MERFISH manual depth annotations and Xenium predicted depth across the full laminar gradient:

![Supertype depth distributions — Glutamatergic](output/presentation/supertype_depth_violins_glutamatergic.png)
*Figure 10. Supertype depth distributions for glutamatergic subclasses. Green = MERFISH manual depth, orange = Xenium predicted depth. Supertypes ordered by median MERFISH depth. The correlation classifier recovers the expected superficial-to-deep ordering within each subclass.*

---

## 6. Robustness

The top disease signals are robust across QC configurations. At the supertype level with the default pipeline (corr QC, 5th-percentile margin), 5 supertypes reach FDR < 0.05 (L5/6 NP_3, L2/3 IT_8, L6b_5, Pax6_3, Pvalb_13) and 8 reach FDR < 0.10. Effect sizes correlate with Nicole's snRNAseq meta-analysis at r = 0.432 (all supertypes) and r = 0.466 (neuronal only).

The same top hits persist regardless of:
- **QC gate**: Switching between `corr_qc_pass` and the legacy `hybrid_qc_pass` (which includes nuclear doublet resolution) changes median |delta logFC| by only 0.0065 across all 106 shared supertypes, with no cell type changing direction of effect (see `docs/pipeline_qc_audit.md`).
- **Margin threshold**: Varying the margin filter from 1st to 10th percentile sharpens the signal but does not change the top hits.
- **Classifier hierarchy**: Flat vs hierarchical correlation classifier yields the same disease signals.
- **Doublet handling**: Including or excluding resolved doublets has negligible impact.
