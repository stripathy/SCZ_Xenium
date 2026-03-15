# Xenium Panel Design for Supertype Classification in SCZ Spatial Transcriptomics

**Date**: 2026-03-14 (revised; original analysis 2026-03-06)
**Project**: SCZ_Xenium (Kwon et al. Schizophrenia Xenium spatial transcriptomics)

---

## 1. Scientific Objective

We aim to design a spatial transcriptomics experiment using 10x Xenium to resolve cell type composition differences between schizophrenia (SCZ) and control donors at the **supertype** level of the SEA-AD MTG taxonomy (128 supertypes across 24 subclasses) in human temporal cortex. The original focus was SST interneuron supertypes (16 types) and L6b glutamatergic supertypes (6 types), but analysis of the Kwon pilot dataset revealed that the classification challenge is universal across all subclasses.

The key constraint: **supertype-level classification requires within-subclass discriminating markers that are absent from most existing spatial panels**, including the panels used in the Kwon pilot study. This document presents the evidence for that claim, quantifies the problem across six spatial platforms, and recommends an optimized panel design.

---

## 2. The Central Finding: Gene Count Does Not Determine Supertype Resolution

### 2.1 Cross-platform marker coverage

We computed within-subclass Wilcoxon rank-sum markers from the SEA-AD snRNAseq reference (137,303 cells, 128 supertypes) — genes that distinguish sibling supertypes within each subclass (e.g., Sst_20 vs other Sst types). We then assessed how many of these top-10 markers each spatial panel contains:

| Panel | Genes | Design | Mean top-10 | ≥3 markers | HIGH confidence |
|---|---|---|---|---|---|
| **Kwon 300** | 300 | Custom (study-specific) | **0.7/10** | **4/126 (3%)** | **4 (3%)** |
| Xenium v1 Brain | 266 | Predesigned (brain) | 1.0/10 | 12/126 (10%) | 14 (11%) |
| SEA-AD MERFISH | 180 | Allen Institute (taxonomy) | 2.0/10 | 45/126 (36%) | 46 (37%) |
| **MERSCOPE 250** | **250** | **Vizgen (brain types)** | **2.9/10** | **71/126 (56%)** | **72 (57%)** |
| Xenium 5K Prime | 5,001 | Pan-tissue | 2.4/10 | 54/126 (43%) | 55 (44%) |
| MERSCOPE 4K | 3,999 | Near-transcriptome | 5.6/10 | 121/126 (96%) | 121 (96%) |

Three observations:

1. **MERSCOPE 250 (250 genes) outperforms Xenium 5K (5,001 genes)**: 57% vs 44% HIGH confidence. The 250-gene panel was specifically designed for brain cell type classification, while the 5K panel broadly covers cell lineages across organs.

2. **MERFISH 180, with fewer genes than either Xenium panel, achieves better coverage** (mean 2.0/10 vs 1.0) because the Allen Institute specifically selected genes for within-subclass discrimination.

3. **The Kwon 300-gene panel performs worst** despite having more genes than MERFISH 180. It was designed for subclass-level cell typing, not supertype discrimination. It shares only 52 genes with Xenium v1, 23 with MERFISH, and 15 with MERSCOPE 250.

**Subclass-level classification is robust across all panels.** The challenge is specifically at the supertype level within subclasses.

### 2.2 Confidence rating framework

Each supertype received a classification confidence rating based on two axes — marker coverage and cortical layer specificity:

- **HIGH**: ≥3 of top-10 within-subclass markers in panel, OR ≥2 markers + layer-distinct
- **MEDIUM**: 2 markers (no layer distinction), OR 1 marker + layer-distinct
- **LOW**: 0–1 markers and no layer distinction

Layer specificity was assessed using median normalized cortical depth from the SEA-AD MERFISH reference (368K depth-annotated cells). Only 7 of 128 supertypes qualify as "layer-distinct" (separation >0.10 from nearest within-subclass neighbor): Chandelier_1/2, Sst Chodl_1/2, Astro_3, Micro-PVM_2, Pax6_1. The large subclasses driving most results (Sst: 16 types, Pvalb: 13, Vip: 16, L2/3 IT: 10) have too many supertypes packed into narrow depth ranges for layer information to help.

---

## 3. Lessons from the Kwon Pilot Dataset

### 3.1 The discordance that triggered this investigation

Our density and compositional (crumblr) analyses of the Kwon Xenium data identified ~20 supertypes with nominally significant SCZ vs Control changes. Cross-referencing with Nicole's snRNAseq meta-analysis revealed striking discordance:

- **Sst_20**: increased in Xenium (density logFC +0.17) but *decreased* in snRNAseq (beta -0.12)
- **Sst_22**: null in Xenium but decreased in snRNAseq

Investigation revealed the Kwon panel contains **zero of the top-10 Sst supertype-discriminating markers**. SCZ itself shifts 54–112 panel genes per Sst supertype (FDR < 0.05), directly corrupting the profiles the correlation classifier relies on. Sst_3 cells in SCZ show significantly reduced classification margins (p = 2.3×10⁻¹⁶), with Sst_20 as a top confusion partner. Spatial neighborhood smoothing (9 parameter combinations) could not rescue classification — without discriminating markers, smoothing reinforces existing misclassifications.

### 3.2 What results can be trusted in the Kwon data

**High confidence:**
- All subclass-level composition and density results
- Supertype results concordant with snRNAseq meta-analysis
- Micro-PVM_2 (FDR 0.093, logFC -0.63; 2 markers + layer separation)

**Medium confidence (suggestive):**
- L5/6 NP_3, L6b_4, Vip_15, Sst_5: each has 2 within-subclass markers

**Low confidence (interpret with caution):**
- 15 of 20 significant supertypes, including the top hits:
  - L2/3 IT_8 (FDR 0.032) — zero within-subclass markers
  - Pvalb_13 (FDR 0.038) — 1 marker (TRHDE)
  - Sst_3 (FDR 0.045) — zero markers, confirmed confusion with Sst_19/20
  - Sst_25 (FDR 0.057) — 1 marker shared with Sst_20 (CDH12)

### 3.3 Recommendations for interpreting Kwon data

1. **Present subclass-level results as primary findings.** Robust to classification uncertainty.
2. **Present supertype results with explicit confidence ratings** as supplementary material.
3. **Use snRNAseq concordance as key validation.** Cross-platform agreement strengthens findings.
4. **Do not interpret Sst supertype discordance as biological.** Most likely a classification artifact.

---

## 4. Panel Design: v1 + Add-On vs 5K

### 4.1 Options considered

| | Option A: v1 + Add-On | Option B: 5K + Add-On |
|---|---|---|
| Base panel | Xenium v1 Brain (266 genes) | Xenium Prime 5K (5,001 genes) |
| Add-on | 100-gene custom | 100-gene custom |
| Total genes | 366 | 5,101 |

Options eliminated: standalone custom panels (101–480 genes) cannot combine with predesigned panels on the same slide and provide fewer sections for the same budget.

### 4.2 Cost analysis

All pricing based on 10x Genomics official pricing and UPitt Xenium Core rates (March 2026).

| | v1 + 100 Add-On | 5K + 100 Add-On |
|---|---|---|
| Add-on panel (one-time) | $8,000 (16 rxn) | $8,000 (16 rxn, est.) |
| Base panel per run | $345 (2 rxn) | Included in UPitt fee |
| UPitt per run (2 sections) | $13,000 | $23,500 |
| Total per run | $13,345 | $23,500 |
| Budget after add-on ($90K) | $82,000 | $82,000 |
| **Maximum runs** | **6** | **3** |
| **Maximum sections** | **12** | **6** |
| Cost per section | $7,339 | $13,083 |

### 4.3 Trade-off analysis

| Factor | Assessment | Favors |
|---|---|---|
| **Statistical power** | 12 vs 6 sections — the most important factor for between-group comparisons | **v1** |
| **Supertype resolution** | v1+100 achieves 97% HIGH confidence across ALL 126 supertypes; 5K alone achieves only 44% HIGH | **v1** |
| **Cost efficiency** | $7,339 vs $13,083 per section | **v1** |
| **Within-subclass markers** | v1+100 provides mean 4.3/10 top markers vs 5K's 2.4/10. The 5K's total gene count creates an illusion of marker coverage — most of its 5,001 genes are subclass-level or non-brain markers | **v1** |
| **Genome-wide DE** | 5K enables unbiased DE across all genes, but supertype-level DE is only interpretable when supertypes are correctly classified (44% HIGH on 5K alone) | 5K (limited) |
| **Flexibility** | 5K enables pivot to non-supertype questions if needed | 5K |

### 4.4 When 5K is still the right choice

- The study goal shifts toward unbiased discovery rather than hypothesis-driven supertype analysis
- Additional funding increases the budget to $140K+ (enough for 6+ sections on 5K)
- The team wants to contribute to a multi-site consortium using standardized 5K panels
- The add-on can be used with 5K too — 5K+100 would achieve both genome-wide coverage and supertype resolution

---

## 5. Add-On Gene Selection

### 5.1 Strategy: greedy selection across all supertypes, restricted to spatial-validated genes

The v1 base panel already handles subclass-level cell type identification — what it lacks is within-subclass supertype discrimination. All 100 add-on slots should therefore go to within-subclass markers, selected to maximize coverage across **all 126 supertypes** (not just SST/L6b). Cardinal markers, subclass-level markers, and DE candidate reserves are unnecessary in the add-on since the v1 base panel covers subclass identity and the add-on's sole purpose is supertype resolution.

To ensure every recommended gene will work in a FISH-based assay, the candidate pool is restricted to **genes that have been successfully measured on at least one spatial platform** (MERFISH 180 ∪ MERSCOPE 250 ∪ MERSCOPE 4K = 4,083 unique genes). Candidates are drawn from the top-50 within-subclass Wilcoxon markers per supertype (computed from snRNAseq), then selected greedily: at each step, the gene providing the greatest weighted coverage improvement is chosen, with under-covered supertypes receiving higher priority (weight = max(0, 5 − current_coverage)).

This spatial-validated greedy approach achieves **identical coverage** to an unrestricted selection at 100 genes (118/126 supertypes at ≥3 markers). The 32 genes excluded by the spatial restriction include several lncRNAs and genes never tested by FISH.

### 5.2 Coverage progression

| Add-on genes | Panel size | Mean top-10 | HIGH | LOW | Milestone |
|---|---|---|---|---|---|
| 0 | 266 | 1.0 | 14 (11%) | 93 (74%) | Xenium v1 baseline |
| **16** | **282** | **2.0** | **40 (32%)** | **59 (47%)** | **Matches MERFISH mean coverage** |
| **18** | **284** | **2.1** | **46 (37%)** | **55 (44%)** | **Matches MERFISH HIGH count** |
| 25 | 291 | 2.5 | 55 (44%) | 45 (36%) | Matches Xenium 5K HIGH count |
| 36 | 302 | 3.0 | 72 (57%) | 28 (22%) | Matches MERSCOPE 250 HIGH count |
| **100** | **366** | **4.6** | **122 (97%)** | **0 (0%)** | **Surpasses MERSCOPE 4K** |

The first 18 genes contribute as much HIGH confidence gain as the next 82 combined.

### 5.3 Cross-platform validation: do snRNAseq markers work spatially?

The within-subclass markers were identified from snRNAseq, so a key concern is whether they retain discriminating power when measured by FISH. We validated this three ways:

**Rank concordance.** We independently computed within-subclass markers from SEA-AD MERFISH spatial data (1.9M cells, 180 genes with supertype labels). For 874 gene-supertype pairs measurable on both platforms: log fold change correlation **r = 0.60** (p = 2.7×10⁻⁸⁶). Genes ranked top-10 in snRNAseq are **79% also top-10 in MERFISH** and 94% in top-20. The markers are genuine supertype discriminators across platforms.

**Convergent gene selection.** Computing markers directly from MERFISH spatial data and running greedy selection independently yields the same gene families: ROBO1/2, RGS6, KIRREL3, GPC5 — axon guidance and cell adhesion molecules. Four genes appear in both the snRNAseq-derived and MERFISH-derived top-20 lists.

**Platform provenance.** Every gene in the recommended list has been measured by at least one FISH platform. Six of the top 20 (RGS6, ROBO2, LRP1B, NLGN1, KCNIP4, ROBO1) were independently selected by the Allen Institute for their MERFISH 180 panel, confirming that domain experts reached the same conclusion about which genes matter for brain cell type classification.

### 5.4 Recommended add-on gene list (top 20)

| Rank | Gene | Spatial panels | Top-10 for N types | Gene family |
|---|---|---|---|---|
| 1 | RGS6 | MERFISH | 8 | RGS signaling |
| 2 | SNTG1 | MERSCOPE 4K | 10 | Syntrophin |
| 3 | LINGO2 | MERSCOPE 250 + 4K | 9 | Leucine-rich repeat |
| 4 | DPP6 | MERSCOPE 250 + 4K | 7 | Dipeptidyl peptidase |
| 5 | SLIT2 | MERSCOPE 250 + 4K | 8 | Axon guidance |
| 6 | EPHA6 | MERSCOPE 250 + 4K | 9 | Ephrin receptor |
| 7 | NCAM2 | MERSCOPE 250 + 4K | 8 | Neural cell adhesion |
| 8 | ROBO2 | MERFISH + MERSCOPE 4K | 8 | Axon guidance |
| 9 | LRP1B | MERFISH + MERSCOPE 250 + 4K | 8 | LDL receptor |
| 10 | PCDH7 | MERSCOPE 250 + 4K | 8 | Protocadherin |
| 11 | SGCZ | MERSCOPE 250 + 4K | 8 | Sarcoglycan |
| 12 | PCDH9 | MERSCOPE 250 + 4K | 6 | Protocadherin |
| 13 | NLGN1 | MERFISH + MERSCOPE 4K | 8 | Neuroligin |
| 14 | KCNIP4 | MERFISH + MERSCOPE 250 + 4K | 8 | K+ channel interactor |
| 15 | SGCD | MERSCOPE 250 + 4K | 7 | Sarcoglycan |
| 16 | ROBO1 | MERFISH + MERSCOPE 250 + 4K | 7 | Axon guidance |
| 17 | DPP10 | MERSCOPE 250 + 4K | 7 | Dipeptidyl peptidase |
| 18 | SOX2-OT | MERSCOPE 4K | 7 | lncRNA |
| 19 | GPC5 | MERFISH + MERSCOPE 250 + 4K | 6 | Glypican |
| 20 | GALNTL6 | MERFISH + MERSCOPE 250 + 4K | 7 | Glycosyltransferase |

These are overwhelmingly **neuronal cell adhesion molecules, axon guidance receptors, and ion channel subunits** — gene families that define neuronal subtype identity through synaptic connectivity and spatial positioning, but which are not typically prioritized in panels designed for broad cell type identification.

Full 100-gene list with spatial panel provenance: `v1_addon_100_spatial_validated.csv`.

### 5.5 Per-subclass improvement

| Subclass | # types | v1 mean | v1+100 mean | Δ |
|---|---|---|---|---|
| Astrocyte | 5 | 0.4 | 3.8 | +3.4 |
| Chandelier | 2 | 0.5 | 5.5 | +5.0 |
| L2/3 IT | 10 | 1.2 | 4.2 | +3.0 |
| L4 IT | 4 | 1.0 | 5.0 | +4.0 |
| L5 ET | 2 | 0.0 | 4.5 | +4.5 |
| L5 IT | 6 | 1.0 | 4.7 | +3.7 |
| L5/6 NP | 5 | 1.0 | 4.6 | +3.6 |
| L6 CT | 4 | 0.5 | 4.2 | +3.8 |
| L6b | 6 | 1.5 | 5.0 | +3.5 |
| Lamp5 | 6 | 1.0 | 5.2 | +4.2 |
| Pvalb | 13 | 1.3 | 5.8 | +4.5 |
| Sst | 16 | 1.1 | 4.7 | +3.6 |
| Vip | 16 | 0.7 | 4.9 | +4.2 |
| Oligodendrocyte | 5 | 0.2 | 3.4 | +3.2 |

### 5.6 Flexibility: DE candidate reserve

If some add-on slots are needed for DE candidate genes (e.g., to spatially validate top snRNAseq DE hits), the diminishing-returns curve provides guidance. The first 70 greedy genes bring 102/126 supertypes to ≥3 markers; even 85 greedy markers achieve ~110. Reserving 10–15 slots for DE candidates is feasible with modest coverage loss, but reserving more than 20 significantly erodes the supertype classification advantage.

---

## 6. Final Recommendation

**Xenium v1 Brain + 100-gene spatial-validated add-on, using all 100 slots for greedy-optimized within-subclass supertype markers.**

| Metric | v1 + 100 add-on | 5K + 100 add-on | 5K alone |
|---|---|---|---|
| Total genes | 366 | 5,101 | 5,001 |
| Sections ($90K budget) | 12 | 6 | 6 |
| Cost per section | $7,339 | $13,083 | $13,083 |
| HIGH confidence supertypes | **122 (97%)** | ~122 (97%) | 55 (44%) |
| All supertypes ≥3 markers | **118/126** | ~118/126 | 54/126 |
| SST supertypes ≥3 markers | **18/18** | 18/18 | — |
| Spatially validated add-on genes | **100/100** | 100/100 | N/A |
| Genome-wide DE | No | Yes | Yes |

The v1+100 design provides:
1. **97% HIGH-confidence supertype classification** — better than any panel except MERSCOPE 4K
2. **12 sections** (6 SCZ + 6 Control) for adequate statistical power
3. **100% spatial-validated genes** — every add-on gene has been measured by MERFISH or MERSCOPE
4. **Better SST coverage than the SST-focused design** — 18/18 vs 13/18 at ≥3 markers

---

## 7. Methods

### 7.1 Within-subclass marker identification

For each of 22 subclasses in the SEA-AD MTG taxonomy (128 supertypes, groups with <20 cells excluded), Wilcoxon rank-sum tests (scanpy `rank_genes_groups`) compared each supertype against all other supertypes in the same subclass. Top 50 markers per supertype, ranked by Wilcoxon score. Reference: SEA-AD MTG snRNAseq (137,303 cells, 33,372 genes), subsampled to max 500 cells per supertype.

### 7.2 Panel coverage and confidence assessment

For each supertype, top-N within-subclass markers (N = 5, 10, 20, 50) present in each panel were counted. Layer specificity assessed using MERFISH-derived median normalized depth. HIGH/MEDIUM/LOW ratings assigned per Section 2.2.

### 7.3 Greedy gene selection

At each step, the candidate gene providing the greatest weighted coverage improvement is selected. Under-covered supertypes receive higher weight: weight = max(0, 5 − current_coverage). Candidates drawn from top-50 within-subclass markers. Spatial-validated version restricted to genes on MERFISH 180, MERSCOPE 250, or MERSCOPE 4K (4,083 total genes).

### 7.4 Cross-platform validation

Within-subclass markers independently computed from SEA-AD MERFISH spatial data (1.9M cells, 180 genes) and compared against snRNAseq-derived markers. Spearman rank and Pearson LFC correlations computed for 874 shared gene-supertype pairs.

### 7.5 Classification fragility (Kwon dataset)

Correlation-based classification margins computed by diagnosis. Neighborhood expression smoothing tested across 9 parameter combinations (K ∈ {15, 30, 50} × α ∈ {0.3, 0.5, 0.7}).

---

## 8. Output Files

All files in `output/marker_analysis/`:

**Add-on gene lists:**

| File | Description |
|---|---|
| `v1_addon_100_spatial_validated.csv` | **Recommended: 100-gene spatial-validated add-on list with panel provenance** |
| `v1_addon_spatial_only_curve.csv` | Greedy add-on progression showing coverage at each gene addition |
| `v1_addon_100_unrestricted.csv` | 100-gene unrestricted add-on list (68% spatial-validated; for reference) |

**Within-subclass markers and cross-platform assessment:**

| File | Description |
|---|---|
| `within_subclass_markers_all.csv` | Wilcoxon markers for all 126 supertypes (snRNAseq-derived) |
| `within_subclass_markers_merfish.csv` | Within-subclass markers computed from MERFISH spatial data |
| `cross_platform_marker_coverage.csv` | Coverage of top-N markers across 6 panels |
| `supertype_classification_confidence.csv` | Per-supertype confidence rating with significance flags |
| `cross_platform_marker_coverage_heatmap.png` | Heatmap: supertypes × panels |
| `cross_platform_marker_coverage_bars.png` | Mean coverage by subclass across panels |

**Kwon dataset classification fragility analysis:**

| File | Description |
|---|---|
| `sst_neighborhood_cell_results.csv` | Cell-level neighborhood smoothing results |
| `sst_neighborhood_sweep.csv` | Parameter sweep for neighborhood smoothing |

---

## 9. Pricing Source

10x Genomics Xenium Custom Panels (v1 chemistry, accessed March 2026):
- Add-On Custom Panel: 51–100 genes, $5,650 (4 rxn) / $8,000 (16 rxn)
- Predesigned Human Brain Panel: $345 per 2 reactions
- UPitt Xenium Core: $11,000/run + $2,000 cell segmentation (v1); $23,500/run (5K Prime)
