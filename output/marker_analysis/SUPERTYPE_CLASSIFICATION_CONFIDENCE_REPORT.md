# Supertype Classification Confidence in the SCZ Xenium Dataset

**Date**: 2026-03-14
**Project**: SCZ_Xenium (Kwon et al. Schizophrenia Xenium spatial transcriptomics)

---

## 1. Motivation

Our density and compositional (crumblr) analyses identified ~20 supertypes with nominally significant or FDR-significant changes between SCZ and Control in the Kwon Xenium dataset. When we cross-referenced these effects with Nicole's snRNAseq meta-analysis of SCZ, several Sst supertypes showed striking discordance:

- **Sst_20**: increased in Xenium (density logFC +0.17) but *decreased* in snRNAseq (beta -0.12)
- **Sst_22**: null effect in Xenium but decreased in snRNAseq

This raised the question: **are these real biological differences between platforms, or artifacts of misclassification?**

The Kwon dataset uses a ~300-gene custom Xenium panel. When we computed within-subclass Wilcoxon markers (one-vs-rest within Sst supertypes using the SEA-AD snRNAseq reference), we found that **zero of the top-10 Sst supertype-discriminating markers** are present in this panel. The correlation classifier must distinguish 16 Sst supertypes using subtle expression differences across non-specific genes — and SCZ itself shifts 54–112 of those panel genes per Sst supertype (FDR < 0.05), directly corrupting the profiles the classifier relies on.

We confirmed classification fragility directly: Sst_3 cells in SCZ show significantly reduced classification margins (p = 2.3×10⁻¹⁶), with Sst_20 as a top confusion partner — potentially explaining the Sst_20 inflation in SCZ. Spatial neighborhood expression smoothing (9 parameter combinations: K ∈ {15, 30, 50} × α ∈ {0.3, 0.5, 0.7}) could not rescue classification; without discriminating markers, smoothing reinforces whatever label the classifier already assigned.

This motivated a **systematic assessment**: for every supertype across all subclasses, how confident can we be in classification given a particular gene panel?

---

## 2. Panel Design Determines Supertype Resolution

### 2.1 The central finding: gene count ≠ supertype discrimination

We assessed six spatial gene panels for their coverage of within-subclass supertype markers. The results reveal that **panel curation for within-subclass discrimination matters far more than total gene count:**

| Panel | Genes | Design | Mean top-10 | ≥3 markers | HIGH confidence |
|---|---|---|---|---|---|
| **Kwon 300** | 300 | Custom (study-specific) | **0.7/10** | **4/126 (3%)** | **4 (3%)** |
| Xenium v1 Brain | 266 | Predesigned (brain) | 1.0/10 | 12/126 (10%) | 14 (11%) |
| SEA-AD MERFISH | 180 | Allen Institute (taxonomy) | 2.0/10 | 45/126 (36%) | 46 (37%) |
| **MERSCOPE 250** | **250** | **Vizgen (brain types)** | **2.9/10** | **71/126 (56%)** | **72 (57%)** |
| Xenium 5K Prime | 5,001 | Pan-tissue | 2.4/10 | 54/126 (43%) | 55 (44%) |
| MERSCOPE 4K | 3,999 | Near-transcriptome | 5.6/10 | 121/126 (96%) | 121 (96%) |

Three observations stand out:

1. **MERSCOPE 250 (250 genes) outperforms Xenium 5K (5,001 genes)** for supertype classification: 57% vs 44% HIGH confidence. The 250-gene panel was specifically designed for brain cell type classification with curated within-subclass markers, while the 5K panel broadly covers cell lineages across organs.

2. **MERFISH 180, with fewer genes than either Xenium panel, achieves better coverage** (mean 2.0/10 vs 1.0) because the Allen Institute specifically selected genes for within-subclass discrimination in their brain taxonomy.

3. **The Kwon 300-gene panel performs worst** despite having more genes than MERFISH 180. It was designed for subclass-level cell typing and specific biological questions, not fine-grained supertype discrimination. It shares only 52 genes with Xenium v1, 23 with MERFISH 180, and 15 with MERSCOPE 250.

**Subclass-level classification is robust across all panels.** The challenge is specifically at the supertype level within subclasses.

### 2.2 Confidence rating framework

Each supertype received a classification confidence rating based on two independent axes — marker coverage and cortical layer specificity:

- **HIGH**: ≥3 of top-10 within-subclass markers in panel, OR ≥2 markers + layer-distinct
- **MEDIUM**: 2 markers (no layer distinction), OR 1 marker + layer-distinct, OR 0 markers + layer-distinct
- **LOW**: 0–1 markers and no layer distinction

Layer specificity was assessed using median normalized cortical depth from the SEA-AD MERFISH reference (368K depth-annotated cells). Supertypes separated by >0.10 normalized depth units from their nearest within-subclass neighbor were flagged as "layer-distinct." However, only 7 supertypes qualify (Chandelier_1/2, Sst Chodl_1/2, Astro_3, Micro-PVM_2, Pax6_1). The large subclasses driving most supertype results (Sst: 16 types, Pvalb: 13, Vip: 16, L2/3 IT: 10) have too many supertypes packed into narrow depth ranges for layer information to help.

---

## 3. Implications for the Kwon SCZ Xenium Dataset

### 3.1 What results can be trusted

**High confidence (trust the result):**
- All subclass-level composition and density results
- Supertype results where snRNAseq meta-analysis shows concordant direction — cross-platform agreement is the strongest validation
- Micro-PVM_2 (FDR 0.093, logFC -0.63; 2 markers + layer separation)

**Medium confidence (suggestive, requires external validation):**
- L5/6 NP_3, L6b_4, Vip_15, Sst_5: each has 2 within-subclass markers in the Kwon panel

**Low confidence (interpret with caution):**
- 15 of 20 significant supertypes, including the most statistically significant hits:
  - L2/3 IT_8 (FDR 0.032, most significant) — zero within-subclass markers
  - Pvalb_13 (FDR 0.038) — 1 marker (TRHDE)
  - Sst_3 (FDR 0.045) — zero markers, confirmed confusion with Sst_19/20
  - Sst_25 (FDR 0.057) — 1 marker shared with Sst_20 (CDH12)
  - Oligo_4 (FDR 0.076) — 1 marker (OPALIN), all Oligo types at same depth

### 3.2 The Sst supertype problem

The Sst discordance with snRNAseq is most parsimoniously explained by classification fragility:

1. The Kwon panel contains 0–1 of the top-10 within-Sst supertype markers for each Sst type
2. SCZ shifts 54–112 panel genes per Sst supertype, directly affecting the expression profiles used for classification
3. Sst_3 classification margins drop significantly in SCZ (p = 2.3×10⁻¹⁶), with Sst_20 as a confusion partner
4. Spatial neighborhood smoothing cannot compensate — it amplifies existing classification biases

The Sst *subclass*-level density result remains trustworthy (SST itself and other subclass markers are well-represented), and the overall downward trend in Sst cells in SCZ is consistent with snRNAseq.

### 3.3 Recommendations for interpretation

1. **Present subclass-level results as primary findings.** These are robust to classification uncertainty.
2. **Present supertype results with an explicit confidence rating.** Include the marker coverage table as supplementary material.
3. **Use snRNAseq concordance as the key validation.** Cross-platform agreement strengthens findings despite limited panel coverage.
4. **Do not interpret Sst supertype discordance as biological.** The Sst_20/Sst_22 Xenium-vs-snRNAseq disagreement is most likely a classification artifact.

---

## 4. Designing a Better Panel: Greedy Gene Selection

### 4.1 The add-on opportunity

The Xenium v1 human brain panel (266 genes) is a natural baseline for improvement, since it is the standard predesigned brain panel and accepts a 100-gene custom add-on. We used a greedy algorithm to select optimal add-on genes: at each step, the gene providing the greatest coverage improvement across under-covered supertypes is selected, with supertypes having fewer current markers weighted more heavily.

The progression shows steep diminishing returns — the first 18 genes contribute as much HIGH confidence gain as the next 82 combined:

| Add-on genes | Panel size | Mean top-10 | HIGH | LOW | Milestone |
|---|---|---|---|---|---|
| 0 | 266 | 1.0 | 14 (11%) | 93 (74%) | Xenium v1 baseline |
| 5 | 271 | 1.3 | 22 (17%) | 82 (65%) | |
| **16** | **282** | **2.0** | **40 (32%)** | **59 (47%)** | **Matches MERFISH mean coverage** |
| **18** | **284** | **2.1** | **46 (37%)** | **55 (44%)** | **Matches MERFISH HIGH count** |
| 25 | 291 | 2.5 | 55 (44%) | 45 (36%) | Matches Xenium 5K HIGH count |
| 36 | 302 | 3.0 | 72 (57%) | 28 (22%) | Matches MERSCOPE 250 HIGH count |
| **100** | **366** | **4.6** | **122 (97%)** | **0 (0%)** | **Surpasses MERSCOPE 4K** |

Just 18 targeted genes would bring Xenium v1 from 11% to 37% HIGH confidence, matching what the Allen Institute achieved with a fully custom 180-gene panel. The full 100-gene add-on achieves 97% HIGH confidence — surpassing every existing panel except MERSCOPE 4K.

### 4.2 Spatial validation of recommended genes

A critical question: do these markers — identified from snRNAseq — actually discriminate supertypes when measured spatially? We validated this three ways:

**Cross-platform rank concordance.** We computed within-subclass markers directly from SEA-AD MERFISH spatial data (1.9M cells, 180 genes) and compared against snRNAseq-derived markers. For 874 gene-supertype pairs measurable on both platforms:
- Log fold change correlation: **r = 0.60** (p = 2.7×10⁻⁸⁶)
- For genes ranked top-10 in snRNAseq: **79% are also top-10 in MERFISH**, 94% are top-20

The markers are genuine supertype discriminators across platforms.

**Spatial-restricted gene selection.** We repeated the greedy selection restricting candidates to genes measured on at least one spatial platform (MERFISH 180 ∪ MERSCOPE 250 ∪ MERSCOPE 4K = 4,083 genes). At 100 genes, both unrestricted and spatial-only approaches achieve **identical ≥3-marker supertype counts** (118/126). The spatial-only list is the safer recommendation; 32 of the original 100 unrestricted genes (including several lncRNAs) have never been FISH-validated.

**Independent MERFISH-derived markers.** Computing markers directly from MERFISH spatial data and running greedy selection independently yields the same gene families: ROBO1, ROBO2, RGS6, KIRREL3, GPC5 — axon guidance and cell adhesion molecules. Four genes appear in both the snRNAseq-derived and MERFISH-derived top-20 lists.

### 4.3 Recommended add-on gene list (spatial-validated)

The top 20 genes, restricted to those proven detectable by FISH:

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

These are overwhelmingly **neuronal cell adhesion molecules, axon guidance receptors, and ion channel subunits** — gene families that define neuronal subtype identity through synaptic connectivity and spatial positioning, but which are not typically prioritized in panels designed for broad cell type identification. Six of the top 20 (RGS6, ROBO2, LRP1B, NLGN1, KCNIP4, ROBO1) were independently selected by the Allen Institute for their MERFISH 180 panel.

The full 100-gene spatial-validated list is in `v1_addon_100_spatial_validated.csv`.

### 4.4 Per-subclass improvement with 100 add-on genes

| Subclass | # types | v1 mean top-10 | v1+100 mean top-10 | Δ |
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

Every subclass improves. The largest gains are in Chandelier (+5.0), L5 ET (+4.5), Pvalb (+4.5), and Vip (+4.2).

---

## 5. Methods

### 5.1 Within-subclass marker identification

For each of the 22 subclasses in the SEA-AD middle temporal gyrus taxonomy (128 supertypes total after excluding groups with <20 cells), we ran Wilcoxon rank-sum tests (scanpy `rank_genes_groups`, method="wilcoxon") within each subclass, comparing each supertype against all other supertypes in the same subclass. This produces markers that distinguish *within-subclass siblings* (e.g., Sst_20 vs other Sst types), not subclass-level markers (Sst vs Pvalb). We extracted the top 50 markers per supertype, ranked by Wilcoxon score.

Reference data: SEA-AD MTG snRNAseq (137,303 cells, 33,372 genes), subsampled to max 500 cells per supertype.

### 5.2 Panel coverage and confidence assessment

For each supertype, we counted how many of its top-N within-subclass markers (N = 5, 10, 20, 50) are present in each spatial panel. Layer specificity was assessed using median normalized cortical depth from the SEA-AD MERFISH reference. Supertypes were rated HIGH/MEDIUM/LOW confidence based on marker coverage and layer separation (see Section 2.2).

### 5.3 Greedy gene selection

At each step, the candidate gene providing the greatest weighted coverage improvement is selected. Under-covered supertypes receive higher weight: weight = max(0, 5 − current_coverage). Candidates were drawn from the top-50 within-subclass markers per supertype. For the spatial-validated version, candidates were restricted to genes measured on at least one spatial platform (MERFISH 180 ∪ MERSCOPE 250 ∪ MERSCOPE 4K = 4,083 genes).

### 5.4 Cross-platform marker validation

Within-subclass Wilcoxon markers were independently computed from SEA-AD MERFISH spatial data (1.9M cells, 180 genes) and compared against snRNAseq-derived markers to assess cross-platform concordance. Spearman rank correlation and Pearson log fold change correlation were computed for 874 gene-supertype pairs measurable on both platforms.

### 5.5 Classification fragility assessment

Classification fragility for Sst supertypes was validated by computing correlation-based classification margins by diagnosis and testing neighborhood expression smoothing across 9 parameter combinations (K ∈ {15, 30, 50} × α ∈ {0.3, 0.5, 0.7}).

---

## 6. Output Files

All files in `output/marker_analysis/`:

| File | Description |
|---|---|
| `within_subclass_markers_all.csv` | Wilcoxon markers for all 126 supertypes (snRNAseq-derived) |
| `within_subclass_markers_merfish.csv` | Within-subclass markers computed from MERFISH spatial data |
| `cross_platform_marker_coverage.csv` | Coverage of top-N markers across 6 panels |
| `supertype_classification_confidence.csv` | Per-supertype confidence rating with significance flags |
| `v1_addon_100_spatial_validated.csv` | **Recommended: 100-gene spatial-validated add-on list with panel provenance** |
| `v1_addon_100_unrestricted.csv` | 100-gene unrestricted add-on list (68% spatial-validated) |
| `v1_addon_spatial_only_curve.csv` | Greedy add-on progression (spatial-restricted) |
| `addon_gap_analysis.csv` | Per-supertype deficit and genes needed to match MERSCOPE 250 |
| `addon_gene_merfish_validation.csv` | MERFISH validation of add-on gene detectability |
| `cross_platform_marker_coverage_heatmap.png` | Heatmap: supertypes × panels |
| `cross_platform_marker_coverage_bars.png` | Mean coverage by subclass across panels |
