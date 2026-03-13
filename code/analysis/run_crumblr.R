#!/usr/bin/env Rscript
#
# crumblr compositional analysis: SCZ vs Control
#
# Runs crumblr + dream on cropped whole-composition data
# (cortical cells only, neurons + non-neurons together).
#
# Input:  output/crumblr/crumblr_input_{subclass,cluster}.csv
# Output: output/crumblr/crumblr_results_{subclass,cluster}.csv
#         output/crumblr/crumblr_results_all.csv
#
# Formula: ~ diagnosis + sex + scale(age)

library(crumblr)
library(dreamlet)
library(variancePartition)
library(limma)

cat("Libraries loaded\n")

# ── Paths ──────────────────────────────────────────────────────────
base_dir <- path.expand("~/Github/SCZ_Xenium")
in_dir <- file.path(base_dir, "output", "crumblr")
out_dir <- in_dir  # results go in same directory

# ── Parse command line args ────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
suffix <- ""
if (length(args) > 0 && args[1] == "--corr") {
  suffix <- "_corr"
  cat("Running with CORR QC inputs (non-default)\n")
} else if (length(args) > 0 && args[1] == "--no-high-umi") {
  suffix <- "_no_high_umi"
  cat("Running with high-UMI cells EXCLUDED\n")
} else if (length(args) > 0 && startsWith(args[1], "--suffix")) {
  # Generic suffix: --suffix _pctl05
  if (length(args) > 1) {
    suffix <- args[2]
  } else {
    suffix <- sub("^--suffix=?", "", args[1])
  }
  cat(sprintf("Running with suffix: %s\n", suffix))
}

# ── Discover input files ───────────────────────────────────────────
if (suffix == "") {
  # Default (hybrid): match files that do NOT end with a known suffix
  all_files <- Sys.glob(file.path(in_dir, "crumblr_input_*.csv"))
  input_files <- all_files[!grepl("_(corr|hybrid|no_high_umi|pctl\\d+|margin_\\w+)\\.csv$", all_files)]
} else {
  input_files <- Sys.glob(file.path(in_dir, sprintf("crumblr_input_*%s.csv", suffix)))
}
cat(sprintf("Found %d input files:\n", length(input_files)))
for (f in input_files) cat(sprintf("  %s\n", basename(f)))

all_results <- list()
idx <- 1

for (fpath in input_files) {
  # Parse level from filename: crumblr_input_{level}[_hybrid].csv
  fname <- tools::file_path_sans_ext(basename(fpath))
  level <- sub("crumblr_input_", "", fname)
  level <- sub("_(corr|hybrid|no_high_umi|pctl\\d+|margin_\\w+)$", "", level)  # strip suffix for clean level name
  cat(sprintf("\n══ Processing: %s ══\n", level))

  # ── Load data ──────────────────────────────────────────────────
  df <- read.csv(fpath, stringsAsFactors = FALSE)
  cat(sprintf("  %d rows, %d donors, %d cell types\n",
              nrow(df), length(unique(df$donor)), length(unique(df$celltype))))

  # ── Pivot to wide count matrix (donors × cell types) ──────────
  count_wide <- reshape(df[, c("donor", "celltype", "count")],
                        idvar = "donor", timevar = "celltype",
                        direction = "wide")
  rownames(count_wide) <- count_wide$donor
  count_wide$donor <- NULL
  colnames(count_wide) <- sub("^count\\.", "", colnames(count_wide))
  count_wide[is.na(count_wide)] <- 0

  # ── Filter: cell types present in ≥50% of samples ─────────────
  presence <- colMeans(count_wide > 0)
  keep <- presence >= 0.5
  cat(sprintf("  Keeping %d / %d types (≥50%% presence)\n",
              sum(keep), length(keep)))
  count_wide <- count_wide[, keep, drop = FALSE]

  if (ncol(count_wide) < 2) {
    cat("  Skipping: fewer than 2 cell types\n")
    next
  }

  # ── Build metadata ─────────────────────────────────────────────
  meta <- unique(df[, c("donor", "diagnosis", "sex", "age")])
  rownames(meta) <- meta$donor
  meta <- meta[rownames(count_wide), ]
  meta$diagnosis <- factor(meta$diagnosis, levels = c("Control", "SCZ"))
  meta$sex <- factor(meta$sex)
  meta$age_num <- as.numeric(meta$age)

  cat(sprintf("  %d Control, %d SCZ\n",
              sum(meta$diagnosis == "Control"),
              sum(meta$diagnosis == "SCZ")))

  # ── Run crumblr ────────────────────────────────────────────────
  count_mat <- as.matrix(count_wide)

  cobj <- tryCatch(
    crumblr(count_mat),
    error = function(e) {
      cat(sprintf("  crumblr error: %s\n", e$message))
      return(NULL)
    }
  )
  if (is.null(cobj)) next

  # ── Fit dream model ────────────────────────────────────────────
  form <- ~ diagnosis + sex + scale(age_num)

  fit <- tryCatch({
    f <- dream(cobj, form, meta)
    eBayes(f)
  }, error = function(e) {
    cat(sprintf("  dream error: %s\n", e$message))
    return(NULL)
  })
  if (is.null(fit)) next

  # ── Extract results ────────────────────────────────────────────
  res <- topTable(fit, coef = "diagnosisSCZ", number = Inf, sort.by = "none")
  res$celltype <- rownames(res)
  res$level <- level

  # Standard error: SE = logFC / t
  res$SE <- res$logFC / res$t

  # Per-level FDR
  res$FDR <- p.adjust(res$P.Value, method = "BH")

  # Save individual results
  res_sorted <- res[order(res$P.Value), ]
  out_file <- file.path(out_dir, sprintf("crumblr_results_%s%s.csv", level, suffix))
  write.csv(res_sorted, out_file, row.names = FALSE)
  cat(sprintf("  Saved: %s (%d types)\n", basename(out_file), nrow(res)))

  all_results[[idx]] <- res
  idx <- idx + 1

  # ── Print summary ────────────────────────────────────────────
  n_fdr05 <- sum(res$FDR < 0.05)
  n_fdr10 <- sum(res$FDR < 0.10)
  n_nom05 <- sum(res$P.Value < 0.05)
  cat(sprintf("  FDR < 0.05: %d | FDR < 0.10: %d | nom p < 0.05: %d\n",
              n_fdr05, n_fdr10, n_nom05))

  # Show significant hits
  sig <- res[res$FDR < 0.10, ]
  sig <- sig[order(sig$P.Value), ]
  if (nrow(sig) > 0) {
    cat("  FDR < 0.10 hits:\n")
    for (i in 1:nrow(sig)) {
      r <- sig[i, ]
      d <- ifelse(r$logFC > 0, "↑SCZ", "↓SCZ")
      cat(sprintf("    %-30s %s logFC=%+.4f p=%.5f FDR=%.4f\n",
                  r$celltype, d, r$logFC, r$P.Value, r$FDR))
    }
  }
}

# ── Combine all results ────────────────────────────────────────────
if (length(all_results) > 0) {
  combined <- do.call(rbind, all_results)
  combined <- combined[order(combined$P.Value), ]
  out_file <- file.path(out_dir, sprintf("crumblr_results_all%s.csv", suffix))
  write.csv(combined, out_file, row.names = FALSE)
  cat(sprintf("\nSaved combined: %s (%d total rows)\n",
              basename(out_file), nrow(combined)))
} else {
  cat("\nNo results to combine!\n")
}

cat("\nDone!\n")
