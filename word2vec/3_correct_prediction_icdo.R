# ============================================================
# Evaluate ICD-O predictions (Top-k) from nn_results_topk_icdo.csv
# - Evaluates ONLY ICD-O (ground truth: "ICD-O-Code")
# - Uses suggestedMetadata* as predictions (must contain ICD-O codes)
# - Keeps all rows; computes Top-1, Top-k, Parent-level (auto: Topography vs Morphology), Rank, MRR
# - Saves per-row and summary CSVs next to the input file
# ============================================================

# Packages
library(readr)
library(dplyr)
library(stringr)
library(tibble)
library(purrr)

# ==== INPUT (adjust) ====
input_csv <- "C:/Users/ali0f/Documents/word2vec/outputs_w2v_icdo/nn_results_topk_icdo.csv"
# ========================

# -------------------------
# Helpers
# -------------------------
normalize_code <- function(x) str_trim(toupper(as.character(x)))

# Decide whether a code looks like ICD-O Morphology (e.g., 8140/3) vs Topography (e.g., C50.9)
is_morph_code <- function(x) {
  x <- normalize_code(x)
  # ICD-O morphology typically: 4 digits + "/" + 1 digit (behavior)
  str_detect(x, "^\\d{4}/\\d$")
}

get_topo3 <- function(x) {
  # Topography parent: Letter + 2 digits (e.g., C50.9 -> C50)
  m <- stringr::str_match(normalize_code(x), "^([A-Z]\\d{2})")
  m[, 2]
}

get_morph4 <- function(x) {
  # Morphology parent: first 4 digits (e.g., 8140/3 -> 8140)
  m <- stringr::str_match(normalize_code(x), "^(\\d{4})")
  m[, 2]
}

# -------------------------
# Read CSV
# -------------------------
df <- readr::read_delim(input_csv, delim = ";", col_types = cols(.default = "c"))

# Required columns for ICD-O evaluation
stopifnot("ICD-O-Code" %in% names(df))

# Predicted columns (suggestedMetadata1..k)
sm_cols <- grep("^suggestedMetadata\\d+$", names(df), value = TRUE)
if (length(sm_cols) == 0 && "suggestedMetadata" %in% names(df)) sm_cols <- "suggestedMetadata"
if (length(sm_cols) == 0) stop("Keine Spalten gefunden, die mit 'suggestedMetadata' beginnen.")

# -------------------------
# Build suggestion list per row (order preserved)
# -------------------------
sugg_list <- df[sm_cols] %>%
  as.data.frame() %>%
  asplit(1) %>%
  lapply(\(v) {
    v <- as.character(unlist(v, use.names = FALSE))
    v <- v[!is.na(v) & nzchar(v)]
    v
  })

# Ground truth ICD-O codes
gold      <- df$`ICD-O-Code`
gold_norm <- normalize_code(gold)

# Normalized suggestions
sugg_norm_list <- lapply(sugg_list, normalize_code)

# -------------------------
# Auto-select parent logic per row:
# - If gold looks like morphology (dddd/d): parent = morph4 (dddd)
# - Else: parent = topo3 (A00 style like C50)
# -------------------------
gold_is_morph <- is_morph_code(gold_norm)
gold_parent <- ifelse(gold_is_morph, get_morph4(gold_norm), get_topo3(gold_norm))

parent_of_vec <- function(vec, morph_flag) {
  if (!length(vec)) return(character(0))
  if (isTRUE(morph_flag)) {
    get_morph4(vec)
  } else {
    get_topo3(vec)
  }
}

# -------------------------
# Metrics
# -------------------------

# Top-1 exact match
top1_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA)
  identical(sugg[[1]], g)
}, sugg_norm_list, gold_norm)

# Top-k exact match
topk_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA)
  any(sugg == g)
}, sugg_norm_list, gold_norm)

# Top-k parent match (topography parent or morphology parent, depending on gold)
topk_parent <- mapply(function(sugg, gpar, morph_flag) {
  if (length(sugg) == 0 || is.na(gpar) || !nzchar(gpar)) return(NA)
  parents <- parent_of_vec(sugg, morph_flag)
  any(parents == gpar, na.rm = TRUE)
}, sugg_norm_list, gold_parent, gold_is_morph)

# Rank exact + MRR
rank_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA_integer_)
  m <- match(g, sugg)
  if (is.na(m)) NA_integer_ else as.integer(m)
}, sugg_norm_list, gold_norm)

mrr_exact <- ifelse(is.na(rank_exact), 0, 1 / rank_exact)

# -------------------------
# Summary
# -------------------------
n_all  <- nrow(df)
n_eval <- sum(!is.na(topk_exact))  # rows with usable gold/suggestions

summary_tbl <- tibble(
  rows_total            = n_all,
  rows_evaluated        = n_eval,
  top1_exact_hits       = sum(top1_exact, na.rm = TRUE),
  top1_exact_rate_pct   = round(mean(top1_exact, na.rm = TRUE) * 100, 2),
  topk_exact_hits       = sum(topk_exact, na.rm = TRUE),
  topk_exact_rate_pct   = round(mean(topk_exact, na.rm = TRUE) * 100, 2),
  topk_parent_hits      = sum(topk_parent, na.rm = TRUE),
  topk_parent_rate_pct  = round(mean(topk_parent, na.rm = TRUE) * 100, 2),
  mrr_exact_mean        = round(mean(mrr_exact, na.rm = TRUE), 4)
)

print(summary_tbl)

# -------------------------
# Per-row table
# -------------------------
eval_per_row <- df %>%
  mutate(
    gold_icdo          = gold,
    gold_icdo_norm     = gold_norm,
    gold_is_morph      = gold_is_morph,
    gold_parent        = gold_parent,
    top1_exact         = top1_exact,
    topk_exact         = topk_exact,
    topk_parent        = topk_parent,
    rank_exact         = rank_exact,
    mrr_exact          = mrr_exact
  )

# -------------------------
# Save outputs
# -------------------------
out_dir <- dirname(input_csv)
readr::write_delim(eval_per_row, file.path(out_dir, "icdo_eval_per_row.csv"), delim = ";")
readr::write_delim(summary_tbl,  file.path(out_dir, "icdo_eval_summary.csv"),  delim = ";")

cat(
  "Fertig. Dateien gespeichert in:\n  - ",
  file.path(out_dir, "icdo_eval_per_row.csv"),
  "\n  - ",
  file.path(out_dir, "icdo_eval_summary.csv"),
  "\n",
  sep = ""
)
