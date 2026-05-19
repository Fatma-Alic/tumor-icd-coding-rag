# ============================================================
# Evaluate ICD-O predictions (Top-1) + exact/parent metrics + F1
# - Evaluates ONLY ICD-O (ground truth column: "ICD-O-Code")
# - Predictions are taken from suggestedMetadata* columns (Top-1 = first)
# - Robust: handles Topography (e.g., C50.9) vs Morphology (e.g., 8140/3)
# - Parent/partial match:
#     * Topography parent = Letter + 2 digits (C50)
#     * Morphology parent = first 4 digits (8140)
# - Epoch support kept (if no epoch column exists, uses "0")
# - Saves per-row and summary CSVs next to the input file
# ============================================================

# Pakete
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)
library(tibble)

# ==== Eingabe anpassen ====
input_csv <- "C:/Users/ali0f/Documents/word2vec/outputs_w2v_icdo/nn_results_topk_icdo.csv"
# ==========================

# -------------------------
# Hilfsfunktionen (ICD-O)
# -------------------------
normalize_code <- function(x) str_trim(toupper(as.character(x)))

# ICD-O Topography: Letter + 2 digits (+ optional .something)  e.g., C50.9
is_valid_icdo_topo <- function(x) {
  x <- normalize_code(x)
  !is.na(x) & nzchar(x) & str_detect(x, "^[A-Z]\\d{2}(\\.[A-Z0-9]{1,4})?$")
}

# ICD-O Morphology: 4 digits / 1 digit  e.g., 8140/3
is_valid_icdo_morph <- function(x) {
  x <- normalize_code(x)
  !is.na(x) & nzchar(x) & str_detect(x, "^\\d{4}/\\d$")
}

# Any valid ICD-O (either topo or morph)
is_valid_icdo <- function(x) is_valid_icdo_topo(x) | is_valid_icdo_morph(x)

# Parent extraction:
# - Topography parent: C50.9 -> C50
get_topo_parent <- function(x) {
  m <- stringr::str_match(normalize_code(x), "^([A-Z]\\d{2})")
  m[, 2]
}
# - Morphology parent: 8140/3 -> 8140
get_morph_parent <- function(x) {
  m <- stringr::str_match(normalize_code(x), "^(\\d{4})")
  m[, 2]
}

# For a given code vector, return parent vector depending on morph_flag
parents_of_vec <- function(vec, morph_flag) {
  if (!length(vec)) return(character(0))
  if (isTRUE(morph_flag)) get_morph_parent(vec) else get_topo_parent(vec)
}

# Detect bool-only answers (should not happen here, but kept for compatibility)
is_bool_only <- function(x) {
  x <- tolower(str_trim(as.character(x)))
  !is.na(x) & nzchar(x) & x %in% c("true", "false", "yes", "no")
}

# -------------------------
# F1-Helfer (macro & weighted) für Klassenvorhersagen
# -------------------------
f1_macro_weighted <- function(y_true, y_pred) {
  valid_idx <- !is.na(y_true)
  y_true <- y_true[valid_idx]
  y_pred <- y_pred[valid_idx]
  
  classes <- sort(unique(y_true))
  supports <- sapply(classes, function(c) sum(y_true == c))
  
  f1_per_class <- sapply(classes, function(c) {
    tp <- sum(y_true == c & y_pred == c, na.rm = TRUE)
    fp <- sum(y_true != c & y_pred == c, na.rm = TRUE)
    fn <- sum(y_true == c & (is.na(y_pred) | y_pred != c), na.rm = TRUE)
    prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
  })
  
  macro <- mean(f1_per_class)
  weighted <- if (sum(supports) == 0) 0 else sum(f1_per_class * supports) / sum(supports)
  list(macro = macro, weighted = weighted)
}

# -------------------------
# CSV lesen
# -------------------------
df <- readr::read_delim(input_csv, delim = ";", col_types = cols(.default = "c"))

# Pflichtspalte prüfen (ICD-O)
stopifnot("ICD-O-Code" %in% names(df))

# Evtl. Epochen-Spalte identifizieren
epoch_col <- c("Epoch", "epoch", "EPOCH")
epoch_col <- epoch_col[epoch_col %in% names(df)]
if (length(epoch_col) == 0) {
  df$Epoch <- "0"
  epoch_col <- "Epoch"
}

# Suggested-Metadata-Spalten finden (Top-k)
sm_cols <- grep("^suggestedMetadata\\d+$", names(df), value = TRUE)
if (length(sm_cols) == 0 && "suggestedMetadata" %in% names(df)) sm_cols <- "suggestedMetadata"
if (length(sm_cols) == 0) stop("Keine Spalten gefunden, die mit 'suggestedMetadata' beginnen.")

# Liste der Vorschläge pro Zeile (geordnet)
sugg_list <- df[sm_cols] %>%
  as.data.frame() %>%
  asplit(1) %>%
  lapply(\(v) {
    v <- as.character(unlist(v, use.names = FALSE))
    v[!is.na(v) & nzchar(v)]
  })

# -------------------------
# Gold-Labels (ICD-O)
# -------------------------
gold_raw  <- df$`ICD-O-Code`
gold_norm <- normalize_code(gold_raw)

gold_is_morph <- is_valid_icdo_morph(gold_norm)
gold_is_topo  <- is_valid_icdo_topo(gold_norm)

# Parent for gold (row-wise)
gold_parent <- ifelse(gold_is_morph, get_morph_parent(gold_norm), get_topo_parent(gold_norm))

# -------------------------
# Top-1 Prediction
# -------------------------
pred_top1_raw  <- vapply(sugg_list, function(v) if (length(v) == 0) NA_character_ else v[[1]], character(1))
pred_top1_norm <- normalize_code(pred_top1_raw)

pred_is_bool    <- is_bool_only(pred_top1_raw)
pred_is_valid   <- is_valid_icdo(pred_top1_raw)
pred_is_invalid <- !(pred_is_valid | pred_is_bool)

pred_is_morph <- is_valid_icdo_morph(pred_top1_norm)
pred_is_topo  <- is_valid_icdo_topo(pred_top1_norm)

pred_parent <- ifelse(pred_is_morph, get_morph_parent(pred_top1_norm), get_topo_parent(pred_top1_norm))

# -------------------------
# Matching-Flags (ICD-O)
# -------------------------
eval_mask <- !is.na(gold_norm) & nzchar(gold_norm) & is_valid_icdo(gold_norm)

exact_match <- eval_mask & pred_is_valid & !is.na(pred_top1_norm) & (pred_top1_norm == gold_norm)

# Partial: same parent type AND same parent value
# - If gold is morph -> compare morph4
# - else -> compare topo3
partial_match <- eval_mask & pred_is_valid & !exact_match &
  ifelse(gold_is_morph,
         (!is.na(gold_parent) & !is.na(get_morph_parent(pred_top1_norm)) & (get_morph_parent(pred_top1_norm) == gold_parent)),
         (!is.na(gold_parent) & !is.na(get_topo_parent(pred_top1_norm))  & (get_topo_parent(pred_top1_norm)  == gold_parent))
  )

false_code <- eval_mask & pred_is_valid & !exact_match & !partial_match
false_bool <- eval_mask & pred_is_bool
invalid_answer <- eval_mask & pred_is_invalid

# -------------------------
# MRR / Rank (exact) from full top-k list
# -------------------------
# Normalize full suggestion lists
sugg_norm_list <- lapply(sugg_list, normalize_code)

rank_exact <- mapply(function(sugg, g) {
  if (!eval_mask[1] && FALSE) return(NA_integer_) # no-op guard (keeps signature)
  if (length(sugg) == 0 || is.na(g) || !nzchar(g) || !is_valid_icdo(g)) return(NA_integer_)
  m <- match(g, sugg)
  if (is.na(m)) NA_integer_ else as.integer(m)
}, sugg_norm_list, gold_norm)

mrr_exact <- ifelse(is.na(rank_exact), 0, 1 / rank_exact)

# -------------------------
# Parent hit within top-k (row-wise type)
# -------------------------
topk_parent_hit <- mapply(function(sugg, gpar, morph_flag, ok) {
  if (!ok || length(sugg) == 0 || is.na(gpar) || !nzchar(gpar)) return(NA)
  parents <- parents_of_vec(sugg, morph_flag)
  any(parents == gpar, na.rm = TRUE)
}, sugg_norm_list, gold_parent, gold_is_morph, eval_mask)

# Top-k exact hit
topk_exact_hit <- mapply(function(sugg, g, ok) {
  if (!ok || length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA)
  any(sugg == g)
}, sugg_norm_list, gold_norm, eval_mask)

# -------------------------
# Summary-Funktion (gesamt & je Epoch)
# -------------------------
compute_summary <- function(idx) {
  n_total <- length(idx)
  n_eval  <- sum(eval_mask[idx])
  
  exact    <- sum(exact_match[idx] & eval_mask[idx], na.rm = TRUE)
  partial  <- sum(partial_match[idx] & eval_mask[idx], na.rm = TRUE)
  f_code   <- sum(false_code[idx]   & eval_mask[idx], na.rm = TRUE)
  f_bool   <- sum(false_bool[idx]   & eval_mask[idx], na.rm = TRUE)
  invalid  <- sum(invalid_answer[idx] & eval_mask[idx], na.rm = TRUE)
  
  # Rates
  acc         <- if (n_eval == 0) 0 else exact / n_eval
  partial_acc <- if (n_eval == 0) 0 else (exact + partial) / n_eval
  invalid_rate <- if (n_eval == 0) 0 else invalid / n_eval
  error_rate   <- if (n_eval == 0) 0 else (f_code + f_bool + invalid) / n_eval
  
  # Answer length (chars of raw Top-1), only eval rows
  avg_len <- mean(nchar(pred_top1_raw[idx][eval_mask[idx] & !is.na(pred_top1_raw[idx])]), na.rm = TRUE)
  if (is.nan(avg_len)) avg_len <- 0
  
  # F1 Exact (classes = exact ICD-O codes), pred = NA for invalid/bool
  y_true_exact <- gold_norm[idx]
  y_pred_exact <- ifelse(pred_is_valid[idx] & eval_mask[idx], pred_top1_norm[idx], NA_character_)
  f1e <- f1_macro_weighted(y_true_exact, y_pred_exact)
  
  # F1 Partial (classes = parent code based on gold type), pred parent based on gold type too
  y_true_part <- gold_parent[idx]
  y_pred_part <- vapply(idx, function(i) {
    if (!eval_mask[i] || !pred_is_valid[i]) return(NA_character_)
    if (gold_is_morph[i]) get_morph_parent(pred_top1_norm[i]) else get_topo_parent(pred_top1_norm[i])
  }, character(1))
  f1p <- f1_macro_weighted(y_true_part, y_pred_part)
  
  # Top-k (exact + parent)
  topk_exact <- sum(topk_exact_hit[idx] & eval_mask[idx], na.rm = TRUE)
  topk_parent <- sum(topk_parent_hit[idx] & eval_mask[idx], na.rm = TRUE)
  topk_exact_rate  <- if (n_eval == 0) 0 else topk_exact / n_eval
  topk_parent_rate <- if (n_eval == 0) 0 else topk_parent / n_eval
  
  # MRR
  mrr_mean <- mean(mrr_exact[idx][eval_mask[idx]], na.rm = TRUE)
  if (is.nan(mrr_mean)) mrr_mean <- 0
  
  tibble(
    Epoch = unique(df[[epoch_col]][idx])[1],
    `Num Questions` = n_eval,
    `Invalid Answer` = invalid,
    `Exact Match` = exact,
    `Partial Match (parent)` = partial,
    `False Code` = f_code,
    `False Bool` = f_bool,
    Accuracy = round(acc, 4),
    `Partial Accuracy` = round(partial_acc, 4),
    `Top-k Exact Rate` = round(topk_exact_rate, 4),
    `Top-k Parent Rate` = round(topk_parent_rate, 4),
    `Invalid Rate` = round(invalid_rate, 4),
    `Error Rate` = round(error_rate, 4),
    `Average Answer Length` = round(avg_len, 2),
    `MRR (exact)` = round(mrr_mean, 4),
    `F1 Exact (macro)` = round(f1e$macro, 4),
    `F1 Exact (weighted)` = round(f1e$weighted, 4),
    `F1 Parent (macro)` = round(f1p$macro, 4),
    `F1 Parent (weighted)` = round(f1p$weighted, 4)
  )
}

# -------------------------
# Zusammenfassung je Epoch + Gesamt
# -------------------------
group_index <- split(seq_len(nrow(df)), df[[epoch_col]])
summary_by_epoch <- bind_rows(lapply(group_index, compute_summary)) %>%
  arrange(suppressWarnings(as.numeric(Epoch)))

summary_all <- compute_summary(seq_len(nrow(df))) %>% mutate(Epoch = "all")
summary_tbl <- bind_rows(summary_by_epoch, summary_all)

# -------------------------
# Detailtabelle (pro Zeile)
# -------------------------
eval_per_row <- df %>%
  mutate(
    gold_icdo_raw         = gold_raw,
    gold_icdo_norm        = gold_norm,
    gold_is_morph         = gold_is_morph,
    gold_is_topo          = gold_is_topo,
    gold_parent           = gold_parent,
    pred_top1_raw         = pred_top1_raw,
    pred_top1_norm        = pred_top1_norm,
    pred_is_valid_icdo    = pred_is_valid,
    pred_is_bool          = pred_is_bool,
    pred_is_invalid       = pred_is_invalid,
    pred_is_morph         = pred_is_morph,
    pred_is_topo          = pred_is_topo,
    pred_parent           = pred_parent,
    exact_match           = exact_match,
    partial_match_parent  = partial_match,
    false_code            = false_code,
    false_bool            = false_bool,
    invalid_answer        = invalid_answer,
    topk_exact_hit        = topk_exact_hit,
    topk_parent_hit       = topk_parent_hit,
    rank_exact            = rank_exact,
    mrr_exact             = mrr_exact,
    answer_length_chars   = nchar(pred_top1_raw)
  )

# -------------------------
# Speichern
# -------------------------
out_dir <- dirname(input_csv)
readr::write_delim(eval_per_row, file.path(out_dir, "icdo_eval_per_row_2.csv"), delim = ";")
readr::write_delim(summary_tbl,  file.path(out_dir, "icdo_eval_summary_2.csv"),  delim = ";")

print(summary_tbl)
cat("Fertig. Dateien gespeichert in:\n  -", file.path(out_dir, "icdo_eval_per_row_2.csv"),
    "\n  -", file.path(out_dir, "icdo_eval_summary_2.csv"), "\n")
