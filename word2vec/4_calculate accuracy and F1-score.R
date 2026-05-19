# Pakete
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)

# ==== Eingabe anpassen ====
input_csv <- "C:/Users/ali0f/Documents/word2vec/outputs_w2v/nn_results_topk.csv"
# ==========================

# Hilfsfunktionen
normalize_code <- function(x) str_trim(toupper(as.character(x)))
get_cat3 <- function(x) {
  m <- stringr::str_match(normalize_code(x), "^([A-Z]\\d{2})")
  m[, 2]
}
is_bool_only <- function(x) {
  x <- tolower(str_trim(as.character(x)))
  !is.na(x) & nzchar(x) & x %in% c("true","false","yes","no")
}
is_valid_icd <- function(x) {
  x <- normalize_code(x)
  !is.na(x) & str_detect(x, "^[A-Z][0-9]{2}(\\.[A-Z0-9]{1,4})?$")
}

# F1-Helfer (macro & weighted) für Klassenvorhersagen
f1_macro_weighted <- function(y_true, y_pred) {
  # y_true/y_pred: Vektoren gleicher Länge, Klassenstrings; NAs in y_pred gelten als "keine Vorhersage"
  # Nur Klassen berücksichtigen, die in y_true vorkommen
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

# CSV lesen
df <- readr::read_delim(input_csv, delim = ";", col_types = cols(.default = "c"))

# Pflichtspalte prüfen
stopifnot("ICD-10-Code" %in% names(df))

# Evtl. Epochen-Spalte identifizieren
epoch_col <- c("Epoch","epoch","EPOCH")
epoch_col <- epoch_col[epoch_col %in% names(df)]
if (length(epoch_col) == 0) {
  df$Epoch <- "0"
  epoch_col <- "Epoch"
}

# Suggested-Metadata-Spalten finden (eine oder mehrere)
sm_cols <- grep("^suggestedMetadata", names(df), value = TRUE)
if (length(sm_cols) == 0 && "suggestedMetadata" %in% names(df)) sm_cols <- "suggestedMetadata"
if (length(sm_cols) == 0) stop("Keine Spalten gefunden, die mit 'suggestedMetadata' beginnen.")

# Liste der Vorschläge pro Zeile
sugg_list <- df[sm_cols] %>%
  as.data.frame() %>%
  asplit(1) %>%
  lapply(\(v) {
    v <- as.character(unlist(v, use.names = FALSE))
    v[!is.na(v) & nzchar(v)]
  })

# Gold-Labels
gold_raw  <- df$`ICD-10-Code`
gold_norm <- normalize_code(gold_raw)
gold_cat3 <- get_cat3(gold_norm)

# Top-1 Rohvorschlag & normalisiert
pred_top1_raw <- vapply(sugg_list, function(v) if (length(v) == 0) NA_character_ else v[[1]], character(1))
pred_top1_norm <- normalize_code(pred_top1_raw)
pred_top1_cat3 <- get_cat3(pred_top1_norm)

# Validität/Typen
pred_is_bool   <- is_bool_only(pred_top1_raw)
pred_is_valid  <- is_valid_icd(pred_top1_raw)
pred_is_invalid <- !(pred_is_valid | pred_is_bool) # weder gültiger Code noch bool

# Matching-Flags
exact_match   <- !is.na(gold_norm) & !is.na(pred_top1_norm) & pred_is_valid & (pred_top1_norm == gold_norm)
partial_match <- !exact_match & pred_is_valid & !is.na(gold_cat3) & !is.na(pred_top1_cat3) & (pred_top1_cat3 == gold_cat3)

false_code <- pred_is_valid & !exact_match & !partial_match
false_bool <- pred_is_bool
invalid_answer <- pred_is_invalid

# Evaluierbare Zeilen (Gold vorhanden)
eval_mask <- !is.na(gold_norm) & nzchar(gold_norm)

# Metriken (gesamt & je Epoch)
compute_summary <- function(idx) {
  n_total <- length(idx)
  n_eval  <- sum(eval_mask[idx])
  
  # Zähler (nur über eval_mask zählen)
  exact    <- sum(exact_match[idx] & eval_mask[idx], na.rm = TRUE)
  partial  <- sum(partial_match[idx] & eval_mask[idx], na.rm = TRUE)
  f_code   <- sum(false_code[idx]   & eval_mask[idx], na.rm = TRUE)
  f_bool   <- sum(false_bool[idx]   & eval_mask[idx], na.rm = TRUE)
  invalid  <- sum(invalid_answer[idx] & eval_mask[idx], na.rm = TRUE)
  
  # Raten
  acc            <- if (n_eval == 0) 0 else exact / n_eval
  partial_acc    <- if (n_eval == 0) 0 else (exact + partial) / n_eval
  invalid_rate   <- if (n_eval == 0) 0 else invalid / n_eval
  error_rate     <- if (n_eval == 0) 0 else (f_code + f_bool + invalid) / n_eval
  
  # Average Answer Length (Zeichen des Roh-Top1, nur eval)
  avg_len <- mean(nchar(pred_top1_raw[idx][eval_mask[idx] & !is.na(pred_top1_raw[idx])]), na.rm = TRUE)
  if (is.nan(avg_len)) avg_len <- 0
  
  # F1 Exact (Klassen = exakte ICD-Codes), Prädikt = NA für ungültige/bool
  y_true_exact <- gold_norm[idx]
  y_pred_exact <- ifelse(pred_is_valid[idx], pred_top1_norm[idx], NA_character_)
  f1e <- f1_macro_weighted(y_true_exact, y_pred_exact)
  
  # F1 Partial (Klassen = Cat3), Prädikt = NA für ungültige/bool
  y_true_part <- gold_cat3[idx]
  y_pred_part <- ifelse(pred_is_valid[idx], pred_top1_cat3[idx], NA_character_)
  f1p <- f1_macro_weighted(y_true_part, y_pred_part)
  
  tibble(
    Epoch = unique(df[[epoch_col]][idx])[1],
    `Num Questions` = n_eval,
    `Invalid Answer` = invalid,
    `Exact Match` = exact,
    `Partial Match Code` = partial,
    `False Code` = f_code,
    `False Bool` = f_bool,
    Accuracy = round(acc, 4),
    `Partial Accuracy` = round(partial_acc, 4),
    `Invalid Rate` = round(invalid_rate, 4),
    `Error Rate` = round(error_rate, 4),
    `Average Answer Length` = round(avg_len, 2),
    `F1 Exact (macro)` = round(f1e$macro, 4),
    `F1 Exact (weighted)` = round(f1e$weighted, 4),
    `F1 Partial (macro)` = round(f1p$macro, 4),
    `F1 Partial (weighted)` = round(f1p$weighted, 4)
  )
}

# Zusammenfassung je Epoch
group_index <- split(seq_len(nrow(df)), df[[epoch_col]])
summary_by_epoch <- bind_rows(lapply(group_index, compute_summary)) %>%
  arrange(as.numeric(Epoch))

# Gesamtsummary (Epoch = "all")
summary_all <- compute_summary(seq_len(nrow(df))) %>% mutate(Epoch = "all")

summary_tbl <- bind_rows(summary_by_epoch, summary_all)

# Detailtabelle (pro Zeile)
eval_per_row <- df %>%
  mutate(
    gold_icd              = gold_raw,
    gold_icd_norm         = gold_norm,
    gold_cat3             = gold_cat3,
    pred_top1_raw         = pred_top1_raw,
    pred_top1_norm        = pred_top1_norm,
    pred_top1_cat3        = pred_top1_cat3,
    pred_is_valid         = pred_is_valid,
    pred_is_bool          = pred_is_bool,
    pred_is_invalid       = pred_is_invalid,
    exact_match           = exact_match,
    partial_match         = partial_match,
    false_code            = false_code,
    false_bool            = false_bool,
    invalid_answer        = invalid_answer,
    answer_length_chars   = nchar(pred_top1_raw)
  )

# Speichern
out_dir <- dirname(input_csv)
readr::write_delim(eval_per_row, file.path(out_dir, "icd_eval_per_row_2.csv"), delim = ";")
readr::write_delim(summary_tbl,   file.path(out_dir, "icd_eval_summary_2.csv"), delim = ";")

print(summary_tbl)
cat("Fertig. Dateien gespeichert in:\n  -", file.path(out_dir, "icd_eval_per_row_2.csv"),
    "\n  -", file.path(out_dir, "icd_eval_summary_2.csv"), "\n")
