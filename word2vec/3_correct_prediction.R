# Pakete
library(readr)
library(dplyr)
library(stringr)
library(tidyr)
library(purrr)

# ==== Eingabe anpassen ====
input_csv <- "C:/Users/ali0f/Documents/word2vec/outputs_w2v/nn_results_topk_icdo.csv"
# ==========================

# Hilfsfunktionen zur Normalisierung von Codes
normalize_code <- function(x) str_trim(toupper(as.character(x)))

# CSV lesen
df <- readr::read_delim(input_csv, delim = ";", col_types = cols(.default = "c"))

# Pflichtspalte prüfen
stopifnot("ICD-10-Code" %in% names(df))

# Suggested-Metadata-Spalten finden (eine oder mehrere)
sm_cols <- grep("^suggestedMetadata", names(df), value = TRUE)
if (length(sm_cols) == 0 && "suggestedMetadata" %in% names(df)) sm_cols <- "suggestedMetadata"
if (length(sm_cols) == 0) stop("Keine Spalten gefunden, die mit 'suggestedMetadata' beginnen.")

# Liste der Vorschläge pro Zeile
sugg_list <- df[sm_cols] %>%
  as.data.frame() %>%
  asplit(1) %>%                               # Liste: jede Zeile -> ein Vektor
  lapply(\(v) {
    v <- as.character(unlist(v, use.names = FALSE))
    v[!is.na(v) & nzchar(v)]
  })

# Gold-Labels (wahrer ICD-10-Code)
gold      <- df$`ICD-10-Code`
gold_norm <- normalize_code(gold)

# Normalisierte Vorschläge (exakt)
sugg_norm_list <- lapply(sugg_list, normalize_code)

# ---- Parent-/Oberklassen-Helfer (3-stellig: Buchstabe + 2 Ziffern) ----
get_cat3 <- function(x) {
  m <- stringr::str_match(normalize_code(x), "^([A-Z]\\d{2})")
  m[, 2]
}

gold_cat3 <- get_cat3(gold_norm)

# Top-1 / Top-k Treffer (exakt)
top1_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA)
  # Top-1 = erste nicht-NA-Spalte
  identical(sugg[[1]], g)
}, sugg_norm_list, gold_norm)

topk_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA)
  any(sugg == g)
}, sugg_norm_list, gold_norm)

# Top-k Treffer auf Oberklassen-Ebene (3-stellig, z. B. "C34")
topk_parent_cat3 <- mapply(function(sugg, g3) {
  if (length(sugg) == 0 || is.na(g3) || !nzchar(g3)) return(NA)
  any(get_cat3(sugg) == g3)
}, sugg_norm_list, gold_cat3)

# Rank / MRR (exakt)
rank_exact <- mapply(function(sugg, g) {
  if (length(sugg) == 0 || is.na(g) || !nzchar(g)) return(NA_integer_)
  m <- match(g, sugg)
  if (is.na(m)) NA_integer_ else as.integer(m)
}, sugg_norm_list, gold_norm)

mrr_exact <- ifelse(is.na(rank_exact), 0, 1 / rank_exact)

# Zusammenfassung
n_all  <- nrow(df)
n_eval <- sum(!is.na(topk_exact))  # Zeilen, die bewertet werden konnten

summary_tbl <- tibble(
  rows_total              = n_all,
  rows_evaluated          = n_eval,
  top1_exact_hits         = sum(top1_exact, na.rm = TRUE),
  top1_exact_rate         = round(mean(top1_exact, na.rm = TRUE) * 100, 2),
  topk_exact_hits         = sum(topk_exact, na.rm = TRUE),
  topk_exact_rate         = round(mean(topk_exact, na.rm = TRUE) * 100, 2),
  topk_parent_cat3_hits   = sum(topk_parent_cat3, na.rm = TRUE),
  topk_parent_cat3_rate   = round(mean(topk_parent_cat3, na.rm = TRUE) * 100, 2),
  mrr_exact_mean          = round(mean(mrr_exact, na.rm = TRUE), 4)
)

print(summary_tbl)

# Detailtabelle (pro Zeile)
eval_per_row <- df %>%
  mutate(
    gold_icd          = gold,
    gold_icd_norm     = gold_norm,
    gold_cat3         = gold_cat3,
    top1_exact        = top1_exact,
    topk_exact        = topk_exact,
    topk_parent_cat3  = topk_parent_cat3,
    rank_exact        = rank_exact,
    mrr_exact         = mrr_exact
  )

# Optional: speichern
out_dir <- dirname(input_csv)
readr::write_delim(eval_per_row, file.path(out_dir, "icd_eval_per_row.csv"), delim = ";")
readr::write_delim(summary_tbl,   file.path(out_dir, "icd_eval_summary.csv"), delim = ";")

cat("Fertig. Dateien gespeichert in:\n  -", file.path(out_dir, "icd_eval_per_row.csv"),
    "\n  -", file.path(out_dir, "icd_eval_summary.csv"), "\n")
