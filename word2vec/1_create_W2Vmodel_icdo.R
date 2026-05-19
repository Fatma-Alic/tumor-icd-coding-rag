#library(wordVectors)# word2vec-modelle
#library("devtools") #install_github("bmschmidt/wordVectors")
library(stringr) 
library(readr)
library(dplyr) #Datenmanipulation 
library(purrr) # Mapping der Codes 
library(tibble) # tibbles
library(stringdist) 

if (!requireNamespace("wordVectors", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
  devtools::install_github("bmschmidt/wordVectors")
}
library(wordVectors)

# Pfade
BASE <- "C:/Users/ali0f/Documents/word2vec"
PATH_ALPHA <- file.path(BASE, "knowledgebase_icdo_codes_ids.csv")
QUERY_FILE <- file.path(BASE, "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv")
MODEL_BIN  <- file.path(BASE, "MODEL_icdo.bin")
CORPUS_TXT <- file.path(BASE, "icdo_corpus.txt")
# Speichern der Ergebnisse im Ordner "outputs_w2v
OUT_DIR       <- file.path(BASE, "outputs_w2v_icdo")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
#Speichern der Zwischenergebnisse 
INDEX_CSV         <- file.path(OUT_DIR, "icd_index.csv")                # Mapping Alpha-ID, Label_norm, Label, ICD-O-Code
TOKEN_FREQ_CSV    <- file.path(OUT_DIR, "token_frequencies.csv")        # Häufigkeiten aus dem Korpus
AID_VECS_CSV      <- file.path(OUT_DIR, "aid_vectors.csv")              # Embeddings (Alpha-ID)
QRY_EMB_CSV       <- file.path(OUT_DIR, "query_embeddings.csv")         # Query-Embeddings (Mittelwert)
NN_RESULTS_CSV    <- file.path(OUT_DIR, "nn_results_topk_icdo.csv")          # Ergebnisdatei: Top-k Nachbarn je Query

# Hauptparameter   
var_nr        <- "var1"
top_k         <- 20
vectors       <- 100
window        <- 10
min_count     <- 1
iter          <- 30
threads       <- parallel::detectCores()

# ============================================================
# 1) Helpers: Normalization, cosine, etc.
# ============================================================

normalize_text <- function(x) {
  x <- ifelse(is.na(x), "", as.character(x))
  x |>
    # (Uppercase umlaut replacements are redundant after tolower, but harmless)
    str_replace_all("Ä","Ae") |> str_replace_all("Ö","Oe") |> str_replace_all("Ü","Ue") |>
    str_replace_all("ä","ae") |> str_replace_all("ö","oe") |> str_replace_all("ü","ue") |>
    str_replace_all("ß","ss") |>
    tolower() |>
    str_replace_all("-", " ") |>
    str_replace_all("[^a-z0-9 ]+", " ") |>
    str_replace_all("\\s+", " ") |>
    str_trim()
}

clean_alpha_id <- function(aid) paste0("aid_", as.character(aid))

cosine_sim <- function(A, B) {
  # A: (n x d), B: (m x d) -> returns (n x m)
  An <- sqrt(rowSums(A * A)); An[An == 0] <- 1
  Bn <- sqrt(rowSums(B * B)); Bn[Bn == 0] <- 1
  A_norm <- A / An
  B_norm <- B / Bn
  A_norm %*% t(B_norm)
}

# Fuzzy helper (uses global "vocab" when called)
fuzzy_in_vocab <- function(tok, max_cand = 20, jw_thresh = 0.15) {
  d <- stringdist(tok, vocab, method = "jw")  # 0=gleich, 1=unähnlich (distance)
  keep <- order(d)
  sel  <- vocab[keep][d[keep] <= jw_thresh]
  head(sel, max_cand)
}

# ============================================================
# 2) Load knowledge base (Extended Label everywhere)
# ============================================================

message("Lade Knowledge Base: ", PATH_ALPHA)
kb <- read_delim(
  PATH_ALPHA,
  delim = ";",
  col_names = TRUE,
  col_types = cols(.default = "c"),
  locale = locale(encoding = "UTF-8")
)

print(names(kb))

# Required columns
stopifnot(all(c("ID", "Extended Label", "ICD-O-Code") %in% names(kb)))

# Build fields (Extended Label used everywhere)
kb <- kb %>%
  mutate(
    AlphaID_token = clean_alpha_id(`ID`),
    Label_raw     = `Extended Label`,                  # <-- Extended Label is the source of truth
    Label_norm    = normalize_text(Label_raw),
    # Optional prefix for training text only:
    Label         = if (identical(var_nr, "var1")) paste0("Tumordiagnose: ", Label_raw) else Label_raw
  )

# Index table (stores raw Extended Label + ICD-O)
index_tbl <- kb %>%
  transmute(
    `ID`          = as.character(`ID`),
    AlphaID_token = AlphaID_token,
    Label_norm    = Label_norm,
    Label         = Label_raw,          # <-- keep output label = Extended Label (raw)
    `ICD-O-Code`  = `ICD-O-Code`
  )

write_delim(index_tbl, INDEX_CSV, delim = ";")

# Corpus lines: "aid_<ID> <Label_norm> <ICD-O-Code>"
corpus_lines <- paste(kb$AlphaID_token, kb$Label_norm, kb$`ICD-O-Code`)
corpus_lines <- corpus_lines[nchar(corpus_lines) > nchar("aid_") + 1]
write_lines(corpus_lines, CORPUS_TXT)

message("Die ersten 5 Zeilen im Korpus:\n", paste(head(corpus_lines, 5), collapse = "\n"))

# Token frequencies
tokens <- str_split(corpus_lines, "\\s+", simplify = FALSE)
freqs  <- sort(table(unlist(tokens)), decreasing = TRUE)
freq_tbl <- tibble(token = names(freqs), freq = as.integer(freqs))
write_delim(freq_tbl, TOKEN_FREQ_CSV, delim = ";")

# ============================================================
# 3) Train Word2Vec
# ============================================================

if (file.exists(MODEL_BIN)) unlink(MODEL_BIN, force = TRUE)

message("Trainiere Word2Vec …")
model <- train_word2vec(
  train_file = CORPUS_TXT,
  output_file = MODEL_BIN,
  vectors = vectors,
  window  = window,
  min_count = min_count,
  iter = iter,
  threads = threads,
  cbow = 0,                 # 0 = skip-gram
  negative_samples = 10,
  force = TRUE
)

rm(model); gc()
message("Training fertig: ", MODEL_BIN)

# ============================================================
# 4) Load model & export AID vectors
# ============================================================

message("Lade Modell & extrahiere AID-Vektoren …")
model <- read.vectors(MODEL_BIN)

all_tokens <- rownames(model)
aid_mask   <- startsWith(all_tokens, "aid_")
aid_tokens <- all_tokens[aid_mask]

vocab <- rownames(model)   # global used by fuzzy_in_vocab()

message("Anzahl aid_tokens: ", length(aid_tokens))
message("Erste aid_tokens: ", paste(head(aid_tokens, 10), collapse = "; "))

# Extract embedding matrix for AIDs
aid_vecs <- model[aid_tokens, , drop = FALSE]
stopifnot(is.matrix(aid_vecs))
storage.mode(aid_vecs) <- "double"
colnames(aid_vecs) <- paste0("V", seq_len(ncol(aid_vecs)))

# Build AID table with metadata (Extended Label + ICD-O)
aid_tbl <- tibble(
  AlphaID_token = aid_tokens,
  `ID` = sub("^aid_", "", aid_tokens, ignore.case = TRUE)
) %>%
  bind_cols(as_tibble(aid_vecs, .name_repair = "minimal")) %>%
  mutate(`ID` = as.character(`ID`))

index_one <- as_tibble(index_tbl) %>%
  mutate(`ID` = as.character(`ID`)) %>%
  distinct(`ID`, .keep_all = TRUE)

idx <- match(aid_tbl$`ID`, index_one$`ID`)

aid_tbl$Label      <- index_one$Label[idx]       # <-- Extended Label (raw)
aid_tbl$Label_norm <- index_one$Label_norm[idx]
aid_tbl$`ICD-O-Code` <- index_one$`ICD-O-Code`[idx]

# Safety checks
stopifnot(all(!is.na(aid_tbl$Label)))
stopifnot(all(!is.na(aid_tbl$`ICD-O-Code`)))

write_delim(aid_tbl, AID_VECS_CSV, delim = ";")
read_delim(AID_VECS_CSV, delim = ";", n_max = 5, show_col_types = FALSE)

# ============================================================
# 5) Load queries & build query embeddings (SIF + remove 1st PC)
# ============================================================

message("Lade Query-Daten: ", QUERY_FILE)
qry <- read_delim(QUERY_FILE, delim = ";", col_types = cols(.default = "c"))

stopifnot(any(c("Text extended", "Text_extended", "Text") %in% names(qry)))
text_col <- if ("Text extended" %in% names(qry)) "Text extended" else if ("Text_extended" %in% names(qry)) "Text_extended" else "Text"

normalize_tokens <- function(s) {
  normalize_text(s) |>
    str_split("\\s+", simplify = TRUE) |>
    as.vector()
}

# --- SIF weights ---
a <- 1e-3
N <- sum(freq_tbl$freq)
w <- setNames(a / (a + (freq_tbl$freq / N)), freq_tbl$token)

embed_sentence_sif <- function(txt) {
  toks <- normalize_tokens(txt)
  toks <- toks[nzchar(toks)]
  if (!length(toks)) return(rep(0, ncol(model)))
  
  in_vocab <- toks[toks %in% vocab]
  if (!length(in_vocab)) {
    cands <- unique(unlist(lapply(toks, fuzzy_in_vocab), use.names = FALSE))
    if (!length(cands)) return(rep(0, ncol(model)))
    M <- model[cands, , drop = FALSE]
    return(colMeans(M))
  }
  
  ww <- w[in_vocab]; ww[is.na(ww)] <- a
  M  <- model[in_vocab, , drop = FALSE]
  as.numeric(colSums(M * ww) / sum(ww))
}

message("Erzeuge Query-Embeddings (SIF, 1. PC entfernen) …")
qry_emb_mat <- do.call(rbind, lapply(qry[[text_col]], embed_sentence_sif))
qry_emb_mat <- as.matrix(qry_emb_mat)
storage.mode(qry_emb_mat) <- "double"

# Remove first principal component (Arora et al.)
if (nrow(qry_emb_mat) > 1 && ncol(qry_emb_mat) > 1) {
  pc <- prcomp(qry_emb_mat, center = FALSE, scale. = FALSE)
  if (!is.null(pc$rotation) && ncol(pc$rotation) >= 1) {
    u1 <- pc$rotation[, 1, drop = FALSE]
    qry_emb_mat <- qry_emb_mat - (qry_emb_mat %*% u1) %*% t(u1)
  }
}

qry_emb_tbl <- as_tibble(qry_emb_mat)
colnames(qry_emb_tbl) <- paste0("V", seq_len(ncol(qry_emb_tbl)))
qry_to_save <- bind_cols(qry, qry_emb_tbl)

write_delim(qry_to_save, QRY_EMB_CSV, delim = ";")
message("Query-Embeddings gespeichert: ", QRY_EMB_CSV)

stopifnot(is.matrix(qry_emb_mat))
stopifnot(nrow(qry_emb_mat) == nrow(qry))   # Query rows are preserved here already

# ============================================================
# 6) Nearest Neighbors: Query vs AID vectors (Top-k)
# ============================================================

message("Berechne Top-", top_k, " Nachbarn je Query …")

A <- qry_emb_mat
B <- as.matrix(aid_vecs)     # n_aid x d

S <- cosine_sim(A, B)        # n_query x n_aid
S <- as.matrix(S)

n_q   <- nrow(S)
n_aid <- ncol(S)
k_eff <- min(top_k, n_aid)
if (k_eff == 0) stop("Keine Kandidaten (n_aid=0).")

get_topk_idx <- function(row, k) {
  fin <- which(is.finite(row))
  if (!length(fin)) return(rep(NA_integer_, k))
  o <- fin[order(row[fin], decreasing = TRUE)]
  out <- head(o, k); length(out) <- k
  out
}

# IMPORTANT FIX: allocate ALWAYS top_k columns, fill only up to k_eff
topk_idx <- matrix(NA_integer_, nrow = n_q, ncol = top_k)
topk_sim <- matrix(NA_real_,    nrow = n_q, ncol = top_k)

tmp_idx <- t(vapply(seq_len(n_q), function(i) get_topk_idx(S[i, ], k_eff),
                    FUN.VALUE = integer(k_eff)))
topk_idx[, seq_len(k_eff)] <- tmp_idx

for (i in seq_len(n_q)) {
  idx <- topk_idx[i, seq_len(k_eff)]
  sel <- !is.na(idx)
  if (any(sel)) topk_sim[i, which(sel)] <- S[i, idx[sel]]
}

# Lookups (Extended Label as suggestedDocuments!)
lab_by_token <- setNames(aid_tbl$Label,        aid_tbl$AlphaID_token)   # = Extended Label
icd_by_token <- setNames(aid_tbl$`ICD-O-Code`, aid_tbl$AlphaID_token)

results_tbl <- qry %>% mutate(`__rowid` = row_number())

# --- Optional Hybrid Re-Ranking (Cosine + JW on Extended Label) ---
lambda <- 0.7

for (i in seq_len(n_q)) {
  idx <- topk_idx[i, seq_len(k_eff)]
  sel <- !is.na(idx)
  if (!any(sel)) next
  
  cand_idx    <- idx[sel]
  cand_tokens <- aid_tokens[cand_idx]
  cand_labels <- unname(lab_by_token[cand_tokens])   # Extended Label (raw)
  cos_i       <- S[i, cand_idx]
  
  qtxt_norm <- normalize_text(qry[[text_col]][i])
  lab_norm  <- normalize_text(ifelse(is.na(cand_labels), "", cand_labels))
  
  jw <- stringdist::stringsim(qtxt_norm, lab_norm, method = "jw")  # 0..1 similarity
  
  hybrid <- lambda * cos_i + (1 - lambda) * jw
  ord    <- order(hybrid, decreasing = TRUE)
  
  topk_idx[i, which(sel)] <- cand_idx[ord]
  topk_sim[i, which(sel)] <- cos_i[ord]
}

# Diagnostics
message("Zum Überprüfen der Matrix")
cat("Queries (n_q) =", nrow(A), "  AlphaIDs (n_aid) =", nrow(B), "\n")
cat("Any NA/NaN/Inf in S? ", any(!is.finite(S)), "\n")
cat("Zero-vector queries: ", sum(rowSums(A^2) == 0), "\n")

# Build output columns (SAFE: loop always up to top_k now)
for (k in seq_len(top_k)) {
  idx_k <- topk_idx[, k]
  sim_k <- topk_sim[, k]
  
  neighbor_tokens <- ifelse(is.na(idx_k), NA_character_, aid_tokens[idx_k])
  neighbor_ids    <- sub("^aid_", "", neighbor_tokens, ignore.case = TRUE)
  
  neighbor_df <- tibble(
    `__rowid` = seq_len(n_q),
    !!paste0("suggestedIDs", k)         := neighbor_ids,
    !!paste0("suggestedDocuments", k)   := unname(lab_by_token[neighbor_tokens]),  # Extended Label
    !!paste0("suggestedMetadata", k)    := unname(icd_by_token[neighbor_tokens]),  # ICD-O-Code
    !!paste0("distances", k)            := 1 - sim_k
  )
  
  # LEFT JOIN keeps ALL query rows
  results_tbl <- left_join(results_tbl, neighbor_df, by = "__rowid")
}

results_tbl <- results_tbl %>% select(-`__rowid`)

# Final: guarantee no query row was dropped
stopifnot(nrow(results_tbl) == nrow(qry))

write_delim(results_tbl, NN_RESULTS_CSV, delim = ";")
message("Fertig ✅  Ergebnisse: ", NN_RESULTS_CSV)


