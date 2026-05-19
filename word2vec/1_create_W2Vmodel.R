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
PATH_ALPHA <- file.path(BASE, "knowledgebase_icd10_codes_2024.csv")
QUERY_FILE <- file.path(BASE, "fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv")
MODEL_BIN  <- file.path(BASE, "MODEL.bin")
CORPUS_TXT <- file.path(BASE, "icd_corpus.txt")
# Speichern der Ergebnisse im Ordner "outputs_w2v
OUT_DIR       <- file.path(BASE, "outputs_w2v")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
#Speichern der Zwischenergebnisse 
INDEX_CSV         <- file.path(OUT_DIR, "icd_index.csv")                # Mapping Alpha-ID, Label_norm, Label, ICD-10-Code, ICD-O-Code
TOKEN_FREQ_CSV    <- file.path(OUT_DIR, "token_frequencies.csv")        # Häufigkeiten aus dem Korpus
AID_VECS_CSV      <- file.path(OUT_DIR, "aid_vectors.csv")              # Embeddings (Alpha-ID)
QRY_EMB_CSV       <- file.path(OUT_DIR, "query_embeddings.csv")         # Query-Embeddings (Mittelwert)
NN_RESULTS_CSV    <- file.path(OUT_DIR, "nn_results_topk.csv")          # Ergebnisdatei: Top-k Nachbarn je Query

# Hauptparameter   
var_nr        <- "var1"
top_k         <- 20
vectors       <- 100
window        <- 10
min_count     <- 1
iter          <- 30
threads       <- parallel::detectCores()

# 1) Datensätze einlesen, normalisieren 
# Normalisierungsfunktion
normalize_text <- function(x) {
  x |>
    tolower() |>
    str_replace_all("Ä","Ae") |> str_replace_all("Ö","Oe") |> str_replace_all("Ü","Ue") |>
    str_replace_all("ä","ae") |> str_replace_all("ö","oe") |> str_replace_all("ü","ue") |>
    str_replace_all("ß","ss") |> str_replace_all("-", " ") |>
    str_replace_all("[^a-z0-9 ]+", " ") |>
    str_replace_all("\\s+", " ") |>
    str_trim()
}

# aid_ vor der Alpha-ID 
clean_alpha_id <- function(aid) paste0("aid_", as.character(aid))

# Tumordiagnose vor der Diagnose einfügen
prepare_bezeichnung_list <- function(df, var_nr) {
  bz <- df$Label |> as.character()
  if (identical(var_nr, "var1")) {
    bz <- paste0("Tumordiagnose: ", bz)
  }
  bz
}

# Cosine-Similarity: Matrix x Matrix
cosine_sim <- function(A, B) {
  # normalisieren
  An <- sqrt(rowSums(A*A)); An[An == 0] <- 1
  Bn <- sqrt(rowSums(B*B)); Bn[Bn == 0] <- 1
  A_norm <- A / An
  B_norm <- B / Bn
  A_norm %*% t(B_norm)
}

# Einlesen der Alpha-Datei 
message("Lade Knowledge Base: ", PATH_ALPHA)
kb <- read_delim(PATH_ALPHA, delim = ";", col_names = TRUE, col_types = cols(.default = "c"), locale = locale(encoding = "UTF-8"))

if ("Tumordiagnose" %in% names(kb)) {
  kb <- kb %>% filter(Tumordiagnose == "1")
}

stopifnot(all(c("Alpha-ID","Label") %in% names(kb)))
has_icd10 <- "ICD-10-Code" %in% names(kb)

# Prüfe Spaltennamen:
print(names(kb))

# 2) Bezeichnungen + Index bauen & speichern
# =========================

kb <- kb %>%
  mutate(
    AlphaID_token = clean_alpha_id(`Alpha-ID`),
    Label_norm    = normalize_text(Label),
    Label         = if (identical(var_nr, "var1")) paste0("Tumordiagnose: ", Label) else Label
  )

index_tbl <- kb %>%
  transmute(
    `Alpha-ID`     = as.character(`Alpha-ID`),
    AlphaID_token  = AlphaID_token,
    Label_norm     = Label_norm,
    Label          = Label,
    `ICD-10-Code`  = if (has_icd10) `ICD-10-Code` else NA_character_
  )

write_delim(index_tbl, INDEX_CSV, delim = ";")

# Korpus: "aid_<ID> <Label_norm>"
corpus_lines <- paste(kb$AlphaID_token, normalize_text(kb$Label), kb$`ICD-10-Code`)
corpus_lines <- corpus_lines[nchar(corpus_lines) > nchar("aid_") + 1]
write_lines(corpus_lines, CORPUS_TXT)
message("Die ersten 5 Zeilen: ", head(corpus_lines, 5))   

# Token-Frequenzen
tokens  <- str_split(corpus_lines, "\\s+", simplify = FALSE)
freqs   <- table(unlist(tokens)) |> sort(decreasing = TRUE)
freq_tbl <- tibble(token = names(freqs), freq = as.integer(freqs))
write_delim(freq_tbl, TOKEN_FREQ_CSV, delim = ";")

# 3) Modell trainieren 
# =========================
fuzzy_in_vocab <- function(tok, max_cand = 20, jw_thresh = 0.15) {
  d <- stringdist(tok, vocab, method = "jw")  # 0=gleich, 1=unähnlich
  keep <- order(d)
  sel  <- vocab[keep][d[keep] <= jw_thresh]
  head(sel, max_cand)
}

if (file.exists(MODEL_BIN)) unlink(MODEL_BIN, force = TRUE) #zum Löschen des vorherigen Models 
message("Trainiere Word2Vec …")
model <- train_word2vec(
    CORPUS_TXT,
    output_file = MODEL_BIN,
    vectors = vectors,
    window  = window,
    min_count = min_count,
    iter = iter,
    threads = threads,
    cbow = 0, #(Continuous Bag of Words) cbow = 0 -> skipgram; skipgram: Zielwort -> Kontext; cbow: Kontext -> Zielwort 
    negative_samples = 10,
    force = TRUE
)
rm(model); gc()
message("Training fertig: ", MODEL_BIN)

# 4) Modell einlesen & Alpha-ID-Vektoren exportieren
# =========================
message("Lade Modell & extrahiere AID-Vektoren …")
model <- read.vectors(MODEL_BIN)
all_tokens <- rownames(model)
aid_mask   <- startsWith(all_tokens, "aid_")
aid_tokens <- all_tokens[aid_mask]
vocab <- rownames(model)
aid_mask   <- startsWith(all_tokens, "aid_")
message("Die 10 ersten aid_mask: ", paste(head(aid_mask, 10), collapse = "; "))

aid_tokens <- all_tokens[aid_mask]
message("Die aid_tokens:", paste(head(aid_tokens, 10), collapse =';'))

# Reine Embedding-Matrix holen (n_aid x d) und SAUBER benennen
aid_vecs <- model[aid_tokens, , drop = FALSE]
stopifnot(is.matrix(aid_vecs))
storage.mode(aid_vecs) <- "double"
colnames(aid_vecs) <- paste0("V", seq_len(ncol(aid_vecs)))  # V1..Vd

# Tibble bauen: IDs + Vektoren
aid_tbl <- tibble(
  AlphaID_token = aid_tokens,
  `Alpha-ID`    = sub("^(?i)aid_", "", aid_tokens)
) %>%
  dplyr::bind_cols(tibble::as_tibble(aid_vecs, .name_repair = "minimal")) %>%
  dplyr::mutate(`Alpha-ID` = as.character(`Alpha-ID`))

# Rechtes Join-Objekt vorbereiten (als Tibble, Key-Typ angleichen, einmalig pro Alpha-ID)
index_tbl <- tibble::as_tibble(index_tbl) %>%
  dplyr::mutate(`Alpha-ID` = as.character(`Alpha-ID`)) %>%
  dplyr::distinct(`Alpha-ID`, .keep_all = TRUE)

# Jetzt erst joinen – beide Seiten sind saubere Tibbles mit eindeutigen Namen
index_one <- index_tbl %>%
  mutate(`Alpha-ID` = as.character(`Alpha-ID`)) %>%
  distinct(`Alpha-ID`, .keep_all = TRUE)

aid_tbl   <- aid_tbl   %>% mutate(`Alpha-ID` = as.character(`Alpha-ID`))

idx <- match(aid_tbl$`Alpha-ID`, index_one$`Alpha-ID`)  # integer-Index, NA wenn kein Treffer

aid_tbl$Label         <- index_one$Label[idx]
aid_tbl$Label_norm    <- index_one$Label_norm[idx]
aid_tbl$`ICD-10-Code` <- index_one$`ICD-10-Code`[idx]

# Persistieren
write_delim(aid_tbl, AID_VECS_CSV, delim = ";")
readr::read_delim(AID_VECS_CSV, delim = ";", n_max = 10, show_col_types = FALSE)

# 5) Query-Daten laden & Embeddings (W2V: Mittelwert der Token)
# =========================
message("Lade Query-Daten: ", QUERY_FILE)
qry <- read_delim(QUERY_FILE, delim = ";", col_types = cols(.default = "c"))
stopifnot(any(c("Text extended","Text_extended","Text") %in% names(qry)))
text_col <- if ("Text extended" %in% names(qry)) "Text extended" else if ("Text_extended" %in% names(qry)) "Text_extended" else "Text"

normalize_tokens <- function(s) {
  normalize_text(s) |> str_split("\\s+", simplify = TRUE) |> as.vector()
}

# --- SIF-Gewichte vorbereiten (nutzt bereits berechnetes freq_tbl) ---
a <- 1e-3
N <- sum(freq_tbl$freq)
w <- setNames(a / (a + (freq_tbl$freq / N)), freq_tbl$token)

# Tokenizer-Vokabular
vocab <- rownames(model)

# OOV-Helfer (du hast fuzzy_in_vocab schon definiert)
embed_sentence_sif <- function(txt) {
  toks <- normalize_tokens(txt)
  toks <- toks[nzchar(toks)]
  if (!length(toks)) return(rep(0, ncol(model)))
  
  in_vocab <- toks[toks %in% vocab]
  if (!length(in_vocab)) {
    # Fallback: fuzzige Kandidaten pro Token, union aller Kandidaten
    cands <- unique(unlist(lapply(toks, fuzzy_in_vocab), use.names = FALSE))
    if (!length(cands)) return(rep(0, ncol(model)))
    M <- model[cands, , drop = FALSE]
    # Gleich gewichten, wenn komplett OOV
    return(colMeans(M))
  }
  
  # SIF-Gewichte
  ww <- w[in_vocab]; ww[is.na(ww)] <- a
  M  <- model[in_vocab, , drop = FALSE]
  # gewichtetes Mittel
  colSums(M * ww) / sum(ww)
}

message("Erzeuge Query-Embeddings (SIF, 1. PC entfernen) …")
qry_emb_mat <- do.call(rbind, lapply(qry[[text_col]], embed_sentence_sif))
# Falls alles Null ist, abbrechen-Guard
if (!is.null(dim(qry_emb_mat)) && ncol(qry_emb_mat) > 0) {
  # Erste Hauptkomponente entfernen (Arora et al.)
  pc <- prcomp(qry_emb_mat, center = FALSE, scale. = FALSE)
  if (!is.null(pc$rotation) && ncol(pc$rotation) >= 1) {
    u1 <- pc$rotation[, 1, drop = FALSE]              # d x 1
    qry_emb_mat <- qry_emb_mat - (qry_emb_mat %*% u1) %*% t(u1)
  }
}

qry_emb_tbl <- as_tibble(qry_emb_mat)
colnames(qry_emb_tbl) <- paste0("V", seq_len(ncol(qry_emb_tbl)))
qry_to_save <- bind_cols(qry, qry_emb_tbl)

write_delim(qry_to_save, QRY_EMB_CSV, delim = ";")
message("Query-Embeddings gespeichert: ", QRY_EMB_CSV)

if (FALSE) {
# Mittelwert-Embeddings pro Query
  message("Erzeuge Query-Embeddings (W2V-Mittelwerte) …")
  get_mean_vec <- function(txt) {
    toks <- normalize_tokens(txt)
    toks <- toks[nzchar(toks)]
    if (!length(toks)) return(rep(0, ncol(model)))
    
    in_vocab <- toks[toks %in% vocab]
    if (length(in_vocab)) return(colMeans(model[in_vocab, , drop = FALSE]))
    
    # OOV: fuzzy Kandidaten pro Token sammeln
    cands <- unique(unlist(lapply(toks, fuzzy_in_vocab), use.names = FALSE))
    if (!length(cands)) return(rep(0, ncol(model)))
    colMeans(model[cands, , drop = FALSE])
    qry_emb_list <- purrr::map(qry[[text_col]], get_mean_vec)
    qry_emb_mat  <- do.call(rbind, qry_emb_list)
    qry_emb_tbl <- as_tibble(qry_emb_mat)
    colnames(qry_emb_tbl) <- paste0(V, seq_len(ncol(qry_emb_tbl)))
    qry_to_save <- bind_cols(qry, qry_emb_tbl)
    
    write_delim(qry_to_save, QRY_EMB_CSV, delim = ";")
    message("Query-Embeddings gespeichert: ", QRY_EMB_CSV)
    
  }
}

str(qry_emb_mat)
nrow(qry_emb_mat); ncol(qry_emb_mat)     # darf nicht NULL sein
is.matrix(qry_emb_mat)                   # sollte TRUE sein


# 6) Nearest Neighbors: Query vs. Alpha-ID-Vektoren (Top-k)
# =========================
message("Berechne Top-", top_k, " Nachbarn je Query …")
A <- qry_emb_mat
B <- as.matrix(aid_vecs)  # Reihen = AID-Tokens

# Cosine-Ähnlichkeit
S <- cosine_sim(A, B)     # (n_query x n_aid)
S <- as.matrix(S)
n_q <- nrow(S); n_aid <- ncol(S)
k_eff <- min(top_k, n_aid)
if (k_eff == 0) stop("Keine Kandidaten (n_aid=0).")

get_topk_idx <- function(row, k) {
  fin <- which(is.finite(row))
  if (!length(fin)) return(rep(NA_integer_, k))
  o <- fin[order(row[fin], decreasing = TRUE)]
  out <- head(o, k); length(out) <- k
  out
}

topk_idx <- t(vapply(seq_len(n_q), function(i) get_topk_idx(S[i, ], k_eff),
                     FUN.VALUE = integer(k_eff)))

topk_sim <- matrix(NA_real_, nrow = n_q, ncol = k_eff)
for (i in seq_len(n_q)) {
  idx <- topk_idx[i, ]
  sel <- !is.na(idx)
  if (any(sel)) topk_sim[i, sel] <- S[i, idx[sel]]
}

# Lookups für Labels/ICD bauen (schnell, ohne mehrfaches match)
lab_by_token <- setNames(aid_tbl$Label,        aid_tbl$AlphaID_token)
icd_by_token <- setNames(aid_tbl$`ICD-10-Code`, aid_tbl$AlphaID_token)
results_tbl <- qry %>% mutate(`__rowid` = row_number())
# --- Hybrid Re-Ranking: Cosine + Jaro-Winkler auf Label ---
lambda <- 0.7  # Gewicht für Cosine (0.6–0.8 testen)

for (i in seq_len(n_q)) {
  idx <- topk_idx[i, ]
  sel <- !is.na(idx)
  if (!any(sel)) next
  
  cand_idx    <- idx[sel]
  cand_tokens <- aid_tokens[cand_idx]
  cand_labels <- unname(lab_by_token[cand_tokens])           # Labels
  cos_i       <- S[i, cand_idx]                              # Cosine dieser Query
  
  # Query-Text + Kandidaten normalisieren
  qtxt_norm <- normalize_text(qry[[text_col]][i])
  lab_norm  <- normalize_text(ifelse(is.na(cand_labels), "", cand_labels))
  
  # Jaro-Winkler-Ähnlichkeit (0..1)
  jw <- stringdist::stringsim(qtxt_norm, lab_norm, method = "jw")
  
  # Hybrid-Score
  hybrid <- lambda * cos_i + (1 - lambda) * jw
  ord    <- order(hybrid, decreasing = TRUE)
  
  # Reihenfolge überschreiben
  topk_idx[i, sel] <- cand_idx[ord]
  topk_sim[i, sel] <- cos_i[ord]   # konsistent zu distances = 1 - sim
}


# Zum Überprüfen der Parameter. Wichtig! 
message("Zum Überpürfen der Matrix")
cat("Queries (n_q) =", nrow(A), "  AlphaIDs (n_aid) =", nrow(B), "\n")
cat("Any NA/NaN/Inf in S? ", any(!is.finite(S)), "\n")

# Sind Query-Vektoren Null (keine bekannten Tokens)?
sum(rowSums(A^2) == 0)

# Ist aid_vecs korrekt und Reihenfolge stabil?
dim(aid_vecs); head(rownames(aid_vecs))
# MUSS passen zu:
length(aid_tokens); head(aid_tokens)

# Kippte apply/t() die Dimensionen?
is.matrix(S); is.matrix(topk_idx); dim(topk_idx)


for (k in seq_len(top_k)) {
  idx_k <- topk_idx[, k]
  sim_k <- topk_sim[, k]
  
  neighbor_tokens <- ifelse(is.na(idx_k), NA_character_, aid_tokens[idx_k])
  neighbor_ids    <- sub("^(?i)aid_", "", neighbor_tokens)
  
  neighbor_df <- tibble(
    `__rowid`                      = seq_len(n_q),
    !!paste0("suggestedIDs", k)    := neighbor_ids,
    !!paste0("suggestedDocuments", k) := unname(lab_by_token[neighbor_tokens]),
    !!paste0("suggestedMetadata", k)  := unname(icd_by_token[neighbor_tokens]),
    !!paste0("distances", k)       := 1 - sim_k
  )
  
  results_tbl <- left_join(results_tbl, neighbor_df, by = "__rowid")
}

results_tbl <- results_tbl %>% select(-`__rowid`)
write_delim(results_tbl, NN_RESULTS_CSV, delim = ";")
message("Fertig ✅  Ergebnisse: ", NN_RESULTS_CSV)


