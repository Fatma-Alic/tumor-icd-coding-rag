# library("devtools")
# install_github("bmschmidt/wordVectors")
library(wordVectors) #verarbeitet W2V-Modelle  
library(stringr) #Stringmanipulation 
library(plyr) #Datenmanipulation 

model = read.vectors('/app/MODEL.bin', binary=1) 
words = rownames(model)
words = str_replace_all(words,'&#32;',' ')
rownames(model)=words

icd_clean = read.delim("C:/Users/ali0f/Documents/word2vec/ICD-10-Katalog.csv_only_tumordiagnoses.csv", sep=";", stringsAsFactors=F, header=T)
names(icd_clean)
icd = read.delim("C:/Users/ali0f/Documents/word2vec/fully_cleaned_gtds_just_2023_2024_AcronymsExtended.csv", sep=";", stringsAsFactors=F, header=T)
names(icd)

replace_umlauts <- function(x) {
  umlauts <- "äöü"
  UMLAUTS <- "ÄÖÜ"
  UMLAUT2 <- "ß"
  UMLAUT3 <- "-"
  x <- gsub(pattern = paste0("([", UMLAUTS, "])"), replacement = "\\1E", x)
  x <- gsub(pattern = paste0("([", umlauts, "])"), replacement = "\\1e", x)
  x <- gsub(pattern = paste0("([", UMLAUT2, "])"), replacement = "ss", x)
  x <- gsub(pattern = paste0("([", UMLAUT3, "])"), replacement = " ", x)
  x <- chartr(old = paste0(UMLAUTS, umlauts), new = "AOUaou", x)
  return(x)
}

normalize_text <- function(x) {
  x |>
    stri_trans_general("Latin-ASCII") |> # extra safety
    tolower() |>
    replace_umlauts() |>
    str_replace_all("[^a-z0-9 ]+", " ") |>
    str_replace_all("\\s+", " ") |>
    str_trim()
}

alpha = read.delim("C:/Users/ali0f/Documents/word2vec/complete_icd10_codes_2024.csv_only_tumordiagnoses.csv", sep=";", stringsAsFactors=F, header=T)
alpha$ICD-Code = ifelse(alpha$V3 == '', alpha$ICD-Code, alpha$V3) #was bedeutet v3? 
alpha$Label = replace_umlauts (tolower(alpha$Label))
alpha= subset(alpha, select=c(ICD-Code, Label))
names(alpha)=c("V1","V2")

icd = unique(rbind(icd, alpha))
icd$V2 = gsub('[^a-z 0-9]+', ' ', icd$V2)
icd$V2 = gsub(' +', ' ', icd$V2)

icd_embeddings_df1 = ddply(icd, .(V1), function(x){
  keywords = unique(x$V2)
  
  return(model[[keywords, average=T]])
})

icd_embeddings_matrix = as.matrix(subset(rbind(icd_embeddings_df1), select=-c(V1)))
row.names(icd_embeddings_matrix) = c(icd_embeddings_df1$V1)
icd_embeddings = as.VectorSpaceModel(icd_embeddings_matrix)

# inference...
#* @get /icd_from_diagnosis
icd_from_diagnosis= function (input="katarakt")
{
  icd_found = ''
  diagnosis = unlist(strsplit(input, ' '))
  
  num_words = length(diagnosis)
  
  if (num_words > 1 & input %in% rownames(model))
    diagnosis = c(input, diagnosis)
  
  diagnosis = unique(diagnosis)
  
  # print(diagnosis)
  
  # direkte treffer bevorzugen
  if (nrow(subset(icd, V2 == input)) > 0)
  {
    icd_found = subset(icd, V2==input)[1,]$V1
  }
  else
  {
    
    query = model[[diagnosis, average=T]]
    d = cosineDist(query, icd_embeddings)
    icd_found = unlist(dimnames(d))[[which.min(d)]]
    
    threshold = 0.5
    
    if (d[which.min(d)] > threshold)
      icd_found = ''
  }
  
  if (nchar(icd_found))
  {
    icd_found = gsub('[+]+', '', icd_found)
    icd_found2 = gsub('[A-Z]+$', '', icd_found)
    icd_found2 = gsub('(.+)[A-Z]+[0-9]+$', '\\1', icd_found2)
    return (c(icd_found, subset(icd_clean, V7 == icd_found2)[1,]$V9))
  }
  
  return (NA)
}