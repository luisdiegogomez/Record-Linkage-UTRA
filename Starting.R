

A <- read.csv("~/Record-Linkage-UTRA/TestCSV.csv")
B <- read.csv("~/Record-Linkage-UTRA/TestCSV.csv")

dataA <- A[, intersect(names(A), names(B))]
dataB <- B[, intersect(names(A), names(B))]

N_a = (length(dataA) - 1)
N_b = (length(dataB) - 1)

fill_compaison_arrays <- function(recordA, recordB){ 
  comparison_arrays <- vector(mode = "character", length = 
  for(a in dataA){ 
    for(b in dataB){
      compare_array =
    }
  }
}

