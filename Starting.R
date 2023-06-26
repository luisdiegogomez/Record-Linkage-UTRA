A <- read.csv("~/R/Record-Linkage-UTRA/TestCSV.csv")
B <- read.csv("~/R/Record-Linkage-UTRA/Test CSV 2.csv")

# Function that outputs data frame (dimensions N_a,N_b) representing the comparison gamma 
# vectors for each pair of records between files A and B
fill_comparison_arrays <- function(recordA, recordB){ 
  
  N_a <- nrow(recordA)
  N_b <- nrow(recordB)
  
  X_a <- recordA[, intersect(names(recordA), names(recordB))]
  X_b <- recordB[, intersect(names(recordA), names(recordB))]
  
  K <- ncol(X_a)
  
  # Initialization of the matrix of comparison gamma vectors 
  return_comparison_arrays <- matrix(data = list(vector(mode = "logical", length = K)),nrow = N_a, ncol = N_b)
  
  # Filling comparison vectors
  for(a in 1:N_a){
    for(b in 1:N_b){
      for(col in 1:K){
        if(X_a[a,col] == X_b[b,col]){
          return_comparison_arrays[a,b][[1]][col] <- TRUE
        }
        else{
          return_comparison_arrays[a,b][[1]][col] <- FALSE
        }
      }
    }
  }
  
  return_comparison_arrays <- as.data.frame(return_comparison_arrays)
  return(return_comparison_arrays)
}

# comparison_arrays_ex <- fill_comparison_arrays(A,B)
# 
# View(comparison_arrays_ex)

