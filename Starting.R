A <- read.csv("~/R/Record-Linkage-UTRA/TestCSV.csv")
B <- read.csv("~/R/Record-Linkage-UTRA/Test CSV 2.csv")

fill_comparison_arrays <- function(recordA, recordB){ 
  
  N_a <- nrow(recordA)
  N_b <- nrow(recordB)
  
  X_a <- recordA[, intersect(names(recordA), names(recordB))]
  X_b <- recordB[, intersect(names(recordA), names(recordB))]
  
  K <- ncol(X_a)
  
  return_comparison_arrays <- matrix(data = vector(mode = "logical",length = K),N_a,N_b)

  for(a in 1:N_a){
    for(b in 1:N_b){
      comparison_array <- vector(mode = "logical", length = K)
      for(r in 1:nrow(X_a)){
        for(c in 1:ncol(X_a)){
          if(X_a[r,c] == X_b[r,c]){
            comparison_array[c] <- TRUE
          }
          else{
            comparison_array[c] <- FALSE
          }
        }
      }
      return_comparison_arrays[a,b] <- comparison_array
    }
  }
  return(return_comparison_arrays)
}

C_matrix_filler <- function(comparison_arrays){ 
  
  ones_vector <- vector(mode = "logical", length = length(comparison_arrays[1,1]))
  for(i in 1:length(ones_vector)){
    ones_vector[i] <- TRUE
  }
  retrun_c_matrix <- matrix(data = 0,nrow(comparison_arrays),ncol(comparison_arrays))
  
  for(r in 1:nrow(comparison_arrays)){
    for(c in 1:ncol(comparison_arrays)){
      if(comparison_arrays[r,c] == ones_vector){
        return_c_matrix[r,c] <- 1
      }
      else{
        return_c_matrix[r,c] <- 0
      }
    }
  }
  
  return(return_c_matrix)
}

comparison_arrays_ex <- fill_comparison_arrays(A,B)
c_matrix <- C_matrix_filler(comparison_arrays_ex)

View(comparison_arrays_ex)
View(c_matrix)

