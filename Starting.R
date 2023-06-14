
main <- function(file_A, file_B){ 
  A <- read.csv(file_A)
  B <- read.csv(file_B)
  
  data_A = A[, intersect(names(A), names(B))]
  data_B = B[, intersect(names(A), names(B))]
  
  
  
  
  
  
  
  

  

}

fill_comparison_array <- function(record_a, record_b){ 
  
  }




