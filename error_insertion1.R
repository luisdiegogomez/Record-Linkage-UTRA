#error insertion: in order to simulate having two datasets that contain known 
# matches and known non-matches, we insert errors into a duplicate dataset 


#errors in name: random character is deleted from given string in data frame
insert_str_deletion_error <- function(myfile, row, col){ 
  string <- myfile[row, col]
  print(string)
  
  
  del_char = sample(1:nchar(string), 1)
  new_string <- paste(substr(string, 1, del_char-1),substr(string, del_char+1, nchar(string)), sep = "")
  
  print(new_string)
  myfile[row, col] <- new_string
}

#errors in name: randomly chosen characters are swapped in given string in data frame 
insert_str_transposition_error <- function(myfile, row, col){ 
  string <- myfile[row, col]
  print(string)
  
  swap_chars <- sample(1:nchar(string), 2, replace = F)
  char_vec <- unlist(strsplit(string, split = ""))
  temp = char_vec[swap_chars[1]]
  char_vec[swap_chars[1]] <- char_vec[swap_chars[2]]
  char_vec[swap_chars[2]] <- temp 
  
  new_string <- paste(char_vec, collapse = "")
  print(new_string)
  myfile[row, col] <- new_string
  
}

#errors in date 
# insert_date_error  <- function(myfile, row, col){
#   
# }


#Inserts errors based on a bernoulli distribution with prob p 
insert_errors_bern <- function(my_file){
  N <- nrow(my_file)
  p = 0.5
  #indicator variable I ~ Bernoulli(p):will insert errors in records i where Ii= 1
  I <- rbinom(N, 1, p)
  print(I)
  for(i in 1: N){ 
    if(I[i] == 1){ 
      insert_str_deletion_error(my_file, i, fields[1])
      insert_str_transposition_error(my_file, i, fields[6])
    }
  }
}

A <- read.csv("~/Record-Linkage-UTRA/TestCSV.csv")
fields <- names(A)
insert_errors_bern(A)