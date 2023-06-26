import numpy as np
import pandas  
from pandas import read_csv
import random

A = pandas.read_csv("TestCSV.csv")
#indices of the columns to insert errors into. We start with the name and age columns 
name_index = 0 
income_index = 5

#delete errors in name: random character is deleted from given string in data frame
def insert_str_deletion_error(myfile, row, col):
    mystr = myfile.iloc[row, col] 
    print(mystr)

    del_char = random.randint(0, len(mystr)-1)
    newstr = mystr[:del_char] + mystr[del_char +1:]
    print(newstr)

    myfile.iloc[row, col] = newstr

#transposition errors in name: randomly chosen characters are swapped in given string in data frame 
def insert_str_transpose_error(myfile, row, col): 
    mystr = myfile.iloc[row, col] 
    print(mystr)
    del_indices = random.sample(range(0, len(mystr)-1), 2)
    char_list = [char for char in mystr]
    temp = char_list[del_indices[0]]
    char_list[del_indices[0]] = char_list[del_indices[1]]
    char_list[del_indices[1]] = temp

    newstr = ''.join(char_list)
    print(newstr)
    myfile.iloc[row, col] = newstr

#Inserts errors based on a bernoulli distribution with prob p 
def insert_errors(mydata): 
    N = len(mydata)
    p = 0.5 
    #indicator variable I ~ Bernoulli(p):will insert errors in records i where Ii= 1
    I = np.random.binomial(1, p, N)
    print(I)
    for i in range(N): 
        if I[i] == 1: 
            insert_str_deletion_error(mydata, i, name_index)
            insert_str_transpose_error(mydata, i, income_index)

insert_errors(A)
A.to_csv('TestCSVcopy.csv')


