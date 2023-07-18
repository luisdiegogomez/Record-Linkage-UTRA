import numpy as np
import pandas  
from pandas import read_csv
import random
from enum import Enum

class ErrorLvl(Enum): 
    HIGH = .6
    MID = .4 
    LOW = .2 

A = pandas.read_csv("generated_csv2.csv")
N_a = len(A.index)

def insert_error(cols, err_func, err_lvls) -> list: 
    for col in cols: 
        err_lvl = err_lvls.get(col)
        row_indices = np.array([i for i in range(N_a)])
        num_err_rows = round(err_lvl*N_a)
        selected_rows = np.random.choice(row_indices, num_err_rows)
        for r in selected_rows: 
            err_func(r, A.columns.get_loc(col))

#delete errors in string: random character is deleted from given string in data frame
def char_deletion(r, col):
    mystr = A.iloc[r, col] 
    del_char = random.randint(0, len(mystr)-1)
    newstr = mystr[:del_char] + mystr[del_char +1:]
    A.iloc[r, col] = newstr    
        
#transposition errors in string: randomly chosen characters are swapped in given string in data frame 
def char_transposition(r, col): 
    mystr = A.iloc[r, col] 
    del_indices = random.sample(range(0, len(mystr)-1), 2)
    char_list = [char for char in mystr]
    temp = char_list[del_indices[0]]
    char_list[del_indices[0]] = char_list[del_indices[1]]
    char_list[del_indices[1]] = temp
    newstr = ''.join(char_list)
    A.iloc[r, col] = newstr

# insert errors for categorical data: randomly selects a new category 
def categorical_swap(r, col): 
    categories = A[col].cat.categories
    A.iloc[r, col] = np.random.choice(categories, 1)

def bool_error(r, col):
    A.iloc[r, col] = np.random.choice([True, False], 1)

#TODO: fix gaussian error insertion 
def int_gaussian_error(r, col): 
    myint = A.iloc[r, col]

def float_gaussian_error(r, col): 
    myfloat = A.iloc[r, col]
    noise = np.random.normal(0,0,1)

#TODO: datetime error insertion 
def datetime_error(r, col): 
    myDate = A.iloc[r, col]
    dateStr = myDate.strftime()
    char_list = [char for char in dateStr]


def insert_errors(newDataPath : str, errorList : list[ErrorLvl]): 

    assert(len(errorList) == len(A.columns)), ("Must provide an error level for each column")
    err_lvls = {(list(A.columns))[i] : errorList[i].value for i in range(len(errorList))}

    integer_columns = A.select_dtypes(include=['int64']).columns
    float_columns = A.select_dtypes(include=['float64']).columns
    object_columns = A.select_dtypes(include=['object']).columns
    datetime_columns = A.select_dtypes(include=['datetime64']).columns
    bool_columns = A.select_dtypes(include=['bool']).columns
    category_columns = A.select_dtypes(include=['category']).columns

    insert_error(category_columns, categorical_swap, err_lvls)
    insert_error(bool_columns, bool_error, err_lvls)
    insert_error(object_columns, char_deletion, err_lvls)
    insert_error(object_columns, char_transposition, err_lvls)
    
    A.to_csv(newDataPath)

error = [ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH ]

insert_errors("generated_csv3.csv", error)