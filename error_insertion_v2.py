import numpy as np
import pandas  
from pandas import read_csv
import random
from enum import Enum

class ErrorLvl(Enum): 
    HIGH = .6
    MID = .4 
    LOW = .2 

A = pandas.read_csv("TestCSV.csv")
N_a = len(A.index)
err_lvls ={}

#delete errors in string: random character is deleted from given string in data frame
def insert_str_deletion_error(str_cols):
    for col in str_cols: 
        err_lvl = err_lvls.get(col)
        selected_rows = np.random.sample([i for i in range(N_a)], round(err_lvl*N_a))
        for r in selected_rows: 
            mystr = A.iloc[r, col] 
            del_char = random.randint(0, len(mystr)-1)
            newstr = mystr[:del_char] + mystr[del_char +1:]
            A.iloc[r, col] = newstr    
    
#transposition errors in string: randomly chosen characters are swapped in given string in data frame 
def insert_str_transpose_error(str_cols): 
    for col in str_cols: 
        err_lvl = err_lvls.get(col)
        selected_rows = np.random.sample([i for i in range(N_a)], round(err_lvl*N_a))
        for r in selected_rows: 
            mystr = A.loc[r, col] 
            del_indices = random.sample(range(0, len(mystr)-1), 2)
            char_list = [char for char in mystr]
            temp = char_list[del_indices[0]]
            char_list[del_indices[0]] = char_list[del_indices[1]]
            char_list[del_indices[1]] = temp
            newstr = ''.join(char_list)
            A.iloc[r, col] = newstr

# insert errors for categorical data: randomly selects a new category 
def insert_category_error(cat_cols): 
    for col in cat_cols: 
        err_lvl = err_lvls.get(col)
        selected_rows = np.random.sample([i for i in range(N_a)], round(err_lvl.value() *N_a))
        categories = A[col].cat.categories
        for r in selected_rows: 
            A.loc[r, col] = np.random.choice(categories, 1)

# def insert_gaussian_error(float_cols, is_int): 
#     for col in float_cols: 
#         err_lvl = err_lvls.get(col)
#         selected_rows = np.random.sample([i for i in range(N_a)], round(err_lvl*N_a))
#         mu = 0 
#         sig = 0 
#         for r in selected_rows:
#             myfloat = A.loc[r, col] 
#             noise = np.random.normal(mu,sig,1)
#             newval = myfloat + noise 
#             if is_int ==1 : 
#                 newval = round(newval)
#             A.loc[r, col] = newval

def insert_bool_error(bool_cols):
     for col in bool_cols: 
        err_lvl = (err_lvls.get(col))
        selected_rows = np.random.sample([i for i in range(N_a)], round(err_lvl*N_a))
        for r in selected_rows: 
            A.loc[r, col] = np.random.choice([True, False], 1)


def insert_errors(newDataPath : str, error_lvls : list[ErrorLvl]): 

    assert(len(error_lvls) == len(A.columns)), ("Must provide an error level for each column")
    err_lvls = {(list(A.columns))[i] : error_lvls[i].value for i in range(len(error_lvls))}

    integer_columns = A.select_dtypes(include=['int64']).columns
    float_columns = A.select_dtypes(include=['float64']).columns
    object_columns = A.select_dtypes(include=['object']).columns
    datetime_columns = A.select_dtypes(include=['datetime64']).columns
    bool_columns = A.select_dtypes(include=['bool']).columns
    category_columns = A.select_dtypes(include=['category']).columns

    insert_category_error(category_columns)
    insert_bool_error(bool_columns)
    insert_str_deletion_error(object_columns)
    insert_str_transpose_error(object_columns)
    # insert_gaussian_error(float_columns, 0)
    # insert_gaussian_error(integer_columns, 1)

    A.to_csv(newDataPath)


error = [ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH, ErrorLvl.HIGH ]

insert_errors("TestCSVcooopy.csv", error)