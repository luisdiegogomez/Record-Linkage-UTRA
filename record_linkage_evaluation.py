import numpy as np
import pandas as pd
import math as math

# Files A and B 
# X% (say 50 %) of records in B chosen refer to the same entities as records in A (call this subset of records C)

# Simulating historical data: randomly chosen Y% (say 30%) of records in B 
#  
# Generate C 
# C combined with additional data to form A 
# C combined with additional data to form B 
# Y% of A, B selected to be “known”/historical data 
# Those from C labeled as known matches in initial linkage structure (py script)
# Those  from C but without pair labeled as known non-matches in initial linkage structure (py script)
# Errors inserted separately into A and B 
# Record linkage algorithm run on A, B 
# Test with different X%, Y%, corruption levels, alpha levels (all needs to be systematized)
# Evaluation metrics: 
# Precision 
# Recall 

# Ask DY 



shared = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\____.csv")
A_partial = pd.read_csv(r"")
B_partial = pd.read_csv(r"")

X_a = A_partial[np.sort(A_partial.columns.intersection(shared.columns))]
X_b = B_partial[np.sort(B_partial.columns.intersection(shared.columns))]

A_temp = pd.concat([shared, A_partial], axis = 0)
B_temp = pd.concat([shared, B_partial], axis = 0)

#A is the larger file 
if len(A_temp.index) >= len(B_temp.index):
    A = A_temp
    B = B_temp
    N_a = len(A.index) # Equivalent to N_a
    N_b = len(B.index) # Equivalent to N_b
else:
    A = B_temp
    B = A_temp
    N_a = len(A.index) # Equivalent to N_a
    N_b = len(B.index) # Equivalent to N_b

known_lvl = 0.5
C_init =  np.full(((N_a * N_b), 2), 0)



