import numpy as np

import pandas as pd

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
A = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/TestCSV.csv")
B = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/Test CSV 2.csv")

## Global Variables:

N_a = len(A.index)
N_b = len(B.index)
  
X_a = A[np.sort(A.columns.intersection(B.columns))]
X_b = B[np.sort(B.columns.intersection(A.columns))]
  
K = len(X_a.columns)

## Filling in Comparison Vectors (Gamma Vectors):

# Function that outputs 3-D array (dimensions N_a,N_b,K) representing the comparison gamma 
# vectors for each pair of records between files A and B:
def fill_comparison_arrays(recordA:pd.DataFrame,recordB:pd.DataFrame) -> np.ndarray:

    # Initializing matrix of comparison gamma vectors:
    comparison_arrays = np.full((N_a, N_b, K), fill_value = 0) # N_a by N_b matrix with each cell containing 
                                                               # a gamma comparison vector (of size K) for each  
                                                               # pair of files in A and B

    # Filling comparison vectors:
    for a in range(N_a):
        for b in range(N_b):
            for col in range(K):
                if X_a.iat[a,col] == X_b.iat[b,col]:
                    comparison_arrays[a,b,col] = 1
                else:
                    comparison_arrays[a,b,col] = 0

    ## Converting the matrix of comparison vectors to a pandas DataFrame
    # return_comparison_arrays = pd.DataFrame(index=range(N_a), columns=range(N_b))
    # for a in range(N_a):
    #     for b in range(N_b):
    #         return_comparison_arrays.iat[a, b] = comparison_arrays[a,b]

    return(comparison_arrays)

test_comp_array = fill_comparison_arrays(A,B)

## Sampling Theta Values for Comparison Vectors:

def theta_and_c_sampler(comparison_arrays:np.ndarray, t:int):
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    theta_M_params = [1,2]
    theta_U_params = [1,1]

    #Initilaizaing C with each record pair having one random match:
    C = np.full((N_a,N_b), 0)
    N_b_shuffled = np.array(range(N_b))
    np.random.shuffle(N_b_shuffled)
    for r in range(N_a):
        C[r,N_b_shuffled[r]] = 1

    # Gibbs Sampler for Theta Values
    theta_values = np.full((K, t, 2), 0) # Array with K rows (one for each comparison variable),
                                         # t columns (one for each number of iterations), and 
                                         # two theta values in each cell (Theta_M and Theta_U 
                                         # values for each comparison variable)

    for i in range(t):
        for gamma_col in range(K):
            ## Sampling for Theta_M Values:
            # First Parameter for Dirichlet Distribution:
            alpha_M_0 = theta_M_params[0] + np.sum(comparison_arrays[:,:,gamma_col]*C)
            # Second Parameter for Dirichlet Distribution:
            alpha_M_1 = theta_M_params[1] + np.sum((1- comparison_arrays[:,:,gamma_col])*C)

            theta_values[gamma_col,i,0] = np.random.dirichlet(np.array(alpha_M_0, alpha_M_1))

            ## Sampling for Theta_U Values:
            # First Parameter for Dirichlet Distribution:
            alpha_U_0 = theta_U_params[0] + np.sum(comparison_arrays[:,:,gamma_col]*(1-C))
            # Second Parameter for Dirichlet Distribution:
            alpha_U_1 = theta_U_params[1] + np.sum((1- comparison_arrays[:,:,gamma_col])*(1-C))

            theta_values[gamma_col,i,1] = np.random.dirichlet(np.array(alpha_U_0, alpha_U_1))

    

    return(theta_values)

theta_values = theta_and_c_sampler(test_comp_array, 10)
print(theta_values)