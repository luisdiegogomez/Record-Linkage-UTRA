import numpy as np

import pandas as pd
import scipy as scipy
import math as math

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
A = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\TestCSV.csv")
B = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\TestCSV.csv")

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

def theta_and_c_sampler(comparison_arrays:np.ndarray, T:int):
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    theta_M_params = [1,2]
    theta_U_params = [1,1]

    #Initilaizaing C with each record pair having one random match:
    C = np.full((N_a,N_b), 0)
    # N_b_shuffled = np.array(range(N_b))
    # np.random.shuffle(N_b_shuffled)
    # for r in range(N_a):
    #     C[r,N_b_shuffled[r]] = 1

    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((K, T, 2), fill_value=np.full((1,2), 0, dtype= float), dtype= np.ndarray) # Array with K rows (one for each comparison variable),
                                                                                            # t columns (one for each number of iterations), and 
                                                                                            # two theta values in each cell (Theta_M and Theta_U 
                                                                                            # values for each comparison variable)

    for t in range(T):
        #Step 1: sampling thetas 
        for gamma_col in range(K):
            ## Sampling for Theta_M Values:
            # First Parameter for Dirichlet Distribution:
            alpha_M_0 = theta_M_params[0] + np.sum(comparison_arrays[:,:,gamma_col]*C)
            # Second Parameter for Dirichlet Distribution:
            alpha_M_1 = theta_M_params[1] + np.sum((1- comparison_arrays[:,:,gamma_col])*C)

            theta_values[gamma_col,t,0] = np.random.dirichlet(np.array([alpha_M_0, alpha_M_1]))
            ## Sampling for Theta_U Values:
            # First Parameter for Dirichlet Distribution:
            alpha_U_0 = theta_U_params[0] + np.sum(comparison_arrays[:,:,gamma_col]*(1-C))
            # Second Parameter for Dirichlet Distribution:
            alpha_U_1 = theta_U_params[1] + np.sum((1- comparison_arrays[:,:,gamma_col])*(1-C))

            theta_values[gamma_col,t,1] = np.random.dirichlet(np.array([alpha_U_0, alpha_U_1]))

        #Step 2: sampling C
        #C[t+1]: empty
        C = np.full((N_a,N_b), 0)

        # For every file a (ie. every row of C)
        for a in range(N_a): 
            # list of length N_b: for each b, elt is b's index if b does not have a link, else None
            b_link_status = [b if not(1 in C[:, b]) else None for b in range(N_b)]
            # list of indices of unlinked b files: we will be choosing a link between file a and one of these unlinked bs
            b_unlinked = list(filter(lambda x: x != None, b_link_status)) 

            # TODO: make neat: 
            def likelihood_ratio(a, b) -> float: 
                m_lh = 1
                u_lh = 1
                for k in range(K): 
                    theta_k_m0 = theta_values[k, t, 0][0]
                    theta_k_m1 = theta_values[k, t, 0][1]
                    m_lh = m_lh * (theta_k_m0)**comparison_arrays[a, b, k] * (theta_k_m1)**(1-comparison_arrays[a, b, k])

                    theta_k_u0 = theta_values[k, t, 1][0]
                    theta_k_u1 = theta_values[k, t, 1][1]
                    u_lh = u_lh * (theta_k_u0)**comparison_arrays[a, b, k] * (theta_k_u1)**(1-comparison_arrays[a, b, k])
                
                lr = m_lh/u_lh 
                return lr

            num = [likelihood_ratio(a, b) for b in b_unlinked]
            denom = [sum(num)] * len(b_unlinked)
            link_probs = [i / j for i, j in zip(num, denom)]

            #samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(b_unlinked))], 1, True, link_probs))[0]        
            C[a, b_unlinked[new_link_index]] = np.random.binomial(1, link_probs[new_link_index])
    
        print(C, t)
    return(theta_values)

theta_values = theta_and_c_sampler(test_comp_array, 10)
#print(theta_values)