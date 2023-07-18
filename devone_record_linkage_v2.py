### DEVONE IMPLEMENTATION V2: Here gamma values are stored in a 1D array of length Na x Nb 

import numpy as np

import pandas as pd

import math as math

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
A_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\generated_csv1.csv")
B_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\generated_csv2.csv")

## Global Variables:
N_a = 0
N_b = 0
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

  
X_a = A[np.sort(A.columns.intersection(B.columns))]
X_b = B[np.sort(B.columns.intersection(A.columns))]

print(X_a)
print(X_b)
  
K = len(X_a.columns)

## Filling in Comparison Vectors (Gamma Vectors):

# Function that outputs 3-D array (dimensions N_a,N_b,K) representing the comparison gamma 
# vectors for each pair of records between files A and B:
def fill_comparison_arrays(recordA:pd.DataFrame,recordB:pd.DataFrame) -> np.ndarray:

    # Initializing matrix of comparison gamma vectors:
    comparison_arrays = np.full((K, (N_a*N_b)), fill_value = 0, dtype= float) # N_a by N_b matrix with each cell containing 
                                                               # a gamma comparison vector (of size K) for each  
                                                               # pair of files in A and B
    # Filling comparison vectors:
    for a in range(N_a):
        for b in range(N_b):
            for k in range(K):
                if X_a.iat[a,k] == X_b.iat[b,k]:
                    comparison_arrays[k, ((N_b*a) + b)] = 1
                else:
                    comparison_arrays[k, ((N_b*a) + b)] = 0

    ## Converting the matrix of comparison vectors to a pandas DataFrame
    # return_comparison_arrays = pd.DataFrame(index=range(N_a), columns=range(N_b))
    # for a in range(N_a):
    #     for b in range(N_b):
    #         return_comparison_arrays.iat[a, b] = comparison_arrays[a,b]

    return(comparison_arrays)

test_comp_array = fill_comparison_arrays(A,B)
print(test_comp_array)

pandas_comp_array = pd.DataFrame(index=range(K), columns=range(N_a*N_b))
for a in range(N_a):
    for b in range(N_b):
            for k in range(K):
                pandas_comp_array.iat[k, N_b*a + b] = test_comp_array[k, N_b*a + b]

print("Comparison Array:")
print(pandas_comp_array)

## Sampling Theta Values for Comparison Vectors:

def theta_and_c_sampler(comparison_arrays:np.ndarray, T:int) -> tuple:
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    theta_M_params = [1,2]
    theta_U_params = [1,1]

    C = np.full((N_a*N_b), 0)

    
    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((K, T, 2), fill_value=np.full((1,2), 0, dtype= float), dtype= np.ndarray) # Array with K rows (one for each comparison variable),
                                                                                            # t columns (one for each number of iterations), and                                                                                             # two theta values in each cell (Theta_M and Theta_U 
                                                                                            # values for each comparison variable)
    for t in range(T):
        #Step 1: sampling thetas 
        for gamma_col in range(K):
            ## Sampling for Theta_M Values:
            # First Parameter for Dirichlet Distribution:
            #TODO: check this!!
            alpha_M_0 = theta_M_params[0] + np.sum(comparison_arrays[gamma_col, :]*C)
            # Second Parameter for Dirichlet Distribution:
            alpha_M_1 = theta_M_params[1] + np.sum((1- comparison_arrays[gamma_col, :])*C)

            theta_values[gamma_col,t,0] = np.random.dirichlet(np.array([alpha_M_0, alpha_M_1]))
            ## Sampling for Theta_U Values:
            # First Parameter for Dirichlet Distribution:
            alpha_U_0 = theta_U_params[0] + np.sum(comparison_arrays[gamma_col, :]*(1-C))
            # Second Parameter for Dirichlet Distribution:
            alpha_U_1 = theta_U_params[1] + np.sum((1- comparison_arrays[gamma_col, :])*(1-C))

            theta_values[gamma_col,t,1] = np.random.dirichlet(np.array([alpha_U_0, alpha_U_1]))

        #Step 2: sampling C
        #C[t+1]: empty
        C = np.full((N_a*N_b), 0)

        # For every file a (ie. every row of C)
        # b_unlked_lst= [b for b in range(N_b)]
        # b_unlinked = {b: 0 for b in b_unlinked_lst}in
        row_order_list = ([a for a in range(N_a)])
        np.random.shuffle(row_order_list)
        for a in row_order_list: 
            # list of length N_b: for each b, elt is b's index if b does not have a link, else None
            b_links = lambda  b : [C[N_b* a_n + b]  for a_n in range(N_a)]
            b_link_status = [b if sum(b_links(b))  == 0 else None for b in range(N_b)]
            # # list of indices of unlinked b files: we will be choosing a link between file a and one of these unlinked bs
            b_unlinked = list(filter(lambda x: x != None, b_link_status)) 
            num_links = N_b - len(b_unlinked)


            # TODO: make neat: 
            def likelihood_ratio(a, b) -> float: 
                m_lh = 1
                u_lh = 1
                for k in range(K): 
                    theta_k_m0 = theta_values[k, t, 0][0]
                    theta_k_m1 = theta_values[k, t, 0][1]
                    m_lh = m_lh * (theta_k_m0)**comparison_arrays[k, (N_b*a + b)] * (theta_k_m1)**(1-comparison_arrays[k, (N_b*a + b)])

                    theta_k_u0 = theta_values[k, t, 1][0]
                    theta_k_u1 = theta_values[k, t, 1][1]
                    u_lh = u_lh * (theta_k_u0)**comparison_arrays[k, (N_b*a + b)] * (theta_k_u1)**(1-comparison_arrays[k, (N_b*a + b)])
                
                lr = m_lh/u_lh 
                return lr
            
            #if there are no more unlinked bs, we just go on to next iteration of the sampler 
            if(b_unlinked == []): 
                break
            
            prob_no_link = (N_a - num_links)*(N_b - num_links)/(num_links + 1)
            num = [likelihood_ratio(a, b) for b in b_unlinked]
            num.append(prob_no_link)
            
            denom = [sum(num)] * len(num)
            link_probs = [i / j for i, j in zip(num, denom)]
           # print(link_probs)

            #samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(link_probs))], 1, True, link_probs))[0]   
            
            #last index in index list == no_link. if it selected a valid index, we want 
            if(new_link_index != len(b_unlinked)):   
                #print(a, b_unlinked[new_link_index], link_probs[new_link_index])
                C[N_b*a + b_unlinked[new_link_index]] = 1
    
    C_return = pd.DataFrame(index=range(N_a), columns=range(N_b))
    for a in range(N_a):
        for b in range(N_b):
            C_return.iat[a, b] = C[N_b*a +b]
    # print(C, t)
    return(theta_values,C_return)


theta_values = theta_and_c_sampler(test_comp_array, 900)
# print("Theta Values:")
# print(theta_values[0])
print("C Structure:")
print(theta_values[1])