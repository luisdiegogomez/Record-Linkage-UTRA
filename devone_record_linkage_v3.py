### DEVONE IMPLEMENTATION V2: Here gamma values are stored in a 1D array of length Na x Nb, and there are L instead of 2 agreement levels (determined by jaro winkler)

import numpy as np

import pandas as pd

import math as math

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
A_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv1.csv")
B_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv2.csv")

## Global Variables:

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

  
X_a = A[np.sort(A.columns.intersection(B.columns))]
X_b = B[np.sort(B.columns.intersection(A.columns))]

K = len(X_a.columns)

L_k = np.arange(0, 1.1 ,0.1)
num_agreement_lvls = len(L_k) # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

#Returns jaro_winkler_distance of two strings
def jaro_winkler_distance(s1, s2):
    # Jaro distance
    len_s1, len_s2 = len(s1), len(s2)
    max_dist = max(len_s1, len_s2) // 2 - 1
    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len_s1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len_s2)
        for j in range(start, end):
            if s1[i] == s2[j]:
                matches += 1
                if i != j:
                    transpositions += 1
                break

    if matches == 0:
        return 0.0

    jaro = (
        (matches / len_s1)
        + (matches / len_s2)
        + ((matches - transpositions / 2) / matches)
    ) / 3

    # Winkler modification
    prefix_len = 0
    for i in range(min(len_s1, len_s2)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    jaro_winkler = round(jaro + (0.1 * prefix_len * (1 - jaro)), 1)
    if jaro_winkler > 1: jaro_winkler = 1
    
    return jaro_winkler

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
                comparison_arrays[k, ((N_b*a) + b)] = jaro_winkler_distance(str(X_a.iat[a,k]), str(X_b.iat[b,k]))
    return(comparison_arrays)



## Sampling Theta Values for Comparison Vectors:

def theta_and_c_sampler(comparison_arrays:np.ndarray, T:int) -> tuple:
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    M_alpha_priors = np.full(num_agreement_lvls, 1, dtype=int)
    U_alpha_priors = np.full(num_agreement_lvls, 1, dtype=int)

    C = np.full((N_a*N_b), 0)
    # C[N_b* 0 + 0] = 1
    # C[N_b* 1 + 1] = 1
    # C[N_b* 4 + 4] = 1
    # C[N_b* 3 + 13] = 1


    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((T, K, 2, num_agreement_lvls), 0.00, dtype=float) # Array with K rows (for number of iterations)
                                         # F columns (one for each comparison variable), and 
                                         # two theta values vectors in each cell (Theta_M and Theta_U 
                                         # vectors of length L_f)
    #fills dirichlet parameters for theta_M  or theta_U depending on if theta_M == True or False

    def alpha_fill(k: int, theta_M: bool) -> np.array: 
        a_lst = []
        for l in range(num_agreement_lvls):
            a_kl = 0 
            for a in range(N_a): 
                for b in range(N_b): 
                    #a_kl += (comparison_arrays[k, N_b* a + b] != None)*(comparison_arrays[k, N_b* a + b] == L_k[l])*(C[N_b* a + b] == theta_M)
                    a_kl += (comparison_arrays[k, N_b* a + b] == round(L_k[l],1))*(C[N_b* a + b] == theta_M)
            a_lst.append(a_kl + M_alpha_priors[l])
        m_alphas = np.array(a_lst)
        return m_alphas
    
     # TODO: make neat: 
    def likelihood_ratio(a, b) -> float: 
        m_lh = 1
        u_lh = 1
        for k in range(K): 
            lvl = comparison_arrays[k, N_b* a + b]

            theta_mkl = theta_values[t, k, 0, int((num_agreement_lvls -1)*lvl)]
            m_lh = m_lh * theta_mkl

            theta_ukl = theta_values[t, k, 1, int((num_agreement_lvls -1)*lvl)]
            u_lh = u_lh * theta_ukl
        
        lr = m_lh/u_lh 
        return lr

    for t in range(T):
        #Step 1: sampling thetas 
        for gamma_col in range(K):
            ## Sampling for Theta_M Values:
            M_alpha_vec = alpha_fill(gamma_col, True)
            theta_values[t,gamma_col, 0] = np.random.dirichlet(M_alpha_vec)
            ## Sampling for Theta_U Values:
            U_alpha_vec = alpha_fill(gamma_col, False)
            theta_values[t,gamma_col, 1] = np.random.dirichlet(U_alpha_vec)

        #Step 2: sampling C
        #C[t+1]: empty
        C = np.full((N_a*N_b), 0)

        row_order_list = ([a for a in range(N_a)])
        #np.random.shuffle(row_order_list)
        for a in row_order_list: 
            # print(a)
            # list of length N_b: for each b, elt is b's index if b does not have a link, else None
            b_links = lambda  b : [C[N_b* a_n + b]  for a_n in range(N_a)]
            b_link_status = [b if sum(b_links(b))  == 0 else None for b in range(N_b)]
            # # list of indices of unlinked b files: we will be choosing a link between file a and one of these unlinked bs
            b_unlinked = list(filter(lambda x: x != None, b_link_status)) 
            num_links = N_b - len(b_unlinked)
            
            #if there are no more unlinked bs, we just go on to next iteration of the sampler 
            if(b_unlinked == []): 
                break
            
            prob_no_link = (N_a - num_links)*(N_b - num_links)/(num_links + 1)
            num = [likelihood_ratio(a, b) for b in b_unlinked]
            num.append(prob_no_link)
            
            denom = [sum(num)] * len(num)
            link_probs = [i / j for i, j in zip(num, denom)]

            #samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(link_probs))], 1, True, link_probs))[0]   
            
            #last index in index list == no_link. if it selected a valid index, we want 
            if(new_link_index != len(b_unlinked)):   
                # print(a, b_unlinked[new_link_index], link_probs[new_link_index])
                C[N_b*a + b_unlinked[new_link_index]] = 1
    
    C_return = pd.DataFrame(index=range(N_a), columns=range(N_b))
    for a in range(N_a):
        for b in range(N_b):
            C_return.iat[a, b] = C[N_b*a +b]
    return(theta_values,C_return)

test_comp_array = fill_comparison_arrays(A,B)
theta_values = theta_and_c_sampler(test_comp_array, 10)
# print("Theta Values:")
# print(theta_values[0])
print("C Structure:")
print(theta_values[1])