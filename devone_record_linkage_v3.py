



### DEVONE IMPLEMENTATION V2: L instead of 2 agreement levels (determined by jaro winkler)

import numpy as np

import pandas as pd

import math as math

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
A_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\generated_csv1.csv")
B_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\generated_csv2.csv")

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
L_k_n = len(L_k) # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

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
def fill_comparison_arrays() -> np.ndarray:
    # Filling comparison vectors:
    comparison_arrays = np.full((K, (N_a*N_b)), fill_value = 0, dtype= float) 
    for a in range(N_a):
        for b in range(N_b):
            for k in range(K):
                comparison_arrays[k, ((N_b*a) + b)] = jaro_winkler_distance(str(X_a.iat[a,k]), str(X_b.iat[b,k]))
    return comparison_arrays


#Gibbs sampler 
def theta_and_c_sampler(comparison_arrays:np.ndarray, T:int, C_init: np.ndarray, alpha: float):
    C = C_init
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    M_alpha_priors = np.full(L_k_n, 1, dtype=int)
    U_alpha_priors = np.full(L_k_n, 1, dtype=int)
    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((T, K, 2, L_k_n), 0.00, dtype=float) # Array with K rows (for number of iterations)
                                         # F columns (one for each comparison variable), and 
                                         # two theta values vectors in each cell (Theta_M and Theta_U 
                                         # vectors of length L_f)

    #fills dirichlet parameters for theta_M  or theta_U depending on if theta_M == True or False
    def alpha_fill(k: int, theta_type: bool) -> np.ndarray: 
        a_lst = []
        for l in range(L_k_n):
            a_kl = 0 
            for a in range(N_a): 
                for b in range(N_b): 
                    a_kl += (comparison_arrays[k, N_b* a + b] == round(L_k[l],1))*(C[N_b* a + b, 0] == theta_type)
            if theta_type: a_lst.append((a_kl + M_alpha_priors[l]))
            else: a_lst.append((a_kl + U_alpha_priors[l]))
        alpha_params = np.array(a_lst)
        return alpha_params
    
    def likelihood_ratio(a, b) -> float: 
        m_lh = 1
        u_lh = 1
        for k in range(K): 
            lvl = comparison_arrays[k, int(N_b* a + b)]

            theta_mkl = theta_values[t, k, 0, int((L_k_n -1)*lvl)]
            m_lh = m_lh * theta_mkl

            theta_ukl = theta_values[t, k, 1, int((L_k_n -1)*lvl)]
            u_lh = u_lh * theta_ukl
        
        lr = m_lh/u_lh 
        return lr

    for t in range(T):
        #Step 1: sampling thetas 
        for k in range(K):
            ## Sampling for Theta_M Values:
            M_alpha_vec = alpha_fill(k, True)
            theta_values[t,k, 0] = np.random.dirichlet(M_alpha_vec)
            ## Sampling for Theta_U Values:
            U_alpha_vec = alpha_fill(k, False)
            theta_values[t,k, 1] = np.random.dirichlet(U_alpha_vec)

        #Step 2: sampling C
        #C[t+1]: for all unknown (ie unfixed) pairs, set link value to 0 
        C[:,0]= np.where(C[:,1] == 0, 0, C[:,0]) 

        #indices of C where C[i, 1] == 1 (known)
        # known_pairs = np.nonzero((C[:,1] == 1))[0]
        # # (N_b*a + b) mod N_b returns b index of pair; ((N_b*a + b) - b)/N_a returns a index of pair 
        # known_pair_bs = known_pairs % N_b
        # known_pair_as = (known_pairs - known_pair_bs)/N_b

        # if len(known_pairs) != 0:    
        #     #joint likelihood ratio for all known pairs
        #     prior_likelihood = math.prod([likelihood_ratio(a, b) for a, b in zip(known_pair_as, known_pair_bs)])
        #     power_prior = prior_likelihood ** alpha
        # else: 
        #     power_prior = 0 

        row_order_list = ([a for a in range(N_a)])
        np.random.shuffle(row_order_list)
        for a in row_order_list: 
            #indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 0 (unknown)
            unlinked_unknown_pairs = np.nonzero((C[:,0] == 0) & (C[:,1] ==0))[0]
            #indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 1 (unknown)
            unlinked_known_pairs = np.nonzero((C[:,0] == 0) & (C[:,1] ==1))[0]
            
            # (N_b*a + b) mod N_b returns b index of pair
            b_unlinked_unknown = list(set(unlinked_unknown_pairs % N_b))
            b_unlinked_known =list(set(unlinked_known_pairs % N_b))

            num_links = N_b - len(b_unlinked_unknown) - len(b_unlinked_known)
            
            #if there are no more unlinked bs, we just go on to next iteration of the sampler 
            if(b_unlinked_unknown == []): 
                break
            
            prob_no_link = (N_a - num_links)*(N_b - num_links)/(num_links + 1)
            num = [likelihood_ratio(a, b) for b in b_unlinked_unknown]
            num.append(prob_no_link)
            
            #TODO: CHECK: power prior implementation 
            denom = [sum(num)] * len(num)
            link_probs = [i / j for i, j in zip(num, denom)]
            link_probs_sum = sum(link_probs)
            link_probs_normalized = link_probs/link_probs_sum

            #samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(link_probs))], 1, True, link_probs_normalized))[0]   
            
            #last index in index list == no_link. if it selected a valid index, we want 
            if(new_link_index != len(b_unlinked_unknown)):   
                C[N_b*a + b_unlinked_unknown[new_link_index], 0] = 1  
    
    return(C, theta_values)

def C_matrix_to_df(C): 
    C_dataframe = pd.DataFrame(index=range(N_a), columns=range(N_b))
    for a in range(N_a):
        for b in range(N_b):
            C_dataframe.iat[a, b] = C[N_b*a +b, 0]
    return C_dataframe

comparison_arrays = fill_comparison_arrays()
C_init = np.full(((N_a * N_b), 2), 0)

C_init_prior = np.full(((N_a * N_b), 2), 0)
C_init_prior[0,0] = 1 
C_init_prior[0,1] = 1 
C_init_prior[(N_b * 1 + 1), 0] = 1
C_init_prior[(N_b * 1 + 1), 1] = 1
C_init_prior[(N_b * 3 + 7), 0] = 1
C_init_prior[(N_b * 3 + 7), 1] = 0


c_and_theta_vals = theta_and_c_sampler(comparison_arrays, 100, C_init_prior, 1)

print("C Structure:")
print(C_matrix_to_df(c_and_theta_vals[0]))

