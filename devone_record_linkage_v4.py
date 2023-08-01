### DEVONE IMPLEMENTATION V4: Structure to contain number of instances of l disagreement occurring

import numpy as np

import pandas as pd

import math as math

# pip install jaro-winkler
import jaro

import time
start_time = time.time()

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
L_k_n = len(L_k) # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

comparison_arrays = np.full((K, (N_a*N_b)), fill_value = 0, dtype= float)
base_l_instaces = np.full((K, L_k_n, 2), fill_value=0, dtype=int)

#Returns jaro_winkler_distance of two strings
def jaro_winkler_distance(s1, s2):
    jaro_winkler = round(jaro.jaro_winkler_metric(s1,s2),1)

    if jaro_winkler > 1: 
        jaro_winkler = 1.0
    
    return jaro_winkler

## Filling in Comparison Vectors (Gamma Vectors):
def fill_comparison_arrays():
    # Filling comparison vectors:
    for a in range(N_a):
        for b in range(N_b):
            for k in range(K):
                distance = jaro_winkler_distance(str(X_a.iat[a,k]), str(X_b.iat[b,k]))
                comparison_arrays[k, ((N_b*a) + b)] = distance
                base_l_instaces[k,int((10*distance)),0] += 1

#Gibbs sampler 
def theta_and_c_sampler(T:int) -> np.ndarray:
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    M_alpha_priors = np.full(L_k_n, 1, dtype=int)
    U_alpha_priors = np.full(L_k_n, 1, dtype=int)
    
    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((T, K, 2, L_k_n), 0.00, dtype=float) # Array with K rows (for number of iterations)
                                         # F columns (one for each comparison variable), and 
                                         # two theta values vectors in each cell (Theta_M and Theta_U 
                                         # vectors of length L_f)
    C = np.full((N_a*N_b), 0)
    temp_l_instances = np.full((K, L_k_n, 2), fill_value=0, dtype=int)
    temp_l_instances[:,:,:] = base_l_instaces[:,:,:]

    #fills dirichlet parameters for theta_M  or theta_U depending on if theta_M == True or False
    def alpha_fill(k: int, theta_type: bool, l_instances: np.ndarray) -> np.ndarray: 
        a_lst = []
        for l in range(L_k_n): 
            if theta_type: 
                a_lst.append((l_instances[k,l,1] + M_alpha_priors[l]))
            else: 
                a_lst.append((l_instances[k,l,0] + U_alpha_priors[l]))
        alpha_params = np.array(a_lst)
        return alpha_params
    
    def likelihood_ratio(a, b) -> float: 
        m_lh = 1
        u_lh = 1
        for k in range(K): 
            lvl = comparison_arrays[k, N_b* a + b]

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
            M_alpha_vec = alpha_fill(k, True, temp_l_instances)
            theta_values[t,k,0] = np.random.dirichlet(M_alpha_vec)
            ## Sampling for Theta_U Values:
            U_alpha_vec = alpha_fill(k, False, temp_l_instances)
            theta_values[t,k,1] = np.random.dirichlet(U_alpha_vec)

        #Step 2: sampling C
        C = np.full((N_a*N_b), 0)
        temp_l_instances[:,:,:] = base_l_instaces[:,:,:]

        row_order_list = ([a for a in range(N_a)])
        np.random.shuffle(row_order_list)

        for a in row_order_list: 
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
                C[N_b*a + b_unlinked[new_link_index]] = 1
                for k in range(K):
                    temp_l_instances[k,int(10*comparison_arrays[k,(N_b*a + b_unlinked[new_link_index])]),0] -= 1
                    temp_l_instances[k,int(10*comparison_arrays[k,(N_b*a + b_unlinked[new_link_index])]),1] += 1

    return(C)

fill_comparison_arrays()

print("Comparison Arrays:")
print(comparison_arrays)
print("L Instances:")
print(base_l_instaces)

C = theta_and_c_sampler(100)

C_dataframe = pd.DataFrame(index=range(N_a), columns=range(N_b))
for a in range(N_a):
    for b in range(N_b):
        C_dataframe.iat[a, b] = C[N_b*a + b]

print("C Structure:")
print(C_dataframe)
print("--- %s seconds ---" % (time.time() - start_time))