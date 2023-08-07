### DEVONE IMPLEMENTATION V5: Power prior and l-instances strucutre implementation

import numpy as np

import pandas as pd

# pip install jaro-winkler
import jaro

import time
start_time = time.time()

## Initilizating Datasets From CSV Files:

# Make sure file paths are based on wherever your files are locally 
# A_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv1.csv")
# B_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv1.csv")

# Make sure file paths are based on wherever your files are locally 
# DONT DELETE JUST COMMENT OUT 
A_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\2015 Shortened.csv")
B_temp = pd.read_csv(r"C:\Users\efiaa\OneDrive\Documents\Record-Linkage-UTRA\2015 Shortened.csv")


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

L_k_n = 11 # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

comparison_arrays = np.full((K, (N_a*N_b)), fill_value = 0, dtype= float)
# Order of stored l instances: known matches (0), known non-matches (1), unknown matches (2), unknown non-matches (3)
base_l_instances = np.full((K, L_k_n, 4), fill_value=0, dtype=int)

C_init = np.full(((N_a * N_b), 2), 0)

## Functions:
# Returns jaro_winkler_distance of two strings
def jaro_winkler_distance(s1, s2):
    jaro_winkler = round(jaro.jaro_winkler_metric(s1,s2),1)

    if jaro_winkler > 1: 
        jaro_winkler = 1.0
    
    return jaro_winkler

# Filling in Comparison Vectors (Gamma Vectors):
def fill_comparison_arrays():
    # Filling comparison vectors:
    for a in range(N_a):
        for b in range(N_b):
            for k in range(K):
                if str(X_a.iat[a,k]) != "" and str(X_b.iat[b,k]) != "":
                    distance = jaro_winkler_distance(str(X_a.iat[a,k]), str(X_b.iat[b,k]))
                    comparison_arrays[k, ((N_b*a) + b)] = distance
                    # Known match counter
                    if (C_init[N_b*a + b, 0] == 1) and (C_init[N_b*a + b, 1] == 1):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),0] += 1
                    # Known non-match counter
                    elif (C_init[N_b*a + b, 0] == 0) and (C_init[N_b*a + b, 1] == 1):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),1] += 1
                    # Unknown match counter  
                    elif (C_init[N_b*a + b, 0] == 1) and (C_init[N_b*a + b, 1] == 0):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),2] += 1
                    # Unknown non-match counter  
                    elif (C_init[N_b*a + b, 0] == 0) and (C_init[N_b*a + b, 1] == 0):
                        base_l_instances[k,int(((L_k_n - 1)*distance)),3] += 1
                else:
                    comparison_arrays[k, ((N_b*a) + b)] = None
                
# Gibbs Sampler 
def theta_and_c_sampler(T:int, alpha: float):
    C = np.full(((N_a * N_b), 2), 0)
    C[:,:] = C_init[:,:]
    #Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    M_alpha_priors = np.full(L_k_n, 1, dtype=int)
    U_alpha_priors = np.full(L_k_n, 1, dtype=int)
    ## Gibbs Sampler for Theta Values:
    theta_values = np.full((T, K, 2, L_k_n), 0.00, dtype=float) # Array with K rows (for number of iterations)
                                                                # F columns (one for each comparison variable), and 
                                                                # two theta values vectors in each cell (Theta_M and Theta_U 
                                                                # vectors of length L_f)
    temp_l_instances = np.full((K, L_k_n, 4), fill_value=0, dtype=int)
    temp_l_instances[:,:,:] = base_l_instances[:,:,:]

    #fills dirichlet parameters for theta_M  or theta_U depending on if theta_M == True or False
    def alpha_fill(k: int, theta_type: bool,) -> np.ndarray: 
        a_lst = []
        for l in range(L_k_n): 
            if theta_type:
                a_lst.append((temp_l_instances[k,l,0]**alpha + temp_l_instances[k,l,2] + M_alpha_priors[l]))
            else: 
                a_lst.append((temp_l_instances[k,l,1]**alpha + temp_l_instances[k,l,3] + U_alpha_priors[l]))
        alpha_params = np.array(a_lst)
        return alpha_params
    
    def likelihood_ratio(a, b) -> float: 
        m_lh = 1
        u_lh = 1
        for k in range(K): 
            lvl = comparison_arrays[k, int(N_b* a + b)]

            if pd.notna(lvl):
                theta_mkl = theta_values[t, k, 0, int((L_k_n-1)*lvl)]
                theta_ukl = theta_values[t, k, 1, int((L_k_n-1)*lvl)]
            else:
                theta_mkl = 1
                theta_ukl = 1
            
            m_lh *= theta_mkl
            u_lh *= theta_ukl
        
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
        temp_l_instances[:,:,:] = base_l_instances[:,:,:]

        row_order_list = ([a for a in range(N_a)])
        np.random.shuffle(row_order_list)
        for a in row_order_list: 
            # indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 0 (unknown)
            unlinked_unknown_pairs = np.nonzero((C[:,0] == 0) & (C[:,1] == 0))[0]
            # indices of C where C[i, 0] == 0 (nonlink) and C[i, 1] == 1 (unknown)
            unlinked_known_pairs = np.nonzero((C[:,0] == 0) & (C[:,1] == 1))[0]
            
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

            # samples b_unlinked index from the , creates a new link at that b with probability associated with that  b 
            new_link_index = (np.random.choice([i for i in range(len(link_probs))], 1, True, link_probs_normalized))[0]   
            
            # last index in index list == no_link. if it selected a valid index, we want 
            if(new_link_index != len(b_unlinked_unknown)):   
                C[N_b*a + b_unlinked_unknown[new_link_index], 0] = 1
                for k in range(K):
                    if pd.notna(comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]):
                        temp_l_instances[k,int((L_k_n - 1)*comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]),3] -= 1
                        temp_l_instances[k,int((L_k_n - 1)*comparison_arrays[k,(N_b*a + b_unlinked_unknown[new_link_index])]),2] += 1  
    
    return(C, theta_values)

def C_matrix_to_df(C): 
    C_dataframe = pd.DataFrame(index=range(N_a), columns=range(N_b))
    for a in range(N_a):
        for b in range(N_b):
            C_dataframe.iat[a, b] = C[N_b*a + b, 0]
    return C_dataframe

fill_comparison_arrays()

c_and_theta_vals = theta_and_c_sampler(1000, 1)

c_df = C_matrix_to_df(c_and_theta_vals[0])
#C_matrix_to_df(c_and_theta_vals[0]).to_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/test_csv.csv")

comparison_df = pd.DataFrame(index=range(N_a), columns=range(N_a))
for a in range(N_a):
        for b in range(N_b):
            if a == b:
                comparison_df.iat[a, b] = 1
            else:
                comparison_df.iat[a, b] = 0

counter = 0

for a in range(N_a):
    if comparison_df.iat[a, a] == 1 and c_df.iat[a,a] == 1:
        counter += 1
correct_percentage = counter/(N_a)

print(c_df)
print("Accuracy:")
print(correct_percentage)

#c_df.to_csv('2015 Shortened C-Structure.csv')

print("--- %s seconds ---" % (time.time() - start_time))