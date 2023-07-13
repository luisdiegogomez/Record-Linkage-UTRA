import numpy as np

import pandas as pd
import scipy as scipy
import math as math

c = np.full((3, 4), 0)
c[0, 3] = 1
c[1, 1] = 1
c[2, 0] = 1 


print(c[1, ])

#Step 2: sampling C

        #helper function computing the likelihood of an observed gamma vector given theta vals from step 1 of iteration. 
        # def gamma_ab_likelihood(a, b, theta_type) -> float: 
        #     p = 1
        #     for k in range(K): 
        #         #since the dirichlet returns vector of length 2 with (theta_m, 1-theta_m) for each k (this is what we are assuming), we have: 
        #         #theta_mk_l0: probability of agreement level 0 for field k when records match (theta_type = 0 = m) or don't match (theta_type = 1 = u)
        #         #theta_mk_l1: probability of agreement level 1 for field k when records match (theta_type = 0 = m) or don't match (theta_type = 1 = u)
        #         theta_k_l0 = theta_values[k, i, theta_type][0]
        #         #print(theta_k_l0)
        #         theta_k_l1 = theta_values[k, i, theta_type][1]
                
        #         #p: likelihood of observing gamma vector ab given theta 
        #         p = p * (theta_k_l1)**comparison_arrays[a, b, k] * (theta_k_l0)**(1-comparison_arrays[a, b, k])

        #         return p 

                
        # for a in range(N_a):
        #     for b in range(N_b):
        #         #likelihood ratio for specific pair 
        #         lr = gamma_ab_likelihood(a, b, 0)/gamma_ab_likelihood(a, b, 1)
                
        #         # Not sure what prob_theta is supposed to be 
        #         prob_Theta = 0.5
        #         #scipy.stats.dirichlet.pdf(np.array([alpha_M_0, alpha_M_1])) * scipy.stats.dirichlet.pdf(np.array([alpha_U_0, alpha_U_1]))
        #         # sum of likelihood ratios across all bs (for all js in [1, B]) - added only when there is not a link in the previous iteration's C strucutre at the pair aj 
        #         sum_lr_across_bs = 0 
        #         for j in range(N_b) :
        #             if (C[a, j] == 0): 
        #                 sum_lr_across_bs += (gamma_ab_likelihood(a, j, 0)/gamma_ab_likelihood(a, j, 1) + prob_Theta) 

        #         prob_link_ab = lr/sum_lr_across_bs
        #         if prob_link_ab > 1: 
        #             prob_link_ab = 1
        #         if prob_link_ab < 0: 
        #             prob_link_ab = 0
                
        #         C[a, b] = np.random.binomial(1, prob_link_ab) 