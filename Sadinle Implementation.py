import numpy as np

import pandas as pd

import math as mt

'''This implementation of record linkage is implemented from Bayesian Estimation of Bipartite Matchings
for Record Linkage by Sandile (2016)'''

## Initilizating Datasets From CSV Files and Global Variables:

# Make sure file paths are based on wherever your files are locally:
X_1_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv1.csv")
#dtype={'_STATE':str,'FMONTH':str,'IDATE':str,'IMONTH':str,'IDAY':str,'IYEAR':str,'DISPCODE':str,'SEQNO':str,'_PSU':str,'CTELENUM':str,'PVTRESD1':str,'COLGHOUS':str,'STATERES':str,'CELLFON3':str,'LADULT':str,'NUMADULT':str,'NUMMEN':str,'NUMWOMEN':str,'CTELNUM1':str,'CELLFON2':str,'CADULT':str,'PVTRESD2':str,'CCLGHOUS':str,'CSTATE':str,'LANDLINE':str,'HHADULT':str,'GENHLTH':str,'PHYSHLTH':str,'MENTHLTH':str,'POORHLTH':str,'HLTHPLN1':str,'PERSDOC2':str,'MEDCOST':str,'CHECKUP1':str,'BPHIGH4':str,'BPMEDS':str,'BLOODCHO':str,'CHOLCHK':str,'TOLDHI2':str,'CVDINFR4':str,'CVDCRHD4':str,'CVDSTRK3':str,'ASTHMA3':str,'ASTHNOW':str,'CHCSCNCR':str,	CHCOCNCR:str,	CHCCOPD1:str,	HAVARTH3:str,	ADDEPEV2:str,	CHCKIDNY:str,	DIABETE3:str,	DIABAGE2:str,	SEX:str,	MARITAL:str,	EDUCA:str,	RENTHOM1:str,	NUMHHOL2:str,	NUMPHON2:str,	CPDEMO1:str,	VETERAN3:str,	EMPLOY1:str,	CHILDREN:str,	INCOME2:str,	INTERNET:str,	WEIGHT2:str,	HEIGHT3:str,	PREGNANT:str,	QLACTLM2:str,	USEEQUIP:str,	BLIND:str,	DECIDE:str,	DIFFWALK:str,	DIFFDRES:str,	DIFFALON	SMOKE100	SMOKDAY2	STOPSMK2	LASTSMK2	USENOW3	ALCDAY5	AVEDRNK2	DRNK3GE5	MAXDRNKS	FRUITJU1	FRUIT1	FVBEANS	FVGREEN	FVORANG	VEGETAB1	EXERANY2	EXRACT11	EXEROFT1	EXERHMM1	EXRACT21	EXEROFT2	EXERHMM2	STRENGTH	LMTJOIN3	ARTHDIS2	ARTHSOCL	JOINPAIN	SEATBELT	FLUSHOT6	FLSHTMY2	IMFVPLAC	PNEUVAC3	HIVTST6	HIVTSTD3	WHRTST10	PDIABTST	PREDIAB1	INSULIN	BLDSUGAR	FEETCHK2	DOCTDIAB	CHKHEMO3	FEETCHK	EYEEXAM	DIABEYE	DIABEDU	PAINACT2	QLMENTL2	QLSTRES2	QLHLTH2	CAREGIV1	CRGVREL1	CRGVLNG1	CRGVHRS1	CRGVPRB1	CRGVPERS	CRGVHOUS	CRGVMST2	CRGVEXPT	VIDFCLT2	VIREDIF3	VIPRFVS2	VINOCRE2	VIEYEXM2	VIINSUR2	VICTRCT4	VIGLUMA2	VIMACDG2	CIMEMLOS	CDHOUSE	CDASSIST	CDHELP	CDSOCIAL	CDDISCUS	WTCHSALT	LONGWTCH	DRADVISE	ASTHMAGE	ASATTACK	ASERVIST	ASDRVIST	ASRCHKUP	ASACTLIM	ASYMPTOM	ASNOSLEP	ASTHMED3	ASINHALR	HAREHAB1	STREHAB1	CVDASPRN	ASPUNSAF	RLIVPAIN	RDUCHART	RDUCSTRK	ARTTODAY	ARTHWGT	ARTHEXER	ARTHEDU	TETANUS	HPVADVC2	HPVADSHT	SHINGLE2	HADMAM	HOWLONG	HADPAP2	LASTPAP2	HPVTEST	HPLSTTST	HADHYST2	PROFEXAM	LENGEXAM	BLDSTOOL	LSTBLDS3	HADSIGM3	HADSGCO1	LASTSIG3	PCPSAAD2	PCPSADI1	PCPSARE1	PSATEST1	PSATIME	PCPSARS1	PCPSADE1	PCDMDECN	SCNTMNY1	SCNTMEL1	SCNTPAID	SCNTWRK1	SCNTLPAD	SCNTLWK1	SXORIENT	TRNSGNDR	RCSGENDR	RCSRLTN2	CASTHDX2	CASTHNO2	EMTSUPRT	LSATISFY	ADPLEASR	ADDOWN	ADSLEEP	ADENERGY	ADEAT1	ADFAIL	ADTHINK	ADMOVE	MISTMNT	ADANXEV	QSTVER	QSTLANG	EXACTOT1	EXACTOT2	MSCODE	_STSTR	_STRWT	_RAWRAKE	_WT2RAKE	_CHISPNC	_CRACE1	_CPRACE	_CLLCPWT	_DUALUSE	_DUALCOR	_LLCPWT	_RFHLTH	_HCVU651	_RFHYPE5	_CHOLCHK	_RFCHOL	_MICHD	_LTASTH1	_CASTHM1	_ASTHMS1	_DRDXAR1	_PRACE1	_MRACE1	_HISPANC	_RACE	_RACEG21	_RACEGR3	_RACE_G1	_AGEG5YR	_AGE65YR	_AGE80	_AGE_G	HTIN4	HTM4	WTKG3	_BMI5	_BMI5CAT	_RFBMI5	_CHLDCNT	_EDUCAG	_INCOMG	_SMOKER3	_RFSMOK3	DRNKANY5	DROCDY3_	_RFBING5	_DRNKWEK	_RFDRHV5	FTJUDA1_	FRUTDA1_	BEANDAY_	GRENDAY_	ORNGDAY_	VEGEDA1_	_MISFRTN	_MISVEGN	_FRTRESP	_VEGRESP	_FRUTSUM	_VEGESUM	_FRTLT1	_VEGLT1	_FRT16	_VEG23	_FRUITEX	_VEGETEX	_TOTINDA	METVL11_	METVL21_	MAXVO2_	FC60_	ACTIN11_	ACTIN21_	PADUR1_	PADUR2_	PAFREQ1_	PAFREQ2_	_MINAC11	_MINAC21	STRFREQ_	PAMISS1_	PAMIN11_	PAMIN21_	PA1MIN_	PAVIG11_	PAVIG21_	PA1VIGM_	_PACAT1	_PAINDX1	_PA150R2	_PA300R2	_PA30021	_PASTRNG	_PAREC1	_PASTAE1	_LMTACT1	_LMTWRK1	_LMTSCL1	_RFSEAT2	_RFSEAT3	_FLSHOT6	_PNEUMO2	_AIDTST3}) 
                                                                                      # Equivalent to A in DeVone Paper, 
                                                                                      # individual record indexed by i
X_2_temp = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/generated_csv2.csv") # Equivalent to B in DeVone Paper, 
                                                                                      # individual record indexed by j

# for r in range(X_1_temp.shape[0]):
#     for c in range(X_1_temp.shape[1]):
#         X_1_temp.iat[r,c] = str(X_1_temp.iat[r,c])

# for r in range(X_2_temp.shape[0]):
#     for c in range(X_2_temp.shape[1]):
#         X_2_temp.iat[r,c] = str(X_2_temp.iat[r,c])

## Global Variables:
if len(X_1_temp.index) >= len(X_2_temp.index):
    X_1 = X_1_temp
    X_2 = X_2_temp
    n_1 = len(X_1.index) # Equivalent to N_a
    n_2 = len(X_2.index) # Equivalent to N_b
else:
    X_1 = X_2_temp
    X_2 = X_1_temp
    n_1 = len(X_1.index) # Equivalent to N_a
    n_2 = len(X_2.index) # Equivalent to N_b

  
# Prior modifications to column headings in csv datasets most likely needed to 
# ensure code identifies the correct intersecting variables:
X_1f = X_1[np.sort(X_1.columns.intersection(X_2.columns))]
X_2f = X_2[np.sort(X_2.columns.intersection(X_1.columns))]
  
F = len(X_1f.columns) # Equivalent to k, 
                      # number of comparison criteria (intersecting varibles for each record) 

L_f = np.arange(0,1.01,0.01)
L_f_n = len(L_f) # Levels of disagreement (100 for 2 decimal place values of Jaro-Winkler Distance)

## Filling in Comparison Vectors (Gamma Vectors):

# Jaro-Winkler Distance for String Comparison
'''Function outputs Jaro-Winkler Distance between the two inputted strings where 0 means 
complete agreement and 1 means complete disagreement and values in between are 
rounded to the nearest 100ths place'''
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
        return 1.0

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

    jaro_winkler = jaro + (0.1 * prefix_len * (1 - jaro))

    jaro_winkler = round((-jaro_winkler + 1), 2)

    if jaro_winkler < 0:
        jaro_winkler = 0
    
    return jaro_winkler

'''Function that outputs tuple with 3-D array (dimensions n_1, n_2, F) representing the comparison gamma 
vectors for each pair of records between all files in X_1 and X_2 as first element of outputted tuple, Pandas Dataframe 
of same gamma vectors in second element of tuple, (meant for visual representation of overall gamma vector for each 
pair of records):'''
def fill_comparison_arrays_matrix() -> tuple:
    # Initializing matrix of comparison gamma vectors:
    comparison_arrays = np.full((n_1,n_2,F), fill_value = 0.00, dtype=float) # n_1 by n_2 matrix with each cell containing 
                                                                          # a gamma comparison vector (of size F) for each  
                                                                          # pair of files (indexed i,j) in X_1 and X_2

    # Filling comparison vectors:
    for i in range(n_1):
        for j in range(n_2):
            for f in range(F):
                if (X_1f.iat[i,f] == "") or (X_2f.iat[j,f] == ""):
                    comparison_arrays[i,j,f] = None
                else:
                    comparison_arrays[i,j,f] = jaro_winkler_distance(str(X_1f.iat[i,f]),str(X_2f.iat[j,f]))

    # Converting the matrix of comparison vectors to a pandas DataFrame
    dataframe_comp_arrays = pd.DataFrame(index=range(n_1), columns=range(n_2))
    for i in range(n_1):
        for j in range(n_2):
            dataframe_comp_arrays.iat[i, j] = comparison_arrays[i,j]

    return(comparison_arrays, dataframe_comp_arrays)

# Function that outputs 2-D array (dimensions F, n_1*n_2) representing the values 
# of each comparison varaible for each pair (meant for easier computations in Gibbs sampling):
def fill_comparison_arrays_f_rows() -> tuple:
    # Initializing matrix of comparison gamma vectors:
    comparison_values = np.full((F, (n_1*n_2)), fill_value = 0.00, dtype=float) # F by n_1*n_2 matrix representing the comparison 
                                                                             # value f at pair of records ij 
                                                                             # (indexed by [n_1*i + j] when indexing starts at 0) 

    # Filling comparison values:
    for i in range(n_1):
        for j in range(n_2):
            for f in range(F):
                if (X_1f.iat[i,f] != "") and (X_2f.iat[j,f] != ""):
                    comparison_values[f, (n_2*i + j)] = jaro_winkler_distance(str(X_1f.iat[i,f]),str(X_2f.iat[j,f]))
                else:
                    comparison_values[f, (n_2*i + j)] = None
                    

    ## Converting the matrix of comparison vectors to a pandas DataFrame
    comparison_values_df = pd.DataFrame(comparison_values)

    return(comparison_values, comparison_values_df)

# gamma_vector_matrices = fill_comparison_arrays_matrix()
# gamma_matrix = gamma_vector_matrices[0]
# gamma_dataframe = gamma_vector_matrices[1]
comparison_array = fill_comparison_arrays_f_rows()
# print("Gamma Vectors Matrix:")
# print(gamma_matrix)
# print("Gamma Vectors DataFrame")
# print(gamma_dataframe)
print("Comparison Array:")
print(comparison_array[0])
print("Pandas Comparison Array:")
print(comparison_array[1])

## Sampling Theta Values for Comparison Vectors:

def theta_and_c_sampler(comparison_array_in:np.ndarray, iterations_t:int) -> tuple:
    # Establishing initial parameters for the Dirchlet Distributions from which we're sampling:
    alpha_priors = np.full(L_f_n, 1, dtype=int)
    beta_priors = np.full(L_f_n, 1, dtype=int)
    # Establishing Prior for Pi (A Priori Proportion of True Matches)
    pi_params = [1,1]
    pi = np.random.beta(pi_params[0],pi_params[1])

    # Initializing array of theta values:
    theta_values = np.full((iterations_t, F, 2, L_f_n), 0.00, dtype=float) # Array with t rows (for number of iterations)
                                         # F columns (one for each comparison variable), and 
                                         # two theta values vectors in each cell (Theta_M and Theta_U 
                                         # vectors of length L_f)

    # Initilaizaing empty Z (0 matches):
    Z_values = np.full(((iterations_t+1),n_2), 0, dtype=int) # Equivalent to C
    for j in range(n_2):
        Z_values[0,j] = n_1 + j

    # Helper Functions for the Calculation of Pair Probability for Z^t+1:
    def a_fl(Z:np.ndarray, f:int, l:float) -> int:
        a_fl_Z = 0
        for i in range(n_1):
            for j in range(n_2):
                a_fl_Z += (comparison_array_in[f,(n_2*i + j)] != None)*(comparison_array_in[f,(n_2*i + j)] == L_f[l])*(Z[j] == i)
        return a_fl_Z
    
    def b_fl(Z:np.ndarray, f:float, l:float) -> int:
        b_fl_Z = 0
        for i in range(n_1):
            for j in range(n_2):
                b_fl_Z += (comparison_array_in[f,(n_2*i + j)] != None)*(comparison_array_in[f,(n_2*i + j)] == L_f[l])*(Z[j] != i)
        return b_fl_Z
    
    def n_12(Z:np.ndarray) -> int:
        n_12_Z = 0
        for j in range(n_2):
            n_12_Z += (Z[j] < n_1)
        return n_12_Z
    
    def p_qj_Z_PHI(q:int,j:int,Z:np.ndarray,PHI:np.ndarray) -> float:
        if q < n_1:
            w_qj = 0.00
            for f in range(F):
                for l in range(L_f_n):
                    w_qj += (comparison_array_in[f,(n_2*q + j)] != None)*(mt.log(PHI[f,0,l]/PHI[f,1,l]))*(comparison_array_in[f,(n_2*q + j)] == L_f[l])
            prob_q = (mt.exp(w_qj))*(q not in (np.append(Z[:j],Z[(j+1):])))
        elif q == (n_1 + j):
            n_12_Z = n_12(Z)
            prob_q = (n_1 - n_12_Z)*((n_2 - n_12_Z - 1 + pi_params[1])/(n_12_Z + pi_params[0]))
        return prob_q
        
    # Gibbs Sampler
    for t in range(iterations_t):
        # Sampling for Theta_M and Theta_U Values:
        for f in range(F):
            alpha_vec = np.full(L_f_n, 0, dtype=int)
            beta_vec = np.full(L_f_n, 0, dtype=int)
            for l in range(L_f_n):
                alpha_vec[l] = a_fl(Z_values[t],f,l) + alpha_priors[l]
                beta_vec[l] = b_fl(Z_values[t],f,l) + beta_priors[l]
            theta_values[t,f,0] = np.random.dirichlet(alpha_vec)
            theta_values[t,f,1]  = np.random.dirichlet(beta_vec)

        # Sampling for Z values (Pairings):
        label_probs = np.full((n_1+1),0.00,dtype=float)
        Z_values[(t+1)] = Z_values[t]
        for j in range(n_2):
            # Filling Probability Vector:
            for q in range(n_1):
                label_probs[q] = p_qj_Z_PHI(q,j,Z_values[(t+1)],theta_values[t])
            label_probs[n_1] = p_qj_Z_PHI((n_1+j),j,Z_values[(t+1)],theta_values[t])

            label_probs = label_probs/np.sum(label_probs)
            q_labels = np.append(np.arange(0,n_1),np.array((n_1+j)))

            # Sampling Z-Values:
            Z_values[(t+1),j] = np.random.choice(q_labels,p=label_probs)

    return(theta_values,Z_values)

output_tuple = theta_and_c_sampler(comparison_array[0],10)
# print("Theta Values:")
# print(output_tuple[0])
print("Z Values:")
print(output_tuple[1])
