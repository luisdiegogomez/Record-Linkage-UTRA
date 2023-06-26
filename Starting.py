import numpy as np

import pandas as pd

# Make sure file paths are based on wherever your files are locally 
A = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/TestCSV.csv")
B = pd.read_csv("~/OneDrive/Documents/R/Record-Linkage-UTRA/Test CSV 2.csv")

# Function that outputs pandas DataFrame (dimensions N_a,N_b) representing the comparison gamma 
# vectors for each pair of records between files A and B
def fill_comparison_arrays(recordA:pd.DataFrame,recordB:pd.DataFrame) -> pd.DataFrame:
    N_a = len(recordA.index)
    N_b = len(recordB.index)
  
    X_a = recordA[np.sort(recordA.columns.intersection(recordB.columns))]
    X_b = recordB[np.sort(recordB.columns.intersection(recordA.columns))]
  
    K = len(X_a.columns)

    # Initializing matrix of comparison gamma vectors
    comparison_arrays = np.full((N_a, N_b, K), fill_value = False)

    # Filling comparison vectors
    for a in range(N_a):
        for b in range(N_b):
            for col in range(K):
                if X_a.iat[a,col] == X_b.iat[b,col]:
                    comparison_arrays[a,b,col] = True

                else:
                    comparison_arrays[a,b,col] = False

    # Converting the matrix of comparison vectors to a pandas DataFrame
    return_comparison_arrays = pd.DataFrame(index=range(N_a), columns=range(N_b))
    for a in range(N_a):
        for b in range(N_b):
            return_comparison_arrays.iat[a, b] = comparison_arrays[a,b]

    return(return_comparison_arrays)

print(fill_comparison_arrays(A,B))