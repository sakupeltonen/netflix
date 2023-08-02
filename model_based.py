import pandas as pd
import numpy as np


# =========================================
#         Compute SVD
# =========================================

def completeMatrixSVD(table_rating, treshold=0.2):
    def fill_nan_with_row_avg(row):
        row[row.isna()] = row.mean()
        return row

    # initialize unspecified values as user average
    tr = table_rating.copy()
    tr = tr.apply(fill_nan_with_row_avg, axis=1)
    row_means = tr.mean(axis=1)
    tr = tr.sub(row_means,axis=0)

    U, S, VT = np.linalg.svd(tr,full_matrices=False)

    diagS = np.diag(S * (S > treshold))  # remove small eigenvalues to avoid overfitting
    USVT = np.dot( np.dot(U,diagS), VT)

    predictions = pd.DataFrame(USVT,index=tr.index)
    predictions.columns = tr.columns
    predictions = predictions.add(row_means, axis=0)

    # fröbenius_norm = ((table_rating - predictions)**2).sum().sum()

    return predictions