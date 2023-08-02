import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import loadData, filterByRatingCount, loadMovies

np.random.seed(42)

file = 'test_data_long'
#file = 'combined_data_1'

# =========================================
#           Initialize DataFrames 
# =========================================

df_rating = loadData(file, 'txt')

df_rating = filterByRatingCount(df_rating, 30, 200)   

# Create a DataFrame of customers and movies
table_rating = df_rating.pivot_table(index='customer', columns='movie', values='rating')


# =========================================
#         Compute SVD
# =========================================

def fill_nan_with_row_avg(row):
    row[row.isna()] = row.mean()
    return row

# initialize unspecified values as user average
tr = table_rating.apply(fill_nan_with_row_avg, axis=1)
row_means = tr.mean(axis=1)
tr = tr.sub(row_means,axis=0)

U, S, VT = np.linalg.svd(tr,full_matrices=False)
treshold = 0.2
diagS = np.diag(S * (S > treshold))  # remove small eigenvalues to avoid overfitting
USVT = np.dot( np.dot(U,diagS), VT)

predictions = pd.DataFrame(USVT,index=tr.index)
predictions.columns = tr.columns
predictions = predictions.add(row_means, axis=0)

fröbenius_norm = ((table_rating - predictions)**2).sum().sum()