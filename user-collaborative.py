import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import loadData, filterByRatingCount

file = 'test_data_long'
#file = 'combined_data_1'

df_rating = loadData(file, 'txt')

df_rating = filterByRatingCount(df_rating, 30, 200)

# print('Fraction of values known: {:.2f}'.format(df_rating.shape[0] / (df_rating['movie'].nunique() * df_rating['customer'].nunique())))

# =========================================
#           Initialize tables 
# =========================================

# Create a DataFrame with information for each customer
means = df_rating.groupby('customer')['rating'].mean()
counts = df_rating.groupby('customer')['rating'].count()
df_customers = pd.concat([means, counts], axis=1)
df_customers.columns = ['average','count']


# Create a DataFrame of customers and movies
table_rating = df_rating.pivot_table(index='customer', columns='movie', values='rating')

# Create a DataFrame of correlations between users
table_corr = table_rating.T.corr()

table_overlap = (table_rating.notna().astype(int)).dot(table_rating.notna().astype(int).T)


# =========================================
# Predict ratings with collaborative filtering
# =========================================

def predictRating(user, movie, k=5):
    """
       Predict rating of user for movie. 
    
       Returns:
       prediction: Weighted average rating of k other customers with the highest correlations in existing ratings.
    """
    S = table_rating[table_rating[movie].notna()].index  # users who have rated movie
    if user in S:
        S = S.drop(user)  # in case user already rated the movie
    sim = table_corr.loc[user]  # correlations between user and others
    sim = sim.loc[S]
    sim = sim.sort_values(ascending=False,na_position='last')  # sort descending. not interested in users who don't have movies in common with user
    sim = sim.iloc[:k]
    Sk = sim.index  # identifiers of users with largest correlation between user
    r = table_rating.loc[Sk][movie]  # ratings of Sk
    rc = (r-df_customers.loc[Sk]['average'])  # ratings of Sk centered around average of Sk
    prediction = df_customers.loc[user]['average'] + (sim * rc).sum() / (sim.abs().sum())
    return prediction


def getPredictions(user):
    """
       Compute predictions for ratings of user for all movies.
    
       Parameters:
       user (int): Customer identifier
    
       Returns:
       df_prediction: DataFrame with movies as the index, 
        df_prediction['data'] storing existing ratings, 
        df_prediction['pred'] containing predictions
    """
    # Initialize a DataFrame with movies as the index
    df_prediction = table_rating.loc[user].to_frame()
    df_prediction.columns = ['data']
    df_prediction['pred'] = float('nan')

    movies = table_rating.columns
    for movie in movies:
        # // remember to not use chained indexing when setting
        df_prediction.loc[movie, 'pred'] = predictRating(user, movie)
    return df_prediction

# user = 57633
user = 786312
df_prediction = getPredictions(user)
print(df_prediction['data'].corr(df_prediction['pred']))


    


