import pandas as pd
import numpy as np


# =========================================
# Predict ratings with collaborative filtering
# =========================================

def predictRating(user, movie, table_rating, df_customers, table_corr, table_overlap, k=5, beta=20):
    """
       Predict rating of user for movie. 

       Parameters:
       user (int): Customer identifier
       k (int): Number of neighbors considered
       beta (int): Treshold number of common movies required from neighbors before weighing down similarity. 
    
       Returns:
       prediction: Weighted average rating of k other customers with the highest correlations in existing ratings.
    """
    # TODO tf-idf weight for movies

    S = table_rating[table_rating[movie].notna()].index  # users who have rated movie
    if user in S:
        S = S.drop(user)
    
    
    sim = table_corr.loc[user]  # correlations between user and others
    sim = sim.loc[S]
    sim = sim.sort_values(ascending=False,na_position='last')  # sort descending. not interested in users who don't have movies in common with user
    sim = sim.iloc[:k]
    Sk = sim.index  # identifiers of users with largest correlation between user

    if beta: 
        # discount sim based on number of common movies
        overlap = table_overlap.loc[user, Sk]
        weight = np.minimum(overlap,beta) / beta
        sim = sim * weight

    r = table_rating.loc[Sk][movie]  # ratings of Sk
    rc = (r-df_customers.loc[Sk]['average'])  # ratings of Sk centered around average of Sk
    prediction = df_customers.loc[user]['average'] + (sim * rc).sum() / (sim.abs().sum())
    return prediction


def getPredictions(table_rating, user,beta=20):
    # TODO delete this
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
        df_prediction.loc[movie, 'pred'] = predictRating(user, movie,beta=beta)
    return df_prediction


def completeMatrixCollaborative(users, movies, table_rating, df_customers, k=5, beta=20):
    result = pd.DataFrame(index=users, columns=movies, dtype=float)
    
    # Precompute a DataFrame of correlations between users
    table_corr = table_rating.T.corr()
    table_overlap = (table_rating.notna().astype(int)).dot(table_rating.notna().astype(int).T)

    for user in users:
        for movie in movies:
            result.loc[user,movie] = predictRating(user, movie, table_rating, df_customers, table_corr, table_overlap, k=5, beta=beta)
    return result

def getPredictionsCollaborative(user, table_rating, df_customers):
    return completeMatrixCollaborative([user], table_rating.columns, table_rating, df_customers).loc[user]