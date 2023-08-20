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


def completeMatrixCollaborative(users, movies, table_rating, df_customers, k=5, beta=20):
    """
       Compute predictions for ratings of each user in users for each movie in movies.
    
       Parameters:
       users (list(int)): List of customer identifiers
       movies (list(int)): List of movie identifiers
       table_rating (DataFrame): Known ratings indexed by users, in columns for each movie
       df_customers (DataFrame): User information (number of movies rated, average rating) indexed by users
       k (int): see predictRating
       beta (int:) see predictRating
    
       Returns:
       df_prediction: DataFrame with users as index, movies as columns, containing predicted ratings 
    """
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