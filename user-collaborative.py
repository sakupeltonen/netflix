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

# print('Fraction of values known: {:.2f}'.format(df_rating.shape[0] / (df_rating['movie'].nunique() * df_rating['customer'].nunique())))



# Create a DataFrame with information for each customer
means = df_rating.groupby('customer')['rating'].mean()
counts = df_rating.groupby('customer')['rating'].count()
df_customers = pd.concat([means, counts], axis=1)
df_customers.columns = ['average','count']

# Create a DataFrame with information for each movie
df_movie = loadMovies()
ratedMovies = df_rating['movie'].values
df_movie = df_movie.loc[df_movie.index.isin(ratedMovies)]  # only interested in movies that are rated
df_movie.loc[ratedMovies,'nrating'] = df_rating.groupby('movie')['rating'].count()
df_movie['nrating'] = df_movie['nrating'].astype(int)

# Create a DataFrame of customers and movies
table_rating = df_rating.pivot_table(index='customer', columns='movie', values='rating')

# Create a DataFrame of correlations between users
table_corr = table_rating.T.corr()

table_overlap = (table_rating.notna().astype(int)).dot(table_rating.notna().astype(int).T)


# =========================================
# Predict ratings with collaborative filtering
# =========================================

def predictRating(user, movie, k=5, beta=20):
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


def getPredictions(user,beta=20):
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

    
def printRatings(df, data=True, pred=True, n=10):
    """
       Print a list of ratings for a given user.
    
       Parameters:
       df (DataFrame): DataFrame with movies as index; known ratings and predictions for user as data.
       data (Bool): Print known ratings or not
       pred (Bool): Print predictions or not
       n (Int): Number of rows printed
    """
    if data:
        df = df[df['data'].notna()]
        df = df.sort_values(by='data', ascending=False)  # sort by rating
    df['title'] = df_movie.loc[df.index]['title']

    count = 0
    for _, row in df.iterrows():
        count += 1
        message = ""
        if data:
            message += f"{row['data']}\t"
        if pred:
            message += f"{row['pred']:.1f}\t"
        message += f"{row['title']}"
        print(message)

        if count >= n-1: 
            break
     

def getRecommendation(user, n=10):
    """
       Get a list of recommendations for user
    
       Parameters:
       user (Int): Unique identifier of user
       n (Int): Number of recommendations
    
       The results are printed.
    """
    df = getPredictions(user)
    df = df.sort_values(by='pred', ascending=False)
    df = df[df['data'].isna()]
    printRatings(df, data=False, n=n)


# user = 57633


def testPredictions(n_users=50):
    corrs = []
    # users = table_rating.index
    sample = table_rating.sample(n_users).index
    for user in sample:
        df_prediction = getPredictions(user,beta=20)
        corr = df_prediction['data'].corr(df_prediction['pred'])
        corrs.append(corr)
        print(f'User {user}. Correlation {corr:.2f}')    
    print(f'Average correlation {np.mean(corrs):.2f}')

testPredictions()

# beta=20, k=5, n_users=50. 0.77 average correlation
# beta=0, k=5, n_users=10. 0.75 average correlation