import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import loadData, filterByRatingCount, loadMovies
from user_collaborative import completeMatrixCollaborative, getPredictionsCollaborative
from model_based import completeMatrixSVD

np.random.seed(42)

# file = 'data/test_data_long'
file = '../data/test_data'
#file = 'combined_data_1'

# =========================================
#           Initialize DataFrames 
# =========================================

df_rating = loadData(file)
# Filter list of ratings to only contain ratings by users with at least 30 and at most 200 ratings. 
df_rating = filterByRatingCount(df_rating, 1, 200)

# count of customers and movies
ncustomer = df_rating['customer'].nunique()
nmovie = df_rating['movie'].nunique()
# print('Fraction of values known: {:.2f}'.format(df_rating.shape[0] / (nmovie * ncustomer)))


# Create a DataFrame of customers and movies
table_rating = df_rating.pivot_table(index='customer', columns='movie', values='rating')


# Create a DataFrame with information for each customer
means = df_rating.groupby('customer')['rating'].mean()
counts = df_rating.groupby('customer')['rating'].count()
df_customers = pd.concat([means, counts], axis=1)
df_customers.columns = ['average','count']


# Create a DataFrame with information for each movie
df_movie = loadMovies()
ratedMovies = df_rating['movie'].values
# filter df_movie to only contain movies that have been rated in the given dataset
df_movie = df_movie.loc[df_movie.index.isin(ratedMovies)]
# add rating count 'nrating' for each movie  
df_movie.loc[ratedMovies,'nrating'] = df_rating.groupby('movie')['rating'].count()
df_movie['nrating'] = df_movie['nrating'].astype(int)
# add weight for each movie, proportional to log of inverse frequency
df_movie['inverseUserFreq'] = np.log(ncustomer / df_movie['nrating'])   




def testPredictions(method, n_users=50):
    corrs = []
    sample = table_rating.sample(n_users).index
    if method == 'collab':
        prediction = completeMatrixCollaborative(sample, table_rating.columns, table_rating, df_customers)
    elif method == 'model-based':
        prediction = completeMatrixSVD(table_rating)
        prediction = prediction.round(decimals=2)  # for debugging
    else:
        print(f'Method {method} not implemented.')
        return

    for user in sample:
        user_pred = prediction.loc[user]
        
        original = table_rating.loc[user]
        corr = original.corr(user_pred)
        corrs.append(corr)
        print(f'User {user}. Correlation {corr:.2f}')    
    print(f'Average correlation {np.mean(corrs):.2f}')

# testPredictions('model-based', n_users=3)
testPredictions('collab', n_users=3)

    
def printRatings(user, pred=None, sortby='pred', n=10):
    """
       Print a list of ratings for a given user.
    
       Parameters:
       data (DataFrame): Contains known ratings. Usually given by table_rating.loc[user]
       pred (DataFrame): Contains predicted ratings. 
       n (Int): Number of rows printed
    """
    df = table_rating.loc[user]  # known data

    if pred is not None:
        df = pd.concat([df, pred], axis=1)
        df.columns = ['data', 'pred']
    df = df.sort_values(by=sortby, ascending=False)     
    # TODO fix case when pred is None: we need to actually make df a DataFrame and not series?   
        
    df['title'] = df_movie.loc[df.index]['title']

    print(f'Printing top {n} rated movies for user {user}, sorted by {sortby}:')

    count = 0
    for _, row in df.iterrows():
        count += 1
        message = f"{row['data']}\t"
        if pred is not None:
            message += f"{row['pred']:.1f}\t"
        message += f"{row['title']}"
        print(message)

        if count >= n-1: 
            break


user = 1333
df_pred = getPredictionsCollaborative(user, table_rating, df_customers)
printRatings(user, pred=df_pred)