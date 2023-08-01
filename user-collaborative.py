import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('test_data_long.txt', header = None, names = ['customer', 'rating'], usecols = [0,1])
#df = pd.read_csv('data/combined_data_1.txt', header = None, names = ['customer', 'rating'], usecols = [0,1])

"""
PREPROCESSING
"""

## df[df['Rating'].isna()] returns a df with the rows where the movie ID appears
## df[df['Rating'].isna()]['Cust_Id'].str.replace(':', '') removes the : from the movie ID
## df['Movie_Id'] =    sets a movie ID attribute for those rows
df['movie'] = df[df['rating'].isna()]['customer'].str.replace(':', '')

## fills the attributes for other rows using the one above
df['movie'].fillna(method='ffill', inplace=True)

## drops the rows that contained the movie ID originally
df.dropna(subset=['rating'], inplace=True)




## number of ratings by customer
rating_counts = df.groupby(['customer'])['customer'].count()
rating_counts = rating_counts.sort_values(ascending=False)

selected_users_counts = rating_counts[(rating_counts >= 30) & (rating_counts <= 200)]
selected_users = selected_users_counts.index

## plot cumulative distribution of rating counts
# rating_countss = np.bincount(selected_users_counts.values)
# plt.plot(np.cumsum(rating_countss))
# plt.show()


df = df[df['customer'].isin(selected_users)]




## --------- BASIC STATISTICS ---------
print('Fraction of values known: {:.2f}'.format(df.shape[0] / (df['movie'].nunique() * df['customer'].nunique())))



"""
Collaborative filtering: predict ranking of movie j for user i
"""
user1 = '57633'
user2 = '786312'
movie = '2'
# assert user in df['customer'].values, f'User {user} not in df'
# assert movie not in df[df['customer']==user]['movie'].values, f'Movie {movie} is already rated by user {user}'


""" Compute customer attributes """
means = df.groupby('customer')['rating'].mean()
counts = df.groupby('customer')['rating'].count()
customers = pd.concat([means, counts], axis=1)
customers.columns = ['average','count']

def pearson(user1, user2):
    user1movies = df[df['customer']==user1][['movie','rating']].rename(columns={'rating':'rating1'}).set_index('movie')
    user2movies = df[df['customer']==user2][['movie','rating']].rename(columns={'rating':'rating2'}).set_index('movie')
    commonMovies = user2movies.merge(user1movies, on='movie')
    avg1 = customers.loc[user1]['average']
    avg2 = customers.loc[user2]['average']
    numerator = ((commonMovies['rating1']-avg1)*(commonMovies['rating2']-avg2)).sum()
    std1 = math.sqrt(((commonMovies['rating1']-avg1)**2).sum())
    std2 = math.sqrt(((commonMovies['rating2']-avg2)**2).sum())
    return numerator / (std1*std2)



# Create the ratings matrix using pivot_table
ratings_matrix = df.pivot_table(index='customer', columns='movie', values='rating')
#movieCorrelations = ratings_matrix.corr()
correlations = ratings_matrix.T.corr()
overlap = (ratings_matrix.notna().astype(int)).dot(ratings_matrix.notna().astype(int).T)

print(pearson(user1,user2))
print(correlations.loc[user1][user2])

# find k closest users for user1
# k = 5
# user1corrs = correlations.loc[user1]#.sort_values(ascending=False,na_position='last')
# usersWithMovieRated = ratings_matrix[movie].notna()
# usefulCorrs = user1corrs[usersWithMovieRated.values]
# usefulCorrs = usefulCorrs.sort_values(ascending=False,na_position='last')
# closeUsers = usefulCorrs.iloc[:k]
# closeUsersMovieRating = ratings_matrix.loc[closeUsers.index][movie]
# x = (closeUsersMovieRating-customers.loc[closeUsers.index]['average'])
# prediction = customers.loc[user1]['average'] + (closeUsers * x).sum() / (closeUsers.abs().sum())


def predictRating(user, movie):
    S = ratings_matrix[ratings_matrix[movie].notna()].index  # users who have rated movie
    sim = correlations.loc[user]  # correlations between user and others
    sim = sim.loc[S]
    sim = sim.sort_values(ascending=False,na_position='last')  # sort descending. not interested in users who don't have movies in common with user
    k = 5
    sim = sim.iloc[:k]
    Sk = sim.index  # identifiers of users with largest correlation between user
    r = ratings_matrix.loc[Sk][movie]  # ratings of Sk
    rc = (r-customers.loc[Sk]['average'])  # ratings of Sk centered around average of Sk
    prediction = customers.loc[user]['average'] + (sim * rc).sum() / (sim.abs().sum())
    return prediction

