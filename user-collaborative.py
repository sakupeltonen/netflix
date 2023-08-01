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
    S = table_rating[table_rating[movie].notna()].index  # users who have rated movie
    sim = table_corr.loc[user]  # correlations between user and others
    sim = sim.loc[S]
    sim = sim.sort_values(ascending=False,na_position='last')  # sort descending. not interested in users who don't have movies in common with user
    sim = sim.iloc[:k]
    Sk = sim.index  # identifiers of users with largest correlation between user
    r = table_rating.loc[Sk][movie]  # ratings of Sk
    rc = (r-df_customers.loc[Sk]['average'])  # ratings of Sk centered around average of Sk
    prediction = df_customers.loc[user]['average'] + (sim * rc).sum() / (sim.abs().sum())
    return prediction


user1 = 57633
user2 = 786312
movie = 2
predictRating(user1, movie)