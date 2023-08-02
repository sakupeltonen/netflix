import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os


def loadData(file, format):
    if os.path.isfile(f'data/{file}.csv'):
        return pd.read_csv(f'data/{file}.csv')
    else:
        df = pd.read_csv(file + '.' + format, header = None, names = ['customer', 'rating'], usecols = [0,1])

        df['movie'] = df[df['rating'].isna()]['customer'].str.replace(':', '')
        ## df[df['Rating'].isna()] returns a df with the rows where the movie ID appears
        ## df[df['Rating'].isna()]['Cust_Id'].str.replace(':', '') removes the : from the movie ID
        ## df['Movie_Id'] =    sets a movie ID attribute for those rows

        df['movie'].fillna(method='ffill', inplace=True)
        ## fills the attributes for other rows using the one above

        df.dropna(subset=['rating'], inplace=True)
        ## drops the rows that contained the movie ID originally

        df['movie'] = df['movie'].astype(int)

        df.to_csv(f'data/{file}.csv')

        return df



def filterByRatingCount(df_rating, treshold_min, treshold_max):
    """
       Create a filtered DataFrame of ratings by users 
       with at least threshold_min and at most treshold_max rated movies
    """
    rating_counts = df_rating.groupby(['customer'])['customer'].count()
    rating_counts = rating_counts.sort_values(ascending=False)

    selected_users_counts = rating_counts[(rating_counts >= treshold_min) & (rating_counts <= treshold_max)]
    selected_users = selected_users_counts.index

    return df_rating[df_rating['customer'].isin(selected_users)]



def loadMovies():
    df_movie = pd.read_csv('data/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['id', 'year', 'title'])
    df_movie.set_index('id', inplace = True)
    # df_movie['year'] = df_movie['year'].astype('int')
    return df_movie
