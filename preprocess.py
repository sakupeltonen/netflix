import pandas as pd
import numpy as np
import math
import os
import networkx as nx


def loadData(file):
    # if os.path.isfile(f'{file}.csv'):
    #     return pd.read_csv(f'{file}.csv')
    # else:
        
    df = pd.read_csv(file + '.txt', header = None, names = ['customer', 'rating'], usecols = [0,1])

    df['movie'] = df[df['rating'].isna()]['customer'].str.replace(':', '')
    ## df[df['Rating'].isna()] returns a df with the rows where the movie ID appears
    ## df[df['Rating'].isna()]['Cust_Id'].str.replace(':', '') removes the : from the movie ID
    ## df['Movie_Id'] =    sets a movie ID attribute for those rows

    df['movie'].fillna(method='ffill', inplace=True)
    ## fills the attributes for other rows using the one above

    df.dropna(subset=['rating'], inplace=True)
    ## drops the rows that contained the movie ID originally

    df['movie'] = df['movie'].astype(int)

    # df.to_csv(f'{file}.csv')

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



def read_netflix_graph(file):
    """Create a graph from netflix user ratings. Needs to be integrated to the node2vec functions"""
    df = loadData(file)
    df = filterByRatingCount(df, 30, 200)
    df['customer'] = 'c' + df['customer']
    df['movie'] = 'm' + df['movie'].astype(str)
    

    # Create a df of customers
    means = df.groupby('customer')['rating'].mean()
    counts = df.groupby('customer')['rating'].count()
    df_customers = pd.concat([means, counts], axis=1)
    df_customers.columns = ['average','count']

    # Construct bipartite graph of normalized ratings
    G = nx.Graph()
    global_average_rating = df['rating'].mean()
    for _, row in df.iterrows():
        customer = row['customer']
        average_rating = df_customers.loc[customer, 'average']
        # weight_normalized = row['rating'] - average_rating
        weight_normalized = row['rating'] - (average_rating + global_average_rating)/2
        if weight_normalized != 0:
            G.add_edge(customer, row['movie'], weight=abs(weight_normalized), signed_weight=weight_normalized)

    return G
