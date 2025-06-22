import pandas as pd
import numpy as np
import datetime
import os
import logging

class DataLoader:
    def __init__(self,data_dir='data'):
        self.data_dir = data_dir
        self.encoding = 'latin-1'
        self.delimiter = '::'

    def load_file(self,filename,column_names):
        file_path = os.path.join(self.data_dir,filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        try:
            data = pd.read_csv(file_path,delimiter=self.delimiter,engine='python',names=column_names,encoding=self.encoding)
            logging.info(f"Successfully loaded {filename}: {data.shape}")
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            raise
        return data

    def load_movies(self):
        movies = self.load_file('movies.dat',['movie_id','movie_title','movie_genres'])
        nulls = movies.isnull().sum().tolist()

        if sum(nulls) != 0:
            movies = movies.dropna()
        
        duplicates = int(movies.duplicated().sum())
        if duplicates !=0:
            movies = movies.drop_duplicates()
        
        movies['movie_id'] = movies['movie_id'].astype('int32')
        movies['movie_title'] = movies['movie_title'].astype(str)
        movies['movie_genres'] = movies['movie_genres'].astype(str)

        return movies

    def load_ratings(self):
        ratings = self.load_file('ratings.dat',['user_id','movie_id','rating','timestamp'])
        nulls = ratings.isnull().sum().tolist()
        if sum(nulls) != 0:
            ratings = ratings.dropna()

        duplicates = int(ratings.duplicated().sum())
        if duplicates !=0:
            ratings = ratings.drop_duplicates()
        
        ratings['user_id'] = ratings['user_id'].astype('int32')
        ratings['movie_id'] = ratings['movie_id'].astype('int32')
        ratings['rating'] = ratings['rating'].astype('float32')
        try:
            ratings['timestamp'] = ratings['timestamp'].apply(datetime.datetime.fromtimestamp)
        except Exception:
            raise TypeError(f'Timestamps contain invalid values')

        return ratings

    def load_users(self):
        users = self.load_file('users.dat',['user_id','gender','age','occupation','zip_code'])
        nulls = users.isnull().sum().tolist()
        if sum(nulls) != 0:
            users = users.dropna()

        duplicates = int(users.duplicated().sum())
        if duplicates !=0:
            users = users.drop_duplicates()
        
        users['user_id'] = users['user_id'].astype('int32')
        users['gender'] = users['gender'].astype(str)
        users['age'] = users['age'].astype('int32')
        users['occupation'] = users['occupation'].astype('int32')
        users['zip_code'] = users['zip_code'].astype(str)

        return users
    
    def load_all_data(self):
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()
        
        return ratings, movies, users
