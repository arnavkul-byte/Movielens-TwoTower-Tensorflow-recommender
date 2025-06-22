import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#Logging information
logger = logging.getLogger(__name__)

#Lets start making the pre-processor class
class DataPreprocessor:
    def __init__(self, min_user_rating=20,min_movie_rating=20):
        self.min_user_ratings = min_user_rating
        self.min_movie_ratings = min_movie_rating
        self.mean_users = None
        self.scaler = StandardScaler()
        self.vectorizer = None
        #containers for processed data
        self.cleaned_data = None
        self.user_item_matrix = None
        self.genre_features = None
        self.title_features = None
        self.content_features = None
        self.content_features_normalized = None
        self.user_profiles = None
        self.movie_profiles = None
        #TFRS specific mappings
        self.user_to_index = None
        self.movie_to_index = None
        self.index_to_user = None
        self.index_to_movie = None
    
    def prepare_data(self,ratings, movies, users):
        logger.info('Merging all datasets...')
        data = pd.merge(users,ratings,on='user_id',how='left')
        data = pd.merge(data,movies,on='movie_id',how='left')
        logger.info(f'datasets merged, shape of the dataset: {data.shape}')
        user_count = data['user_id'].value_counts()
        movie_counts = data['movie_id'].value_counts()
        valid_users = user_count[user_count>=self.min_user_ratings].index
        valid_movies = movie_counts[movie_counts>=self.min_movie_ratings].index
        self.cleaned_data = data[
            (data['user_id'].isin(valid_users)) & (data['movie_id'].isin(valid_movies)).reset_index(drop=True)
        ]
        logger.info(f"Filtered data: {len(self.cleaned_data)} ratings from "
                   f"{self.cleaned_data['user_id'].nunique()} users and "
                   f"{self.cleaned_data['movie_id'].nunique()} movies")
        return self.cleaned_data
    
    def user_item_matrix(self):
        logger.info('creating user item matrix')
        #Make a pivot table
        self.user_item_matrix = self.cleaned_data.pivot_table(
            index='user_id',
            columns='movie_id',
            values='ratings',
            fill_value=0
        )
        #calculate sparsity of the table
        total_entries = self.user_item_matrix.size
        non_zero_entries = (self.user_item_matrix != 0).sum().sum()
        sparsity = 1-(non_zero_entries/total_entries)

        logger.info(f'Shape of user-item matrix: {self.user_item_matrix.shape}')
        logger.info(f"Matrix sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
        return self.user_item_matrix
    
    def content_features(self, movies):
        logger.info('creating content features')
        valid_movies = movies[movies['movie_id'].isin(self.user_item_matrix.columns)]
        self.genre_features = valid_movies['movie_genres'].str.get_dummies(sep='|')
        self.genre_features.index = valid_movies['movie_id']
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1,2),
            max_df=2
        )
        tfidf_matrix = tfidf.fit_transform(valid_movies['title'])
        self.title_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=valid_movies['movie_id'],
            columns=[f'title_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        self.content_features = pd.concat([self.genre_features, self.title_features],axis=1)
        scaler = StandardScaler()
        self.content_features_normalized = pd.DataFrame(
            scaler.fit_transform(self.content_features),
            index=self.content_features.index,
            columns=self.content_features.columns
            )
        
        logger.info(f"Content features shape: {self.content_features.shape}")
        logger.info(f"Genre features: {self.genre_features.shape[1]} genres")
        logger.info(f"Title features: {self.title_features.shape[1]} TF-IDF terms")
        return self.content_features_normalized
    
    def create_profiles(self):
        logger.info('Creating user profiles')
        self.user_profiles = self.cleaned_data.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'movie_id': 'nunique'
        }).round(3)
        
        self.user_profiles.columns = ['_'.join(col).strip() for col in self.user_profiles.columns]
        self.user_profiles = self.user_profiles.rename(columns={'movie_id_nunique': 'movies_rated'})
        self.user_profiles['rating_std'] = self.user_profiles['rating_std'].fillna(0)
        
        self.movie_profiles = self.cleaned_data.groupby('movie_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).round(3)
        
        self.movie_profiles.columns = ['_'.join(col).strip() for col in self.movie_profiles.columns]
        self.movie_profiles = self.movie_profiles.rename(columns={'user_id_nunique': 'users_rated'})
        self.movie_profiles['rating_std'] = self.movie_profiles['rating_std'].fillna(0)
        
        logger.info(f"User profiles shape: {self.user_profiles.shape}")
        logger.info(f"Movie profiles shape: {self.movie_profiles.shape}")
        return self.user_profiles, self.movie_profiles
    def tfrs_mappings(self):
        logger.info('creating mappings for tfrs...')
        unique_users = sorted(self.cleaned_data['user_id'].unique())
        unique_movies = sorted(self.cleaned_data['movie_id'].unique())

        self.user_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.index_to_user = {idx: user_id for user_id, idx in self.user_to_index.items()}
        self.index_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_index.items()}

        logger.info(f'created mappings for {len(unique_users)} users and {len(unique_movies)} movies.')
        return self.user_to_index, self.movie_to_index
    
    def merge_features(self):
        logger.info('merging features to tfrs using mappings')

        merged_data = pd.merge(self.cleaned_data, self.user_profiles.reset_index(), on='user_id',how='left')
        content_features_reset = self.content_features_normalized.reset_index()
        content_features_reset = content_features_reset.rename(columns={'index':'movie_id'})
        merged_data = pd.merge(merged_data,content_features_reset,on='movie_id',how='left')
        merged_data = merged_data.fillna(0)
        logger.info(f'Merged data shape:{merged_data.shape}')
        return merged_data

    
    def preprocess_Data(self, ratings, movies, users):
        logger.info('Starting data preprocessing pipeline')
        self.prepare_data(ratings,movies, users)
        self.user_item_matrix()
        self.content_features(movies)
        self.create_profiles()
        self.tfrs_mappings()
        merged_features = self.merge_features()
        logger.info('preprocessing completed')
        return {
            'cleaned_data': self.cleaned_data,
            'user_item_matrix': self.user_item_matrix,
            'genre_features': self.genre_features,
            'title_features': self.title_features,
            'content_features': self.content_features_normalized,
            'user_profiles': self.user_profiles,
            'movie_profiles': self.movie_profiles,
            'merged_features': merged_features,
            'user_to_index': self.user_to_index,
            'movie_to_index': self.movie_to_index,
            'num_users': len(self.user_to_index),
            'num_movies': len(self.movie_to_index)
        }