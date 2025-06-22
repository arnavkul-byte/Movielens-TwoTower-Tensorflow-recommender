import pandas as pd
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import sparse
import json
import warnings
import tensorflow as tf
import tensorflow_recommenders as tfrs
from src.model import Model
from src.preprocessing import DataPreprocessor
from src.data_loader import DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Define data paths (update these to your actual file paths)
    data_dir = "data"
    ratings_path = os.path.join(data_dir, "ratings.csv")
    movies_path = os.path.join(data_dir, "movies.csv")
    users_path = os.path.join(data_dir, "users.csv")
    
    # Load data
    logger.info("Loading datasets...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    users = pd.read_csv(users_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(min_user_rating=20, min_movie_rating=20)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessed_data = preprocessor.preprocess_Data(ratings, movies, users)
    
    # Train recommender model
    logger.info("Training recommender model...")
    model, history = model(
        preprocessed_data,
        epochs=10,
        learning_rate=0.001
    )
    
    # Build retrieval index
    logger.info("Building retrieval index...")
    index = build_retrieval_index(model, preprocessed_data)
    
    # Evaluate model
    logger.info("Evaluating model...")
    _, test_ds, _, _ = prep_tfrs_data(preprocessed_data)
    metrics = evaluate_recommendations(model, test_ds)
    
    # Get sample recommendations
    sample_user_id = preprocessed_data['cleaned_data']['user_id'].iloc[0]
    recommendations = get_recommendations(
        model, index, sample_user_id, preprocessed_data, top_k=10
    )
    
    logger.info(f"Sample recommendations for user {sample_user_id}: {recommendations}")
    
    # Save model and index
    save_path = save_model(model, index)
    
    logger.info("Recommendation system pipeline completed successfully!")
    return model, index, preprocessed_data

if __name__ == '__main__':
    main()


