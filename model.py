import tensorflow_recommenders as tfrs
import tensorflow as tf
from tensorflow import keras
import keras
import pandas as pd
import numpy as np
import os, logging, warnings
import logging
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
class Model(tfrs.Model):
    def __init__(self,num_users, num_movies, user_feature_dim, movie_feature_dim, embedding_dim=64, hidden_dim=[128,64]):
        super().__init__()
        self.embedding_dim = embedding_dim
        #User Tower
        self.user_embedding = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-6)
            )
        #Movie tower
        self.movie_embedding = keras.layers.Embedding(
            input_dim=num_movies,
            output_dim=embedding_dim,
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        #User profiling network
        user_layers=[]
        input_dim = user_feature_dim
        for hidden_dim in hidden_dim:
            user_layers.extend([
                keras.layers.Dense(hidden_dim,activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.BatchNormalization()
            ])
        input_dim = hidden_dim
        user_layers.append(keras.layers.Dense(embedding_dim))
        self.user_feature_network = keras.Sequential(user_layers)

        movie_layers = []
        input_dim = movie_feature_dim
        for hidden_dim in [256, 128, 64]:
            movie_layers.extend([
                tf.keras.layers.Dense(hidden_dim, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization()
            ])
            input_dim = hidden_dim
        movie_layers.append(tf.keras.layers.Dense(embedding_dim))
        self.movie_feature_network = tf.keras.Sequential(movie_layers)

        #Final projection layers
        self.user_projection = tf.keras.layers.Dense(embedding_dim, activation=None)
        self.movie_projection = tf.keras.layers.Dense(embedding_dim, activation=None)

        #Retrival tasks
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.range(num_movies).batch(1000)
                    .map(lambda x: self.movie_model({'movie_index': x}))
            )
        )
        #Rating prediction
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.retrieval_task = tfrs.tasks.Retrieval()
        self.rating_task = tfrs.tasks.Ranking(loss=keras.losses.MeanSquaredError())
        
        # Task weights
        self.retrieval_weight = 1.0
        self.rating_weight = 0.5
    
    def user_model(self,features):
        user_id_emb = self.user_embedding(features['user_index'])
        user_features = self.user_feature_network(features['user_features'])
        combined = tf.concat([user_id_emb,user_features],axis=-1)
        output = self.user_projection(combined)
        return tf.nn.l2_normalize(output,axis=-1)
    
    def movie_model(self, features):
        movie_id_emb = self.movie_embedding(features['movie_index'])
        movie_features = self.movie_feature_network(features['movie_features'])
        combined = tf.concat([movie_id_emb, movie_features], axis=-1)
        output = self.movie_projection(combined)
        return tf.nn.l2_normalize(output, axis=-1)
    #Forward pass
    def call(self,features):
        user_embeddings = self.user_model(features)
        movie_embeddings = self.movie_model(features)
        return {
            'user_embeddings':user_embeddings,
            'movie_embeddings':movie_embeddings,
            'predicted_rating':tf.reduce_sum(user_embeddings*movie_embeddings,axis=-1)
        }
    #As we combine 2 models here for a hybrid model, we need to explicitly compute the total loss(linear combination of both losses)
    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model({
            'user_index': features['user_index'],
            'user_features': features['user_features']
        })
        
        movie_embeddings = self.movie_model({
            'movie_index': features['movie_index'],
            'movie_features': features['movie_features']
        })
        
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)
        
        predicted_ratings = tf.reduce_sum(
            user_embeddings * movie_embeddings, 
            axis=1, 
            keepdims=True
        )
        rating_loss = self.rating_task(
            labels=features['rating'],
            predictions=predicted_ratings
        )
        return retrieval_loss + 0.5 * rating_loss
    
    def prep_tfrs_data(preprocessed_data,test_size=0.2,batch_size=8192):
        merged_data = preprocessed_data[',erged_features']
        user_profile_cols = ['rating_count','rating_mean','rating_std','movies_rated']
        content_feature_cols = preprocessed_data['content_features'].columns.tolist()

        train_data, test_data = train_test_split(merged_data,test_size=test_size,random_state=42,stratify=merged_data['user_id'])

        def create_dataset(data_df):
            return tf.data.Dataset.from_tensor_slices({
                'user_index': data_df['user_index'].values.astype(np.int32),
                'movie_index': data_df['movie_index'].values.astype(np.int32),
                'user_features': data_df[user_profile_cols].values.astype(np.float32),
                'movie_features': data_df[content_feature_cols].values.astype(np.float32),
                'rating': data_df['rating'].values.astype(np.float32)[:,None]
            })
        train_ds = (create_dataset(train_data)
                    .shuffle(buffer_size=100000)
                    .batch(batch_size)
                    .cache()
                    .prefetch(tf.data.AUTOTUNE))
        
        test_ds = (create_dataset(test_data)
                .batch(batch_size)
                .cache()
                .prefetch(tf.data.AUTOTUNE))
    
        return train_ds, test_ds, len(user_profile_cols), len(content_feature_cols)
    #Start training!!!
    def train_hybrid_recommender(self,preprocessed_data,epochs=20,learning_rate = 0.001):
        train_ds,test_ds,user_feat_dim,movie_feat_dim = self.prep_tfrs_data(preprocessed_data)

        model = Model(
            num_users=preprocessed_data['num_users'],
            num_movies=preprocessed_data['num_movies'],
            user_feature_dim=user_feat_dim,
            movie_feature_dim=movie_feat_dim,
            embedding_dim=128)
        model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
        run_eagerly=False
    )
    
        # Callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    def build_retrieval_index(model, preprocessed_data):
        # Create movie dataset for indexing
        movie_indices = np.arange(preprocessed_data['num_movies'])
        content_features = preprocessed_data['content_features'].values.astype(np.float32)
        
        movie_ds = tf.data.Dataset.from_tensor_slices({
            'movie_index': tf.constant(movie_indices),
            'movie_features': tf.constant(preprocessed_data['content_features'].values.astype(np.float32))
        }).batch(128)
        
        # Build ScaNN index for fast approximate retrieval
        index = tfrs.layers.factorized_top_k.ScaNN(
            model.user_model,
            identifiers=tf.constant(movie_indices),
            k=100
        )
        
        index.index_from_dataset(
            movie_ds.map(lambda x: model.movie_model(x))
        )
        
        return index

    def get_recommendations(model, index, user_id, preprocessed_data, top_k=10):
        """Get recommendations for a specific user"""
        
        if user_id not in preprocessed_data['user_to_index']:
            raise ValueError(f"User {user_id} not found in training data")
        
        user_index = preprocessed_data['user_to_index'][user_id]
        user_profile = preprocessed_data['user_profiles'].loc[user_id].values.astype(np.float32)
        
        # Get recommendations
        _, movie_indices = index({
            'user_index': tf.constant([user_index]),
            'user_features': tf.constant([user_profile])
        })
        
        # Convert indices back to movie IDs
        recommended_movie_ids = [
            preprocessed_data['index_to_movie'][idx.numpy()]
            for idx in movie_indices[0, :top_k]
        ]
        
        return recommended_movie_ids
    
    def evaluate_recommendations(model, test_ds):
        metrics = model.evaluate(test_ds, return_dict=True)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(model, index, save_path="model"):
        # Save model
        model_path = os.path.join(save_path, "model")
        tf.saved_model.save(model, model_path)
        
        # Save index
        index_path = os.path.join(save_path, "index")
        tf.saved_model.save(index, index_path)
        
        logger.info(f"Model and index saved to {save_path}")
        return save_path