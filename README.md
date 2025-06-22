# MovieLens Hybrid Recommender System

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Hybrid Recommendation System**: Combines collaborative filtering (user-item interactions) with content-based filtering (genres, TF-IDF features) for superior recommendation quality [5][6]
- **Two-Tower Deep Learning Architecture**: Efficient user and movie embedding towers optimized for large-scale datasets [3][7]
- **Advanced Feature Engineering**: Statistical user profiling and rich movie content features including genre encoding and TF-IDF vectorization [5]
- **Production-Ready Retrieval**: Fast approximate nearest neighbor search using TensorFlow Recommenders' ScaNN implementation [3]
- **Scalable Pipeline**: Modular, extensible codebase designed for easy adaptation and experimentation [2]
- **Multi-Task Learning**: Joint optimization for both retrieval and rating prediction tasks [7]

## Architecture

The system implements a hybrid two-tower architecture that processes user and movie features separately before computing similarity scores [3][7]:

```
User Tower: [User ID Embedding + User Profile Features] → User Representation
Movie Tower: [Movie ID Embedding + Content Features] → Movie Representation
Similarity: Dot Product(User Representation, Movie Representation)
```

## Installation

### Prerequisites

- Python 3.8 or higher [8]
- pip package manager [8]

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movielens-hybrid-recommender.git
   cd movielens-hybrid-recommender
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Core Dependencies

- `tensorflow>=2.8.0` - Deep learning framework [4]
- `tensorflow-recommenders>=0.7.0` - Recommendation system library [1][2]
- `pandas>=1.3.0` - Data manipulation and analysis [5]
- `numpy>=1.21.0` - Numerical computing [5]
- `scikit-learn>=1.0.0` - Machine learning utilities [5]

## Dataset Requirements

Place your MovieLens dataset files in the `data/` directory with the following structure [9][10]:

- `ratings.csv` - Must contain columns: `user_id`, `movie_id`, `rating`
- `movies.csv` - Must contain columns: `movie_id`, `title`, `movie_genres`  
- `users.csv` - Must contain column: `user_id` (additional user features optional)

**Data Format Requirements**:
- Ratings should be on a numerical scale (e.g., 1-5)
- Movie genres should be pipe-separated (e.g., "Action|Comedy|Drama")
- User and movie IDs can be any format (strings or integers)

Download MovieLens datasets from [GroupLens Research](https://grouplens.org/datasets/movielens/).

## Usage

### Quick Start

Run the complete training pipeline [9]:

```bash
python main.py
```

This command will:
1. Load and preprocess the MovieLens dataset
2. Train the hybrid recommendation model
3. Build retrieval indices for fast serving
4. Evaluate model performance on test data
5. Generate sample recommendations
6. Save the trained model to `movie_recommender_model/`

### Custom Configuration

Modify training parameters in `main.py` [11]:

```python
# Adjust filtering thresholds
preprocessor = DataPreprocessor(min_user_rating=10, min_movie_rating=15)

# Modify training parameters
model, history = train_hybrid_recommender(
    preprocessed_data,
    epochs=25,
    learning_rate=0.0005
)
```

### Getting Recommendations

After training, use the model to generate recommendations:

```python
# Get top 10 recommendations for a specific user
user_id = 123
recommendations = get_recommendations(
    model, index, user_id, preprocessed_data, top_k=10
)
print(f"Recommendations for user {user_id}: {recommendations}")
```

## Project Structure

```
movielens-hybrid-recommender/
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   └── users.csv
├── src/
│   └── model.py
|   └── data_loader.py
|   └── preprocessing.py
├── main.py
├── LICENSE
└── README.md
```

## Model Architecture

### Data Preprocessing Pipeline

1. **Data Filtering**: Removes users and movies with insufficient interactions (configurable thresholds) [10]
2. **Feature Engineering**: Creates user profiles (rating statistics) and movie content features (genres + TF-IDF) [5]
3. **Integer Mapping**: Converts string IDs to integer indices required for embedding layers [4]

### Neural Network Components

- **User Tower**: Embedding layer + Dense networks processing user profiles [3][7]
- **Movie Tower**: Embedding layer + Dense networks processing content features [3][7]
- **Multi-Task Learning**: Joint optimization for retrieval and rating prediction [7]
- **Regularization**: L2 regularization, dropout, and batch normalization [11]

### Training Configuration

- **Optimizer**: Adagrad with adaptive learning rate [4]
- **Loss Function**: Weighted combination of retrieval and ranking losses [7]
- **Batch Size**: 8192 (optimized for memory efficiency) [3]
- **Early Stopping**: Prevents overfitting with validation monitoring [11]

## Performance

The hybrid model typically achieves:
- **Retrieval Accuracy**: Improved cold-start handling through content features [5][6]
- **Training Speed**: Efficient two-tower architecture scales to large datasets [3]
- **Serving Latency**: Sub-millisecond recommendation generation using approximate retrieval [3]

## Contributing

We welcome contributions to improve the recommendation system [12][13]:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

Please ensure your code follows Python best practices and includes appropriate documentation [12][13].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details [8].

## Acknowledgments

- **MovieLens Dataset**: [GroupLens Research](https://grouplens.org/datasets/movielens/) for providing the movie ratings dataset
- **TensorFlow Recommenders**: [TensorFlow team](https://www.tensorflow.org/recommenders) for the excellent recommendation system framework [1][2]
- **Scikit-learn**: For machine learning utilities and preprocessing tools [5]
- **Community**: Thanks to all contributors and the open-source community
