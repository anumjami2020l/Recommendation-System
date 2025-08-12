# 🎬 Movie Recommendation System

This repository contains **two movie recommendation applications** built with **Streamlit** using **K-Nearest Neighbors (KNN)**:

1. **Content-Based Filtering** – Recommends movies based on textual content like genres, keywords, overview, cast, and crew using **TF-IDF** vectorization.  
2. **Collaborative Filtering** – Recommends movies based on **user ratings similarity** using a ratings matrix.

---

## 📂 Project Structure

├── content.py # Content-Based Filtering app

├── collbartive.py # Collaborative Filtering app

├── movies.csv # Movie metadata for collaborative filtering

├── ratings.csv # User ratings dataset for collaborative filtering

├── tmdb_5000_credits.csv # Movie credits for content-based filtering

├── tmdb_5000_movies.csv # Movie details for content-based filtering

└── README.md # Project documentation

---

## 🧠 Recommendation Techniques

### 1. Content-Based Filtering (`content.py`)
- **Data Source**: TMDB 5000 dataset.
- **Method**:
  - Merges metadata and credits.
  - Combines genres, keywords, overview, cast, and crew into a single text feature.
  - Uses **TF-IDF Vectorization** to transform text into feature vectors.
  - Finds similar movies using **KNN** with cosine similarity.
- **Example**: If you liked *Avatar*, the system will recommend movies with similar genres, cast, and storyline.

---

### 2. Collaborative Filtering (`collbartive.py`)
- **Data Source**: MovieLens dataset.
- **Method**:
  - Uses user–movie ratings matrix.
  - Filters popular movies with a minimum rating count threshold.
  - Finds similar movies based on rating patterns using **KNN** with cosine similarity.
- **Example**: If you liked *The Dark Knight*, the system will recommend movies that users with similar tastes also rated highly.

