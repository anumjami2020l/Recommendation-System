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

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Make sure you have Python 3.8+ installed.

3. Run the Content-Based App
bash
Copy
Edit
streamlit run content.py
4. Run the Collaborative Filtering App
bash
Copy
Edit
streamlit run collbartive.py
📊 Datasets
Content-Based:

tmdb_5000_movies.csv

tmdb_5000_credits.csv
(Available on Kaggle)

Collaborative Filtering:

movies.csv

ratings.csv
(Available on MovieLens Dataset)

🎨 UI Features
Modern Streamlit interface.

Hover effects for recommendation cards.

IMDb links for quick access to movie details.

Adjustable number of recommendations.

⚙️ Tech Stack
Python

Streamlit

scikit-learn

pandas, numpy

SciPy

TF-IDF Vectorization

KNN (Cosine Similarity)

📌 Example
Content-Based Filtering Example:

pgsql
Copy
Edit
Input: "Inception"
Output: Recommendations include movies with similar genres, plot themes, and actors.
Collaborative Filtering Example:

pgsql
Copy
Edit
Input: "The Dark Knight"
Output: Recommendations include movies rated highly by users with similar tastes.
