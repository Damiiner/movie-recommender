# 🎬 Movie Recommender

A full-stack recommendation engine built on the MovieLens dataset. It combines
popularity, content-based, and collaborative filtering, served through a FastAPI
backend with a React frontend and SQLite storage for new users.

## Screenshots

**Personalized recommendations for a new user (after rating movies):**

![Recommendations](docs/recommendations.png)

**Search and similar movies:**

![Search and similar movies](docs/search-similar.png)

**MovieLens user (read-only view):**

![MovieLens user](docs/old-user.png)

## Features

- **Popularity baseline**: top-rated movies with a minimum rating count
- **Content-based**: "similar movies" from genre TF-IDF vectors, blended with average rating
- **Collaborative filtering**: user-based k-NN with mean-centered cosine similarity
- **New users**: create an account, rate movies, and get personalized recommendations
  after 5 ratings (stored in SQLite, computed on the fly)
- **Cold-start handling**: unknown or new users fall back to popular movies

## Tech stack

Python, Pandas, NumPy, scikit-learn, FastAPI, SQLite, React (Vite)

## Architecture

```
React (Vite)  ──HTTP──▶  FastAPI  ──▶  precomputed model (joblib)
                            │
                            └──▶  SQLite (app users + their ratings)
```

## Results

Evaluated on an 80/20 random split of MovieLens ml-latest-small
(about 100k ratings, 610 users, 9.7k movies). RMSE measures rating prediction
error (lower is better). Precision@10 and Recall@10 count a movie as relevant
if the user rated it 4 or higher in the held-out test set.

| Method               | RMSE   | Precision@10 | Recall@10 |
| -------------------- | ------ | ------------ | --------- |
| Global mean          | 1.0488 | n/a          | n/a       |
| Movie mean           | 0.9827 | n/a          | n/a       |
| Popularity           | n/a    | 0.072        | 0.049     |
| User-based CF, k=10  | 0.9645 | 0.120        | 0.121     |
| User-based CF, k=30  | 0.9331 | 0.043        | 0.041     |
| User-based CF, k=60  | 0.9163 | 0.014        | 0.013     |
| User-based CF, k=100 | 0.9042 | 0.006        | 0.004     |

**Findings**

- Collaborative filtering predicts ratings better than both baselines (lower RMSE),
  and the error keeps dropping as k grows.
- For top-10 recommendations the opposite happens: k=10 gives the best precision
  (0.120, higher than popularity at 0.072), and larger k values do worse.
- So the best k depends on the goal: rating prediction or ranking.
- The demo app uses k=30.

## Limitations

- **Popularity bias**: neighbor-based CF favors widely rated, highly rated movies.
- **Cold start**: new users need at least 5 ratings before personalization.
- **Uniform ratings**: because ratings are mean-centered, a user who rates everything
  the same gives no taste signal.
- **Scalability**: the user-item matrix is held in memory, which is fine at this scale.
- **Evaluation**: a single random split on a small dataset, so small differences
  between settings may be noise.

## Run it locally

**1. Get the data**: download `ml-latest-small` from
[grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
and copy `movies.csv` and `ratings.csv` into `data/`.

**2. Install and train**

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook                 # open notebooks/recommender.ipynb → Run All
```

Running the notebook writes the model files to `backend/artifacts/`.

**3. Start the API**

```bash
cd backend
uvicorn main:app --reload        # docs at http://127.0.0.1:8000/docs
```

**4. Start the frontend**

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## API

| Endpoint                   | Description                    |
| -------------------------- | ------------------------------ |
| `GET /recommend/user/{id}` | Personalized recommendations   |
| `GET /similar/{movie_id}`  | Similar movies (content-based) |
| `GET /movies/search?q=`    | Search by title                |
| `GET /popular`             | Top-rated popular movies       |
| `POST /users`              | Create an app user             |
| `POST /rate`               | Add or update a rating         |
| `GET /users/{id}/ratings`  | A user's saved ratings         |

## Possible improvements

Matrix factorization (SVD), hybrid content and collaborative scoring, a popularity
penalty, cross-validation, Docker, and deployment.

## Data

F. Maxwell Harper and Joseph A. Konstan. 2015. _The MovieLens Datasets: History and
Context._ ACM TiiS 5, 4, Article 19. https://doi.org/10.1145/2827872
