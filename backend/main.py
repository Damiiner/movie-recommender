import sqlite3
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Settings ----------
ART = Path(__file__).parent / "artifacts"
DB_PATH = Path(__file__).parent / "app.db"
K_NEIGHBORS = 30        # use the same value as your BEST_K
MIN_SUPPORT = 5         # a movie needs at least this many neighbors who rated it
MIN_RATINGS = 5         # ratings needed before personalizing a new user
FIRST_APP_USER_ID = 1000

# ---------- Load models once at startup ----------
data = joblib.load(ART / "model.joblib")
cf = joblib.load(ART / "cf_matrix.joblib")
genre_matrix = sparse.load_npz(ART / "genre_matrix.npz")

pred = data["pred"]
support = data["support"]
seen = data["seen"]
movies = data["movies"].reset_index(drop=True)   # row i == row i of genre_matrix
stats = data["stats"]
movie_idx = pd.Series(movies.index, index=movies["movieId"])

centered = cf["centered"]
rated_mask = cf["rated"]
cf_columns = np.array(cf["columns"])
col_pos = {int(m): i for i, m in enumerate(cf_columns)}

popular = (
    stats[stats["n_ratings"] >= 50]
    .merge(movies, on="movieId")
    .sort_values("avg_rating", ascending=False)
)


# ---------- Database ----------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL REFERENCES users(id),
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, movie_id)
            );
            """
        )


init_db()


# ---------- Helpers ----------
def format_recs(movie_ids, scores, supports):
    out = []
    for mid, s, sup in zip(movie_ids, scores, supports):
        info = movies.iloc[movie_idx[int(mid)]]
        out.append({
            "movieId": int(mid),
            "title": info["title"],
            "genres": info["genres"],
            "predicted_rating": round(float(s), 2),
            "neighbors": int(sup),
        })
    return out


def recommend_from_ratings(rows, n):
    """Collaborative filtering for a user who exists only in SQLite."""
    r = np.zeros(len(cf_columns), dtype=np.float32)
    mask = np.zeros(len(cf_columns), dtype=np.float32)
    for row in rows:
        j = col_pos.get(row["movie_id"])
        if j is not None:
            r[j] = row["rating"]
            mask[j] = 1

    count = int(mask.sum())
    if count < MIN_RATINGS:
        return None, count

    mean = r[mask == 1].mean()
    cen = (r - mean) * mask                       # mean-centered, like in the notebook

    sims = cosine_similarity(cen.reshape(1, -1), centered).ravel()
    top = np.argpartition(-sims, K_NEIGHBORS)[:K_NEIGHBORS]
    w = np.zeros_like(sims)
    w[top] = np.clip(sims[top], 0, None)          # only positive similarities

    num = w @ centered
    den = w @ rated_mask
    delta = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    scores = mean + delta
    sup = (w > 0).astype(np.float32) @ rated_mask

    cand = np.where((sup >= MIN_SUPPORT) & (mask == 0))[0]
    cand = cand[np.argsort(-scores[cand])][:n]
    return format_recs(cf_columns[cand], scores[cand], sup[cand]), count


# ---------- App ----------
app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RatingIn(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(ge=0.5, le=5)


@app.get("/health")
def health():
    return {"status": "ok", "users": len(pred), "movies": len(movies)}


@app.get("/movies/search")
def search(q: str = Query(..., min_length=2), n: int = Query(20, ge=1, le=50)):
    found = movies[movies["title"].str.contains(q, case=False, regex=False)].head(n)
    return found.to_dict(orient="records")


@app.get("/popular")
def get_popular(n: int = Query(10, ge=1, le=50)):
    top = popular.head(n)
    return top[["movieId", "title", "genres", "avg_rating", "n_ratings"]].round(2).to_dict(orient="records")


@app.post("/users")
def create_user():
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO users (id) SELECT COALESCE(MAX(id), ?) + 1 FROM users",
            (FIRST_APP_USER_ID - 1,),
        )
        return {"user_id": cur.lastrowid}


@app.post("/rate")
def rate(r: RatingIn):
    if r.movie_id not in movie_idx.index:
        raise HTTPException(status_code=404, detail="Movie not found")
    if round(r.rating * 2) != r.rating * 2:
        raise HTTPException(status_code=422, detail="Rating must be a multiple of 0.5")
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE id = ?", (r.user_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found")
        conn.execute(
            """
            INSERT INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id)
            DO UPDATE SET rating = excluded.rating, updated_at = CURRENT_TIMESTAMP
            """,
            (r.user_id, r.movie_id, r.rating),
        )
    return {"ok": True}


@app.get("/users/{user_id}/ratings")
def user_ratings(user_id: int):
    with db() as conn:
        rows = conn.execute(
            "SELECT movie_id, rating FROM ratings WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [
        {
            "movieId": r["movie_id"],
            "title": movies.iloc[movie_idx[r["movie_id"]]]["title"],
            "rating": r["rating"],
        }
        for r in rows
    ]


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int, n: int = Query(10, ge=1, le=50)):
    # 1) MovieLens user: precomputed predictions
    if user_id in pred.index:
        row = pred.loc[user_id].where(support.loc[user_id] >= MIN_SUPPORT).dropna()
        row = row[~row.index.isin(seen.get(user_id, set()))]
        top = row.nlargest(n)
        return {
            "strategy": "collaborative",
            "recommendations": format_recs(top.index, top.values, support.loc[user_id, top.index].values),
        }

    # 2) App user stored in SQLite
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
        rows = conn.execute(
            "SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,)
        ).fetchall()

    # 3) Unknown user: cold start
    if exists is None:
        return {"strategy": "popular_unknown_user", "recommendations": get_popular(n)}

    recs, count = recommend_from_ratings(rows, n)
    if recs is None:
        return {"strategy": "popular_need_ratings", "rated_count": count,
                "needed": MIN_RATINGS, "recommendations": get_popular(n)}
    if not recs:
        return {"strategy": "popular_no_signal", "rated_count": count,
                "needed": MIN_RATINGS, "recommendations": get_popular(n)}
    return {"strategy": "collaborative", "rated_count": count,
            "needed": MIN_RATINGS, "recommendations": recs}


@app.get("/similar/{movie_id}")
def similar(movie_id: int, n: int = Query(10, ge=1, le=50), min_ratings: int = 20):
    if movie_id not in movie_idx.index:
        raise HTTPException(status_code=404, detail="Movie not found")

    i = movie_idx[movie_id]
    sims = cosine_similarity(genre_matrix[i], genre_matrix).ravel()

    df = movies.copy()
    df["similarity"] = sims
    df = df.merge(stats, on="movieId", how="left").fillna({"n_ratings": 0, "avg_rating": 0})
    df = df[(df["movieId"] != movie_id) & (df["n_ratings"] >= min_ratings)]
    df["score"] = df["similarity"] * 0.7 + (df["avg_rating"] / 5) * 0.3

    top = df.sort_values("score", ascending=False).head(n)
    return top[["movieId", "title", "genres", "similarity", "avg_rating", "n_ratings"]].round(3).to_dict(orient="records")