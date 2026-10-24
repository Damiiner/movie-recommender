import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const APP_USER_MIN_ID = 1000;
const FAIL_MSG = "Can't reach the API. Is uvicorn running on port 8000?";

async function api(path, options) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

const post = (path, body) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

function Stars({ value = 0, onRate }) {
  return (
    <div className="stars">
      {[1, 2, 3, 4, 5].map((s) => (
        <button
          key={s}
          className={s <= value ? "star on" : "star"}
          onClick={() => onRate(s)}
          aria-label={`${s} stars`}
        >
          ★
        </button>
      ))}
    </div>
  );
}

function MovieCard({ movie, badge, onSimilar, canRate, myRating, onRate }) {
  return (
    <div className="card">
      <div className="card-title">{movie.title}</div>
      <div className="card-genres">{movie.genres.split("|").join(" · ")}</div>
      {badge && <span className="badge">{badge}</span>}
      {canRate && <Stars value={myRating} onRate={(s) => onRate(movie, s)} />}
      <div className="card-footer">
        <button className="link" onClick={() => onSimilar(movie)}>
          Find similar
        </button>
      </div>
    </div>
  );
}

function Section({ title, subtitle, children }) {
  return (
    <section>
      <h2>{title}</h2>
      {subtitle && <p className="subtitle">{subtitle}</p>}
      {children}
    </section>
  );
}

export default function App() {
  const [userId, setUserId] = useState(() => Number(localStorage.getItem("userId")) || 1);
  const [userInput, setUserInput] = useState(String(userId));
  const isAppUser = userId >= APP_USER_MIN_ID;

  const [rec, setRec] = useState({ strategy: "", recommendations: [] });
  const [myRatings, setMyRatings] = useState({});
  const [popular, setPopular] = useState([]);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [searched, setSearched] = useState(false);

  const [similarTo, setSimilarTo] = useState(null);
  const [similar, setSimilar] = useState([]);

  const [error, setError] = useState("");

  const loadRecs = useCallback(() => {
    api(`/recommend/user/${userId}?n=10`)
      .then((d) => {
        setRec(d);
        setError("");
      })
      .catch(() => setError(FAIL_MSG));
  }, [userId]);

  useEffect(() => {
    api("/popular?n=20").then(setPopular).catch(() => setError(FAIL_MSG));
  }, []);

  useEffect(() => {
    loadRecs();
  }, [loadRecs]);

  useEffect(() => {
    if (!isAppUser) return;
    api(`/users/${userId}/ratings`)
      .then((list) => setMyRatings(Object.fromEntries(list.map((r) => [r.movieId, r.rating]))))
      .catch(() => setError(FAIL_MSG));
  }, [userId, isAppUser]);

  const switchUser = (id) => {
    localStorage.setItem("userId", String(id));
    setUserId(id);
    setUserInput(String(id));
    setSimilarTo(null);
  };

  const submitUser = (e) => {
    e.preventDefault();
    const id = parseInt(userInput, 10);
    if (!Number.isNaN(id)) switchUser(id);
  };

  const newUser = () => {
    post("/users", {})
      .then((d) => {
        setMyRatings({});
        switchUser(d.user_id);
      })
      .catch(() => setError(FAIL_MSG));
  };

  const rate = (movie, stars) => {
    post("/rate", { user_id: userId, movie_id: movie.movieId, rating: stars })
      .then(() => {
        setMyRatings((prev) => ({ ...prev, [movie.movieId]: stars }));
        loadRecs();
      })
      .catch(() => setError("Could not save the rating."));
  };

  const submitSearch = (e) => {
    e.preventDefault();
    if (query.trim().length < 2) return;
    api(`/movies/search?q=${encodeURIComponent(query)}&n=12`)
      .then((d) => {
        setResults(d);
        setSearched(true);
      })
      .catch(() => setError(FAIL_MSG));
  };

  const findSimilar = (movie) => {
    setSimilarTo(movie);
    api(`/similar/${movie.movieId}?n=10`).then(setSimilar).catch(() => setError(FAIL_MSG));
  };

  let subtitle = "";
  if (rec.strategy === "collaborative") {
    subtitle = `Personalized for user ${userId} (collaborative filtering)`;
  } else if (rec.strategy === "popular_need_ratings") {
    subtitle = `Rate ${rec.needed - rec.rated_count} more movie(s) below to unlock personalized picks (showing popular movies for now)`;
  } else if (rec.strategy === "popular_no_signal") {
    subtitle = "Your ratings are too uniform to find similar users. Try mixing high and low ratings.";
  } else if (rec.strategy) {
    subtitle = `User ${userId} not found, showing popular movies`;
  }

  const cardProps = (m) => ({
    movie: m,
    onSimilar: findSimilar,
    canRate: isAppUser,
    myRating: myRatings[m.movieId],
    onRate: rate,
  });

  return (
    <div className="app">
      <header>
        <h1>🎬 Movie Recommender</h1>
        <div className="header-actions">
          <form onSubmit={submitUser} className="user-form">
            <label>User ID</label>
            <input
              type="number"
              min="1"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
            />
            <button>Switch</button>
          </form>
          <button onClick={newUser}>+ New user</button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      {!isAppUser && (
        <div className="hint">
          You're viewing MovieLens user {userId} (read-only). Click “+ New user” to start rating
          movies yourself.
        </div>
      )}

      <Section title="Recommended for you" subtitle={subtitle}>
        <div className="grid">
          {rec.recommendations.map((m) => (
            <MovieCard
              key={m.movieId}
              {...cardProps(m)}
              badge={
                m.predicted_rating
                  ? `Match ${m.predicted_rating} · ${m.neighbors} similar users`
                  : `★ ${m.avg_rating} · ${m.n_ratings} ratings`
              }
            />
          ))}
        </div>
      </Section>

      <Section title="Search a movie">
        <form onSubmit={submitSearch} className="search-form">
          <input
            placeholder="Try: matrix, toy story, batman..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button>Search</button>
        </form>
        {searched && results.length === 0 && <p className="subtitle">No results.</p>}
        <div className="grid">
          {results.map((m) => (
            <MovieCard key={m.movieId} {...cardProps(m)} />
          ))}
        </div>
      </Section>

      {similarTo && (
        <Section title={`Similar to "${similarTo.title}"`}>
          <div className="grid">
            {similar.map((m) => (
              <MovieCard
                key={m.movieId}
                {...cardProps(m)}
                badge={`★ ${m.avg_rating} · ${m.n_ratings} ratings`}
              />
            ))}
          </div>
        </Section>
      )}

      <Section title="Popular movies" subtitle="Highest rated, with at least 50 ratings">
        <div className="grid">
          {popular.map((m) => (
            <MovieCard
              key={m.movieId}
              {...cardProps(m)}
              badge={`★ ${m.avg_rating} · ${m.n_ratings} ratings`}
            />
          ))}
        </div>
      </Section>
    </div>
  );
}