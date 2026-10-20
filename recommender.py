"""Content-based movie recommender.

Each movie's genres are turned into a vector (CountVectorizer), and cosine
similarity between those vectors tells us how alike two movies are.

Usage:
    python recommender.py                       # interactive mode
    python recommender.py --title "Inception"   # movies similar to a title
    python recommender.py --genre "Sci-Fi"      # top movies for a genre
    python recommender.py --list-genres
"""

import argparse
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "movies.csv"


def split_genres(text: str) -> list[str]:
    """Turn 'Crime, Drama' into ['crime', 'drama']."""
    return [g.strip().lower() for g in text.split(",") if g.strip()]


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame):
        if not {"Title", "Genre"}.issubset(movies.columns):
            raise ValueError("Data must have 'Title' and 'Genre' columns.")
        self.movies = movies.dropna(subset=["Title", "Genre"]).reset_index(drop=True)
        self.vectorizer = CountVectorizer(tokenizer=split_genres, token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(self.movies["Genre"])
        self.similarity = cosine_similarity(self.matrix)
        self._titles_lower = self.movies["Title"].str.lower().tolist()

    @classmethod
    def from_csv(cls, path: Path = DATA_PATH) -> "MovieRecommender":
        return cls(pd.read_csv(path))

    @property
    def genres(self) -> list[str]:
        return sorted(g.title() if g != "sci-fi" else "Sci-Fi" for g in self.vectorizer.get_feature_names_out())

    def find_title(self, query: str) -> int | None:
        """Return the row index of the best title match (typo tolerant), or None."""
        query = query.strip().lower()
        if not query:
            return None
        if query in self._titles_lower:
            return self._titles_lower.index(query)
        close = difflib.get_close_matches(query, self._titles_lower, n=1, cutoff=0.6)
        if close:
            return self._titles_lower.index(close[0])
        partial = [i for i, t in enumerate(self._titles_lower) if query in t]
        return partial[0] if partial else None

    def _rank(self, scores: np.ndarray, exclude: int | None, top_n: int) -> pd.DataFrame:
        # Stable sort keeps dataset order for ties, so results are reproducible.
        order = [i for i in np.argsort(-scores, kind="stable") if i != exclude][:top_n]
        result = self.movies.iloc[order][["Title", "Genre"]].copy()
        result["Score"] = scores[order].round(2)
        return result.reset_index(drop=True)

    def recommend_by_title(self, title: str, top_n: int = 5) -> tuple[str, pd.DataFrame]:
        """Movies most similar to `title`. Returns (matched title, results)."""
        idx = self.find_title(title)
        if idx is None:
            raise KeyError(f"No movie found matching '{title}'.")
        return self.movies.at[idx, "Title"], self._rank(self.similarity[idx], idx, top_n)

    def recommend_by_genre(self, genre: str, top_n: int = 5) -> pd.DataFrame:
        """Movies that best match a single genre (purer genre fits rank higher)."""
        query = self.vectorizer.transform([genre])
        if query.nnz == 0:
            raise KeyError(f"Unknown genre '{genre}'. Try --list-genres.")
        scores = cosine_similarity(query, self.matrix).ravel()
        results = self._rank(scores, None, top_n)
        return results[results["Score"] > 0].reset_index(drop=True)


def print_results(df: pd.DataFrame) -> None:
    for i, row in df.iterrows():
        print(f"{i + 1}. {row['Title']}  [{row['Genre']}]  (similarity {row['Score']:.2f})")


def interactive(rec: MovieRecommender, top_n: int) -> None:
    print("Welcome to the Movie Recommendation System!")
    print("Type a movie you like, 'g' to browse by genre, or 'q' to quit.")
    while True:
        text = input("\nMovie you like: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            if text.lower() == "g":
                print("Genres:", ", ".join(rec.genres))
                genre = input("Genre: ").strip()
                print(f"\nTop {top_n} for '{genre}':")
                print_results(rec.recommend_by_genre(genre, top_n))
            else:
                matched, results = rec.recommend_by_title(text, top_n)
                print(f"\nBecause you liked '{matched}':")
                print_results(results)
        except KeyError as err:
            print(err.args[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    parser.add_argument("--title", help="a movie you like")
    parser.add_argument("--genre", help="a genre to browse")
    parser.add_argument("--top", type=int, default=5, help="number of results (default 5)")
    parser.add_argument("--list-genres", action="store_true", help="show available genres")
    args = parser.parse_args()

    rec = MovieRecommender.from_csv()
    try:
        if args.list_genres:
            print(", ".join(rec.genres))
        elif args.title:
            matched, results = rec.recommend_by_title(args.title, args.top)
            print(f"Because you liked '{matched}':")
            print_results(results)
        elif args.genre:
            print_results(rec.recommend_by_genre(args.genre, args.top))
        else:
            interactive(rec, args.top)
    except KeyError as err:
        parser.exit(1, err.args[0] + "\n")


if __name__ == "__main__":
    main()
