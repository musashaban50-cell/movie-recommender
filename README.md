# 🎬 CineMatch – Content-Based Movie Recommender

Tell it a movie you love and it suggests similar ones. Built with Python, pandas and scikit-learn.

<!-- Add a screenshot of the terminal output or your website mockup here:
![demo](docs/demo.png) -->

## How it works

1. **Vectorize** – each movie's genres (e.g. `Crime, Drama, Thriller`) are turned into a numeric vector with scikit-learn's `CountVectorizer` and a custom tokenizer.
2. **Compare** – cosine similarity is computed between every pair of movies. Movies with similar genre mixes score close to 1.
3. **Recommend** – for the movie you enter, the most similar *other* movies are returned, ranked by score. Ties keep dataset order, so results are reproducible.

Example: *Inception* (`Action, Adventure, Sci-Fi`) vs *The Matrix* (`Action, Sci-Fi`) = 2 ÷ (√3 × √2) ≈ **0.82**.

## Quick start

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt

python recommender.py                        # interactive mode
python recommender.py --title "Inception"    # similar movies
python recommender.py --genre "Sci-Fi" --top 3
python recommender.py --list-genres
```

Sample output:

```
Because you liked 'Inception':
1. The Matrix  [Action, Sci-Fi]  (similarity 0.82)
2. Interstellar  [Adventure, Drama, Sci-Fi]  (similarity 0.67)
...
```

Title search is typo-tolerant (`"inceptoin"` still finds *Inception*).

## Project structure

| File | Purpose |
|------|---------|
| `recommender.py` | Recommender class and command-line interface |
| `movies.csv` | Dataset: 50 well-known films with genres |
| `test_recommender.py` | Unit tests (run with `pytest`) |
| `requirements.txt` | Dependencies |

## Tests

```bash
pip install pytest
pytest
```

## Limitations

- Only genres are used, so many movies tie. *Inception* recommends *Gladiator* as often as *Interstellar* because both share two genres with it.
- The dataset is small (50 films) and hand-made.

## Roadmap

- [ ] Add plot descriptions, directors and cast using TF-IDF
- [ ] Switch to a larger dataset (MovieLens / TMDB)
- [ ] Add user ratings for a hybrid recommender
- [ ] Add an evaluation metric (e.g. precision@k on held-out ratings)
- [ ] Turn the website mockup into a working Streamlit app

## License

MIT
