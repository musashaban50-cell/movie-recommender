import pytest

from recommender import MovieRecommender


@pytest.fixture(scope="module")
def rec():
    return MovieRecommender.from_csv()


def test_never_recommends_the_input_movie(rec):
    matched, results = rec.recommend_by_title("Inception", top_n=10)
    assert matched == "Inception"
    assert "Inception" not in results["Title"].tolist()


def test_returns_requested_number_of_results(rec):
    _, results = rec.recommend_by_title("The Godfather", top_n=3)
    assert len(results) == 3


def test_title_match_tolerates_typos_and_case(rec):
    matched, _ = rec.recommend_by_title("inceptoin")
    assert matched == "Inception"


def test_unknown_title_raises(rec):
    with pytest.raises(KeyError):
        rec.recommend_by_title("zzzzqqqq")


def test_genre_results_contain_that_genre(rec):
    results = rec.recommend_by_genre("sci-fi", top_n=5)
    assert len(results) == 5
    assert all("sci-fi" in g.lower() for g in results["Genre"])


def test_unknown_genre_raises(rec):
    with pytest.raises(KeyError):
        rec.recommend_by_genre("Cooking")
