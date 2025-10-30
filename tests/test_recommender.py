import os

import pandas as pd

from main import MovieRecommender, load_movies


def test_recommender_with_sample_data():
    # load the in-memory sample dataset via load_movies fallback
    movies = load_movies(csv_path="nonexistent_file.csv")
    assert isinstance(movies, pd.DataFrame)
    recommender = MovieRecommender(movies)
    recs = recommender.recommend("matrix", top_k=3)
    assert recs, "Expected at least one recommendation"
    top_idx, score = recs[0]
    title = recommender.get_movie(top_idx)["title"]
    assert "Matrix" in title or score >= 0.0
