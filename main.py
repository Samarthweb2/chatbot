"""Movie recommendation chatbot (main)

This script provides a small chatbot-style CLI that recommends movies
using scikit-learn's TF-IDF + cosine similarity.

Behavior:
- If a `movies.csv` file exists in the repository with columns
  `title,genres,description` it will be loaded. Otherwise a small
  built-in sample dataset is used.
- Type a query like "recommend action movies" or "i like space movies"
  and the bot will return the top matches.
- Type `exit` or `quit` to stop.

This file is designed to be a clear single-file example suitable for
including in a GitHub repo `main.py` entry point.
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_movies(csv_path: str = "movies.csv") -> pd.DataFrame:
	"""Load movies from CSV if present, otherwise return a small sample DataFrame.

	Expected CSV columns (if provided): title, genres, description
	"""
	if os.path.exists(csv_path):
		df = pd.read_csv(csv_path)
		# Basic check and filling
		for col in ("title", "genres", "description"):
			if col not in df.columns:
				df[col] = ""
		df = df[[(c in df.columns and c) or "title" for c in ["title", "genres", "description"]]]
		return df

	# Fallback sample dataset (small, illustrative)
	sample: List[Dict[str, str]] = [
		{"title": "The Matrix", "genres": "Action Sci-Fi", "description": "A hacker learns about the true nature of his reality and his role in the war against its controllers."},
		{"title": "Inception", "genres": "Action Sci-Fi Thriller", "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea."},
		{"title": "Interstellar", "genres": "Sci-Fi Drama", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
		{"title": "The Godfather", "genres": "Crime Drama", "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
		{"title": "Pulp Fiction", "genres": "Crime Drama", "description": "The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence and redemption."},
		{"title": "Toy Story", "genres": "Animation Family", "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."},
		{"title": "The Shawshank Redemption", "genres": "Drama", "description": "Two imprisoned men bond over years, finding solace and eventual redemption through acts of common decency."},
		{"title": "The Dark Knight", "genres": "Action Crime Drama", "description": "Batman faces the Joker, a criminal mastermind who plunges Gotham City into anarchy."},
		{"title": "La La Land", "genres": "Musical Romance", "description": "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles."},
		{"title": "Alien", "genres": "Horror Sci-Fi", "description": "The crew of a commercial spacecraft encounter a deadly lifeform after investigating a distress call."},
		{"title": "Back to the Future", "genres": "Adventure Comedy Sci-Fi", "description": "A teenager is accidentally sent 30 years into the past in a time-traveling DeLorean invented by a slightly mad scientist."},
	]
	return pd.DataFrame(sample)


class MovieRecommender:
	"""Simple recommender using TF-IDF on combined text (genres + description)."""

	def __init__(self, movies: pd.DataFrame):
		self.df = movies.copy().reset_index(drop=True)
		# create a single text field
		self.df["text"] = (self.df["genres"].fillna("") + " " + self.df["description"].fillna(""))
		self.vectorizer = TfidfVectorizer(stop_words="english")
		self.tfidf_matrix = self.vectorizer.fit_transform(self.df["text"])

	def recommend(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
		"""Return list of (index, score) for top_k movies matching the query."""
		if not query or not query.strip():
			return []
		q_vec = self.vectorizer.transform([query])
		cosine_sim = linear_kernel(q_vec, self.tfidf_matrix).flatten()
		if np.all(cosine_sim == 0):
			# no overlap found; return top popular-ish by default (here by index)
			indices = list(range(min(top_k, len(self.df))))
			return [(i, 0.0) for i in indices]
		top_indices = cosine_sim.argsort()[::-1][:top_k]
		return [(int(i), float(cosine_sim[i])) for i in top_indices]

	def get_movie(self, idx: int) -> Dict[str, str]:
		row = self.df.iloc[idx]
		return {"title": row["title"], "genres": row["genres"], "description": row["description"]}


def format_recommendations(recs: List[Tuple[int, float]], recommender: MovieRecommender) -> str:
	if not recs:
		return "I couldn't find any movie matches. Try different keywords (genre, mood, or plot)."
	lines = []
	for rank, (idx, score) in enumerate(recs, start=1):
		m = recommender.get_movie(idx)
		lines.append(f"{rank}. {m['title']} ({m['genres']}) — score: {score:.3f}\n   {m['description']}")
	return "\n\n".join(lines)


def run_chatbot():
	print("Movie Recommender Chatbot (type 'help' for tips, 'exit' to quit)\n")
	movies = load_movies()
	recommender = MovieRecommender(movies)

	while True:
		try:
			text = input("You: ").strip()
		except (KeyboardInterrupt, EOFError):
			print("\nGoodbye!")
			break

		if not text:
			continue
		low = text.lower()
		if low in {"exit", "quit"}:
			print("Bye — happy movie watching!")
			break
		if low in {"help", "h", "?"}:
			print("Hints: ask for a genre (e.g. 'recommend sci-fi'), a mood ('something romantic'), or mention a movie you like.")
			continue

		# If user asks explicitly for N recommendations like 'recommend 3 action movies'
		top_k = 5
		# naive extraction of a leading number
		tokens = low.split()
		if tokens and tokens[0].isdigit():
			top_k = max(1, min(10, int(tokens[0])))

		recs = recommender.recommend(text, top_k=top_k)
		print("\nHere are some recommendations:\n")
		print(format_recommendations(recs, recommender))
		print("\n---\n")


if __name__ == "__main__":
	run_chatbot()
