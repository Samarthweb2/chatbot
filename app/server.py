from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import os

from main import load_movies, MovieRecommender
from models.intent import IntentClassifier

app = FastAPI(title="Movie Recommender API")

# Serve the static UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class RecommendRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
def startup_event():
    # load movies and recommender once
    movies = load_movies()
    app.state.recommender = MovieRecommender(movies)
    # initialize intent classifier
    # try to load a persisted intent model first
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "intent.pkl")
    model_path = os.path.normpath(model_path)
    if os.path.exists(model_path):
        try:
            app.state.intent = IntentClassifier.load(model_path)
        except Exception:
            # fallback to training a fresh classifier
            try:
                app.state.intent = IntentClassifier()
                # attempt to save it for next time
                try:
                    app.state.intent.save(model_path)
                except Exception:
                    pass
            except Exception:
                app.state.intent = None
    else:
        try:
            app.state.intent = IntentClassifier()
            try:
                app.state.intent.save(model_path)
            except Exception:
                pass
        except Exception:
            app.state.intent = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    recommender: MovieRecommender = app.state.recommender
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    recs = recommender.recommend(req.query, top_k=req.top_k)
    results = []
    for idx, score in recs:
        m = recommender.get_movie(idx)
        results.append({"title": m["title"], "genres": m["genres"], "description": m["description"], "score": score})
    return {"query": req.query, "results": results}


@app.post("/chat")
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    # Use intent classifier if available
    intent = None
    if getattr(app.state, "intent", None):
        try:
            intent = app.state.intent.predict(text)
        except Exception:
            intent = None

    recommender: MovieRecommender = app.state.recommender

    if intent == "recommend" or (intent is None and ("recommend" in text.lower() or "suggest" in text.lower())):
        recs = recommender.recommend(text, top_k=5)
        results = []
        for idx, score in recs:
            m = recommender.get_movie(idx)
            results.append({"title": m["title"], "genres": m["genres"], "description": m["description"], "score": score})
        return {"reply_type": "recommendations", "results": results}

    if intent == "similar" or (intent is None and "like" in text.lower()):
        # attempt to find a mentioned movie title by substring match
        q = text.lower()
        found = None
        for i, row in recommender.df.iterrows():
            if str(row["title"]).lower() in q:
                found = row["title"]
                break
        if found is None:
            # try a looser match: look for any word in title
            tokens = [t for t in q.split() if len(t) > 2]
            for t in tokens:
                for i, row in recommender.df.iterrows():
                    if t in str(row["title"]).lower():
                        found = row["title"]
                        break
                if found:
                    break
        if found:
            sim = recommender.recommend_similar(found, top_k=5)
            results = []
            for idx, score in sim:
                m = recommender.get_movie(idx)
                results.append({"title": m["title"], "genres": m["genres"], "description": m["description"], "score": score})
            return {"reply_type": "similar", "for": found, "results": results}
        else:
            return {"reply_type": "echo", "message": "I couldn't find the movie you mentioned. Try giving a full title like 'I like Inception'."}

    # Default fallback: echo plus a suggestion
    return {"reply_type": "echo", "message": f"I heard: {text}. Try asking 'recommend action movies' or mention a movie you like."}


@app.get("/")
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Movie Recommender API. Visit /docs for interactive API."}
