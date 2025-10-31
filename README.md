# Movie Recommender Chatbot

This repository contains a small movie recommendation chatbot built with
scikit-learn and a lightweight FastAPI-based web UI. The project is intentionally
modular so you can later plug in PyTorch or TensorFlow models for richer NLP
capabilities.

Quick items included:
- `main.py` — core recommender implementation (TF-IDF + cosine similarity)
- `app/server.py` — FastAPI server exposing `/recommend` and `/chat` endpoints and serving a simple web UI
- `app/static/index.html` — minimal frontend to chat with the bot
- `models/nn_models.py` — optional neural model stubs with safe fallbacks

Requirements:
- Python 3.8+ (tested with 3.10+)
- Install the main dependencies:

```powershell
pip install -r requirements.txt
```

Optional heavy ML frameworks (PyTorch/TensorFlow) are listed in `requirements-optional.txt` and are not required to run the basic app.

Run the API server locally:

```powershell
# from project root
uvicorn app.server:app --reload --port 8000
```

Open http://localhost:8000/ to use the simple web UI, or visit http://localhost:8000/docs for the automatic API docs.

If you'd like, I can add a Dockerfile, CI improvements, or a fuller neural-model demo using PyTorch or TensorFlow (these require installing the optional packages). Let me know which you'd prefer next.
