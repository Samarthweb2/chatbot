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

