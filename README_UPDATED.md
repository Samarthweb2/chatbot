# chatbot

A simple movie recommendation chatbot example using Python and scikit-learn.

This repository provides `main.py`, a small CLI chatbot that recommends movies
using TF-IDF on movie genres+descriptions and cosine similarity.

## Usage

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the chatbot:

```powershell
python main.py
```

Try queries like:
- "recommend sci-fi"
- "3 romantic comedies"
- "i like the matrix"

If you place a `movies.csv` file in the repository root with columns
`title,genres,description`, the script will load it. Otherwise a small
sample dataset is used.

License

MIT (add LICENSE file if you publish this repository)
