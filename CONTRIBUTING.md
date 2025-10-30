# Contributing

Thanks for your interest in contributing! A few small conventions to keep the repo tidy:

- Create a branch per PR (feature/..., fix/...).
- Write small commits with clear messages.
- Add unit tests for new logic and run `pytest` locally.
- Follow basic linting and typing conventions (consider adding ruff/mypy later).

To run tests locally:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
pytest -q
```
