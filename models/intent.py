"""Intent classifier with scikit-learn and optional PyTorch fallback.

This module provides a small IntentClassifier with a `predict` API.
It trains a tiny supervised model on a few examples at startup. If PyTorch
is available, a future extension could use it; for now we prefer the
lightweight scikit-learn pipeline so the repo remains easy to run.
"""
from typing import List, Optional
import os

try:
    import joblib
except Exception:
    joblib = None

try:
    # sklearn is a project dependency
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except Exception:
    raise


class IntentClassifier:
    """Train a tiny intent classifier with TF-IDF + LogisticRegression.

    If PyTorch is available we use a tiny torch model; otherwise scikit-learn
    pipeline is used for simplicity and fast startup.
    """

    def __init__(self):
        # small training set — extend as needed
        self.texts = [
            "recommend action movies",
            "suggest something romantic",
            "i like the matrix",
            "find movies like inception",
            "hello",
            "hi",
            "bye",
            "thanks",
        ]
        self.labels = [
            "recommend",
            "recommend",
            "similar",
            "similar",
            "greet",
            "greet",
            "goodbye",
            "thanks",
        ]

        # prefer a torch-based classifier when torch is installed
        self._use_torch = False
        try:
            import torch  # type: ignore
            self._use_torch = True
        except Exception:
            self._use_torch = False

        if self._use_torch:
            self._init_torch_model()
        else:
            self._init_sklearn()

    def _init_sklearn(self):
        self.pipeline = make_pipeline(TfidfVectorizer(stop_words="english"), LogisticRegression(solver="liblinear"))
        self.pipeline.fit(self.texts, self.labels)

    def save(self, path: str) -> None:
        """Save the classifier to disk (sklearn pipeline or torch artifacts)."""
        # prefer joblib if available
        if joblib is None:
            raise RuntimeError("joblib is required to save the model; please install joblib")

        # If using torch, save the small torch model plus vectorizer and classes
        if self._use_torch:
            to_save = {
                "torch_state_dict": self._model.state_dict(),
                "classes": self._classes,
            }
            # joblib can store dicts; also persist the TF-IDF vectorizer separately
            joblib.dump({"torch": to_save, "tfidf": self.tfidf}, path)
        else:
            joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "IntentClassifier":
        """Load a persisted IntentClassifier from disk (joblib file).

        Returns an IntentClassifier instance with loaded internals.
        """
        if joblib is None:
            raise RuntimeError("joblib is required to load the model; please install joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        obj = joblib.load(path)
        inst = cls.__new__(cls)
        # set texts/labels to empty lists to avoid retraining
        inst.texts = []
        inst.labels = []
        inst._use_torch = False
        # detect format
        if hasattr(obj, "predict"):
            # sklearn pipeline
            inst.pipeline = obj
            inst._use_torch = False
            return inst

        # dict format for torch
        if isinstance(obj, dict) and "torch" in obj:
            try:
                import torch
            except Exception:
                raise RuntimeError("torch is required to load a torch-based intent model")
            data = obj["torch"]
            inst._use_torch = True
            inst._torch = torch
            inst._classes = data["classes"]
            # restore tfidf vectorizer
            inst.tfidf = obj.get("tfidf")
            input_dim = len(inst.tfidf.get_feature_names_out()) if inst.tfidf is not None else 0
            num_classes = len(inst._classes)
            inst._model = torch.nn.Linear(input_dim, num_classes)
            inst._model.load_state_dict(data["torch_state_dict"])
            return inst

        raise RuntimeError("Unrecognized model format in joblib file")

    def _init_torch_model(self):
        # A tiny Torch model: TF-IDF features -> linear classifier trained for a few epochs.
        import torch
        import numpy as _np
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(stop_words="english")
        X = self.tfidf.fit_transform(self.texts).toarray().astype(_np.float32)
        classes = sorted(set(self.labels))
        self._classes = classes
        y = _np.array([classes.index(l) for l in self.labels], dtype=_np.int64)

        input_dim = X.shape[1]
        num_classes = len(classes)
        self._torch = torch
        self._model = torch.nn.Linear(input_dim, num_classes)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self._model.parameters(), lr=0.01)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        self._model.train()
        for epoch in range(50):
            opt.zero_grad()
            logits = self._model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

    def predict(self, text: str) -> str:
        """Return the predicted intent label for the given text."""
        if not text or not text.strip():
            return "unknown"
        if self._use_torch:
            # compute TF-IDF using stored vectorizer
            X = self.tfidf.transform([text]).toarray().astype('float32')
            xt = self._torch.from_numpy(X)
            self._model.eval()
            with self._torch.no_grad():
                logits = self._model(xt)
                idx = int(self._torch.argmax(logits, dim=-1).item())
            return self._classes[idx]
        else:
            return str(self.pipeline.predict([text])[0])

    def predict_proba(self, text: str):
        """Return class probabilities (dict label->prob)."""
        if self._use_torch:
            X = self.tfidf.transform([text]).toarray().astype('float32')
            xt = self._torch.from_numpy(X)
            self._model.eval()
            with self._torch.no_grad():
                logits = self._model(xt)
                probs = self._torch.softmax(logits, dim=-1).numpy().flatten()
            return dict(zip(self._classes, probs.tolist()))
        else:
            proba = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_
            return dict(zip(classes, proba))


def predict_intent(text: str) -> str:
    ic = IntentClassifier()
    return ic.predict(text)
