"""Optional neural model utilities with safe fallbacks.

This module demonstrates how you might plug in PyTorch or TensorFlow models
without making them required for the repo to run. If torch/tensorflow are not
installed we fall back to a simple TF-IDF embedding (scikit-learn).
"""
from typing import List

# lazy imports
_torch_available = None
_tf_available = None


def _check_torch():
    global _torch_available
    if _torch_available is None:
        try:
            import torch  # type: ignore
            _torch_available = True
        except Exception:
            _torch_available = False
    return _torch_available


def _check_tf():
    global _tf_available
    if _tf_available is None:
        try:
            import tensorflow as tf  # type: ignore
            _tf_available = True
        except Exception:
            _tf_available = False
    return _tf_available


def embed_text(texts: List[str]):
    """Return embeddings for a list of texts.

    If PyTorch or TensorFlow are available, this is where you'd call a model.
    Otherwise falls back to simple TF-IDF vectors (dense numpy arrays).
    """
    if _check_torch():
        import torch
        # Placeholder: a real project would load a model and run tokenizer + model
        # For demonstration, we'll return random vectors of fixed size.
        vecs = torch.randn(len(texts), 768)
        return vecs.numpy()

    if _check_tf():
        import numpy as _np
        # Placeholder: return random vectors
        return _np.random.randn(len(texts), 768)

    # fallback: TF-IDF (scikit-learn)
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as _np
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(texts)
    return mat.toarray()
