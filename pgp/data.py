# data.py — Data normalization (ZScore + Log-space) for GP input standardization
#
# Consolidated from data.py (legacy) + data_log_ext.py.
# The old data_loader, parse_params, sample_train_test, maximin_split utilities
# have been removed — they were dead code not imported anywhere in the project.
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

Array = np.ndarray


@dataclass
class ZScoreNormalizer:
    """Z-score normalizer: (X - mean) / std."""
    mean_: Optional[Array] = None
    std_: Optional[Array] = None
    eps: float = 1e-12

    def fit(self, X: Array) -> "ZScoreNormalizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, self.eps, self.std_)
        return self

    def transform(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: Array) -> Array:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first"
        return Z * self.std_ + self.mean_


@dataclass
class LogStandardizer(ZScoreNormalizer):
    """log10 then z-score. Only affects GP input space.

    For parameters spanning orders of magnitude (e.g., Dx ∈ [1e-4, 1e-2]),
    log10 transform compresses the range before z-scoring. This makes GP
    lengthscale estimation much more stable.
    """
    base: float = 10.0
    clip_min: float = 1e-300
    log_mask: Optional[Array] = None

    def _log(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if self.log_mask is None:
            X = np.clip(X, self.clip_min, None)
            return np.log(X) / np.log(self.base)
        Xc = X.copy()
        if Xc.ndim == 1:
            Xc = Xc.reshape(1, -1)
        mask = self.log_mask.astype(bool)
        Xc[:, mask] = np.log(np.clip(Xc[:, mask], self.clip_min, None)) / np.log(self.base)
        return Xc

    def fit(self, X: Array) -> "LogStandardizer":
        return super().fit(self._log(X))

    def transform(self, X: Array) -> Array:
        return super().transform(self._log(X))

    def inverse_transform(self, Z: Array) -> Array:
        Xl = super().inverse_transform(Z)
        if self.log_mask is None:
            return np.power(self.base, Xl)
        X = Xl.copy()
        mask = self.log_mask.astype(bool)
        X[:, mask] = np.power(self.base, X[:, mask])
        return X


def data_normalizer(kind: str = "zscore", **kwargs):
    """Factory function for normalizers.

    Parameters
    ----------
    kind : str
        "zscore" / "z" / "standard" → ZScoreNormalizer
        "log" / "logz" / "log-zscore" / "log10" → LogStandardizer
    """
    if kind.lower() in ("z", "zscore", "standard"):
        return ZScoreNormalizer(**kwargs)
    if kind.lower() in ("log", "logz", "log-zscore", "log10"):
        return LogStandardizer(**kwargs)
    raise ValueError(f"Unknown normalizer kind: {kind}")


def data_normalizer_log(**kwargs):
    """Convenience: returns LogStandardizer (log10 → z-score)."""
    return data_normalizer(kind="logz", **kwargs)
