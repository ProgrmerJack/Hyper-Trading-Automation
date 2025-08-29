"""Purged K-Fold with embargo for financial time series."""
from dataclasses import dataclass
from typing import Iterator, Tuple
import numpy as np

@dataclass
class PurgedKFold:
    n_splits: int = 5
    embargo: float = 0.0  # fraction of samples to embargo around test split boundaries

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_sizes = (n // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_mask = np.ones(n, dtype=bool)
            # purge overlapping info (here we assume sequential dependence)
            train_mask[start:stop] = False
            if self.embargo > 0:
                e = int(self.embargo * n)
                lo = max(0, start - e)
                hi = min(n, stop + e)
                train_mask[lo:hi] = False
            train_idx = indices[train_mask]
            yield train_idx, test_idx
            current = stop
