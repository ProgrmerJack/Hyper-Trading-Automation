"""
Multiplicative weights allocator (Hedge algorithm).

This module implements a simple version of the Hedge algorithm for
portfolio allocation among multiple trading strategies.  At each
time step, the allocator receives returns (or rewards) from each
strategy and updates a set of weights that determine how much
capital to allocate to each strategy on the next step.

Algorithm outline:

* Maintain a weight vector ``w`` for ``n`` strategies.  Initialise
  all weights equally (e.g., 1 / n).
* After each evaluation interval (e.g., a trading day or hour),
  compute the reward ``r_i`` for each strategy (this can be
  realised PnL, risk‑adjusted return or any performance metric).
* Update weights multiplicatively: ``w_i <- w_i * exp(eta * r_i)``
  where ``eta`` is a learning rate controlling how quickly the
  allocator adapts.
* Normalise weights to sum to one.

In this implementation, rewards should be scaled to fall within a
reasonable range (e.g., between -1 and 1) to prevent numerical
instability.  Negative returns decrease weights, positive returns
increase them.  A small regularisation term can be added to avoid
weights collapsing to zero.

For more information on the Hedge algorithm, see:

  * Littlestone, N. and Warmuth, M. K., "The Weighted Majority
    Algorithm" (1989).

Usage
-----

Instantiate a ``HedgeAllocator`` with the number of strategies and
optionally specify the learning rate ``eta``.  Then call
``update(rewards)`` after each time period to update the weights.
Access the current weights via the ``weights`` attribute.  See
example below:

```
alloc = HedgeAllocator(n_strategies=3, eta=0.5)
print(alloc.weights)  # [0.3333, 0.3333, 0.3333]
alloc.update([0.01, -0.02, 0.005])
print(alloc.weights)  # updated weights favouring first and third
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class HedgeAllocator:
    """Multiplicative weights allocator for strategy selection.

    Parameters
    ----------
    n_strategies : int
        Number of strategies to allocate across.
    eta : float, optional
        Learning rate controlling how aggressively to update weights.
        Default is 0.5.  Smaller values yield slower adaptation.
    min_weight : float, optional
        Minimum allowable weight for any strategy.  This prevents
        weights from collapsing to zero and keeps all strategies
        active (default 1e-6).
    """

    n_strategies: int
    eta: float = 0.5
    min_weight: float = 1e-6
    weights: List[float] = field(init=False)

    def __post_init__(self) -> None:
        # Initialise equal weights
        if self.n_strategies <= 0:
            raise ValueError("n_strategies must be positive")
        self.weights = [1.0 / self.n_strategies] * self.n_strategies

    def update(self, rewards: List[float]) -> None:
        """Update weights based on observed rewards.

        Parameters
        ----------
        rewards : list of float
            Reward or performance metric for each strategy.  Should
            have length equal to ``n_strategies``.  Values should
            ideally be scaled to fall in a small range (e.g., [-1, 1]).

        Notes
        -----
        The update rule is ``w_i <- w_i * exp(eta * r_i)``, followed
        by normalisation and clipping of weights to enforce
        ``min_weight``.
        """
        if len(rewards) != self.n_strategies:
            raise ValueError("rewards length must equal n_strategies")
        # Apply multiplicative update
        new_weights = []
        for w, r in zip(self.weights, rewards):
            # exponentiate reward scaled by eta
            factor = math.exp(self.eta * r)
            new_weights.append(w * factor)
        # Normalise
        total = sum(new_weights)
        if total == 0:
            # Avoid division by zero; reset to equal weights
            self.weights = [1.0 / self.n_strategies] * self.n_strategies
            return
        self.weights = [max(w / total, self.min_weight) for w in new_weights]
        # Renormalise to ensure weights sum to 1 after clipping
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]