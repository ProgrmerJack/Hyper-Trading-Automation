"""Combinatorially Symmetric Cross-Validation (CSCV) to estimate PBO.
Reference: Bailey et al., The Probability of Backtest Overfitting.
This is a compact educational implementation.
"""
import itertools
import numpy as np

def cscv_scores(strategy_matrix: np.ndarray, k_test: int = 2):
    """Given matrix of shape (n_periods, n_strategies) with period returns,
    compute CSCV logit scores and PBO estimate.
    """
    T, M = strategy_matrix.shape
    assert T >= k_test, "Not enough periods"
    # enumerate all test splits of size k_test
    from math import comb
    idx = np.arange(T)
    test_sets = list(itertools.combinations(idx, k_test))
    logits = []
    wins = 0; total = 0
    for tst in test_sets:
        trn = np.array(sorted(set(idx)-set(tst)))
        # pick best on train
        mu_tr = strategy_matrix[trn].mean(0)
        j = int(mu_tr.argmax())
        # see its rank on test
        mu_te = strategy_matrix[tst].mean(0)
        ranks = mu_te.argsort().argsort()  # 0 .. M-1
        rj = ranks[j]
        u = (rj + 1) / (M + 1)  # uniformized rank
        logit = np.log(u/(1-u))
        logits.append(logit)
        wins += (rj == M-1); total += 1
    logits = np.array(logits, dtype=float)
    pbo = 1.0 - (wins/total)  # proxy: probability best-in-sample is not best out-of-sample
    return logits, float(pbo)
