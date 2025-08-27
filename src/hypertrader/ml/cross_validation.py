"""
Purged K-Fold Cross-Validation with Embargo for Financial Time Series
Implements proper CV to avoid leakage and evaluate PBO/DSR.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator, Optional, Dict, Any
import logging
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import log_loss, accuracy_score
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold CV with embargo to prevent data leakage in financial ML.
    
    Essential for time series to avoid look-ahead bias from overlapping samples.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_td: pd.Timedelta = pd.Timedelta('1H'),
        purge_td: pd.Timedelta = pd.Timedelta('30min'),
        pct_embargo: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td
        self.pct_embargo = pct_embargo
        
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged and embargoed train/test splits.
        
        Args:
            X: Feature matrix with datetime index
            y: Target series with datetime index
            groups: Not used
            
        Yields:
            (train_idx, test_idx) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for temporal splits")
            
        indices = np.arange(len(X))
        test_ranges = self._get_test_ranges(X.index)
        
        for test_start_idx, test_end_idx in test_ranges:
            # Test set indices
            test_idx = indices[test_start_idx:test_end_idx]
            
            # Train set with purging and embargo
            train_idx = self._get_purged_train_idx(
                X.index, test_idx, test_start_idx, test_end_idx
            )
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx
            
    def _get_test_ranges(self, index: pd.DatetimeIndex) -> List[Tuple[int, int]]:
        """Divide timeline into test ranges."""
        test_ranges = []
        split_size = len(index) // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.n_splits - 1 else len(index)
            test_ranges.append((start_idx, end_idx))
            
        return test_ranges
        
    def _get_purged_train_idx(
        self, 
        index: pd.DatetimeIndex,
        test_idx: np.ndarray,
        test_start_idx: int,
        test_end_idx: int
    ) -> np.ndarray:
        """Get training indices with purging and embargo applied."""
        # Start with all non-test indices
        train_mask = np.ones(len(index), dtype=bool)
        train_mask[test_idx] = False
        
        # Apply purging (remove samples too close before test period)
        if test_start_idx > 0:
            purge_cutoff = index[test_start_idx] - self.purge_td
            purge_mask = index < purge_cutoff
            train_mask &= purge_mask
            
        # Apply embargo (remove samples too close after test period)  
        if test_end_idx < len(index):
            embargo_cutoff = index[test_end_idx - 1] + self.embargo_td
            embargo_mask = index > embargo_cutoff
            # Only keep post-test data if it's after embargo
            post_test_mask = (index > index[test_end_idx - 1]) & embargo_mask
            pre_test_mask = index < index[test_start_idx]
            train_mask &= (pre_test_mask | post_test_mask)
            
        return np.where(train_mask)[0]


class ModelEvaluator:
    """
    Evaluate models with financial ML metrics: PBO, DSR, PSR.
    Prevents promotion of overfit models.
    """
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        
    def probability_backtest_overfitting(
        self,
        returns_is: np.ndarray,
        returns_oos: np.ndarray,
        n_trials: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate Probability of Backtest Overfitting (PBO).
        
        Args:
            returns_is: In-sample strategy returns
            returns_oos: Out-of-sample strategy returns  
            n_trials: Number of strategy trials
            
        Returns:
            Dict with PBO metrics
        """
        if n_trials is None:
            n_trials = self.n_trials
            
        # Ensure arrays are same length
        min_len = min(len(returns_is), len(returns_oos))
        returns_is = returns_is[:min_len]
        returns_oos = returns_oos[:min_len]
        
        # Calculate Sharpe ratios
        sharpe_is = self._sharpe_ratio(returns_is)
        sharpe_oos = self._sharpe_ratio(returns_oos)
        
        # PBO calculation
        # Rank strategies by IS performance
        is_ranks = stats.rankdata(-returns_is.reshape(-1, 1).mean(axis=1) if returns_is.ndim > 1 else [-returns_is.mean()])
        
        # Check OOS performance of top IS performers
        if returns_oos.ndim == 1:
            returns_oos = returns_oos.reshape(-1, 1)
            
        top_is_strategies = np.argsort(is_ranks)[:max(1, n_trials // 2)]
        
        # Count how many top IS strategies underperform in OOS
        underperform_count = 0
        for strategy_idx in top_is_strategies:
            if strategy_idx < len(returns_oos) and returns_oos[strategy_idx] < 0:
                underperform_count += 1
                
        pbo = underperform_count / len(top_is_strategies) if top_is_strategies.size > 0 else 1.0
        
        return {
            'pbo': pbo,
            'sharpe_is': sharpe_is,
            'sharpe_oos': sharpe_oos,
            'n_strategies': len(top_is_strategies),
            'underperform_count': underperform_count
        }
        
    def deflated_sharpe_ratio(
        self,
        observed_sr: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0,
        kurtosis: float = 3
    ) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio to account for multiple testing.
        
        Args:
            observed_sr: Observed Sharpe ratio
            n_trials: Number of strategy trials tested
            n_observations: Number of return observations
            skewness: Return skewness
            kurtosis: Return kurtosis
            
        Returns:
            Dict with DSR metrics
        """
        if n_observations <= 1 or n_trials <= 1:
            return {'dsr': np.nan, 'pvalue': 1.0, 'threshold_sr': np.nan}
            
        # Expected maximum Sharpe ratio under null hypothesis
        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
        
        # Variance of maximum Sharpe ratio
        var_max_sr = 1 / np.sqrt(2 * np.log(n_trials)) * (1 + gamma / np.sqrt(2 * np.log(n_trials)))
        
        # Adjustment for non-normal returns
        moments_adj = (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) / n_observations
        
        # Deflated Sharpe ratio
        dsr = (observed_sr - expected_max_sr) / np.sqrt(var_max_sr + moments_adj)
        
        # p-value (probability of observing this SR by chance)
        pvalue = 2 * (1 - stats.norm.cdf(abs(dsr)))
        
        # Threshold Sharpe ratio for significance
        threshold_sr = expected_max_sr + 1.96 * np.sqrt(var_max_sr + moments_adj)
        
        return {
            'dsr': dsr,
            'pvalue': pvalue,
            'threshold_sr': threshold_sr,
            'expected_max_sr': expected_max_sr,
            'is_significant': pvalue < 0.05
        }
        
    def probabilistic_sharpe_ratio(
        self,
        returns: np.ndarray,
        benchmark_sr: float = 0.0,
        freq: int = 252
    ) -> Dict[str, float]:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        Args:
            returns: Strategy returns
            benchmark_sr: Benchmark Sharpe ratio
            freq: Return frequency (252 for daily)
            
        Returns:
            Dict with PSR metrics
        """
        if len(returns) <= 1:
            return {'psr': 0.0, 'pvalue': 1.0}
            
        observed_sr = self._sharpe_ratio(returns) * np.sqrt(freq)
        n_obs = len(returns)
        
        # Skewness and kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=True)
        
        # PSR statistic
        psr_stat = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2)
        
        # p-value
        pvalue = stats.norm.cdf(psr_stat)
        
        return {
            'psr': psr_stat,
            'pvalue': pvalue,
            'observed_sr': observed_sr,
            'is_better_than_benchmark': pvalue > 0.95
        }
        
    def _sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        excess_returns = returns - risk_free
        return excess_returns.mean() / (excess_returns.std() + 1e-8)
        
    def comprehensive_evaluation(
        self,
        returns_is: np.ndarray,
        returns_oos: np.ndarray,
        n_trials: int,
        benchmark_sr: float = 0.0
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics.
        
        Returns:
            Dict with all evaluation metrics and pass/fail decisions
        """
        # PBO evaluation
        pbo_results = self.probability_backtest_overfitting(returns_is, returns_oos, n_trials)
        
        # DSR evaluation
        observed_sr = self._sharpe_ratio(returns_oos)
        dsr_results = self.deflated_sharpe_ratio(
            observed_sr, n_trials, len(returns_oos),
            stats.skew(returns_oos), stats.kurtosis(returns_oos, fisher=True)
        )
        
        # PSR evaluation
        psr_results = self.probabilistic_sharpe_ratio(returns_oos, benchmark_sr)
        
        # Overall assessment
        passes_pbo = pbo_results['pbo'] < 0.5  # Less than 50% chance of overfitting
        passes_dsr = dsr_results.get('is_significant', False)
        passes_psr = psr_results.get('is_better_than_benchmark', False)
        
        overall_pass = passes_pbo and (passes_dsr or passes_psr)
        
        return {
            'pbo_results': pbo_results,
            'dsr_results': dsr_results,
            'psr_results': psr_results,
            'passes_pbo': passes_pbo,
            'passes_dsr': passes_dsr,
            'passes_psr': passes_psr,
            'overall_pass': overall_pass,
            'recommendation': 'PROMOTE' if overall_pass else 'REJECT'
        }


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_func,
    train_window: pd.Timedelta = pd.Timedelta('30D'),
    test_window: pd.Timedelta = pd.Timedelta('7D'),
    step_size: pd.Timedelta = pd.Timedelta('1D')
) -> pd.DataFrame:
    """
    Walk-forward validation for time series models.
    
    Args:
        X: Feature matrix with datetime index
        y: Target series with datetime index
        model_func: Function that takes (X_train, y_train) and returns fitted model
        train_window: Training window size
        test_window: Testing window size  
        step_size: Step size between iterations
        
    Returns:
        DataFrame with predictions and actual values
    """
    results = []
    
    start_date = X.index.min() + train_window
    end_date = X.index.max() - test_window
    
    current_date = start_date
    
    while current_date <= end_date:
        # Define windows
        train_end = current_date
        train_start = train_end - train_window
        test_start = current_date
        test_end = test_start + test_window
        
        # Get data
        X_train = X.loc[train_start:train_end]
        y_train = y.loc[train_start:train_end]
        X_test = X.loc[test_start:test_end]
        y_test = y.loc[test_start:test_end]
        
        if len(X_train) < 10 or len(X_test) < 1:
            current_date += step_size
            continue
            
        try:
            # Fit model
            model = model_func(X_train, y_train)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test)
                
            # Store results
            for i, (timestamp, actual, pred) in enumerate(zip(X_test.index, y_test, y_pred)):
                results.append({
                    'timestamp': timestamp,
                    'actual': actual,
                    'predicted': pred,
                    'train_start': train_start,
                    'train_end': train_end
                })
                
        except Exception as e:
            logger.warning(f"Model training failed for window {train_start} to {train_end}: {e}")
            
        current_date += step_size
        
    if not results:
        logger.error("No successful predictions in walk-forward validation")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    results_df.set_index('timestamp', inplace=True)
    
    logger.info(f"Walk-forward validation completed: {len(results_df)} predictions")
    return results_df
