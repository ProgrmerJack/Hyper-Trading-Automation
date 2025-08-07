import pandas as pd

from hypertrader.utils.volatility import rank_symbols_by_volatility


def test_rank_symbols_by_volatility():
    # synthetic data: symbol B has larger range -> higher ATR
    data = {
        "A": pd.DataFrame(
            {
                "open": [1, 1, 1],
                "high": [2, 2, 2],
                "low": [0, 0, 0],
                "close": [1, 1, 1],
            }
        ),
        "B": pd.DataFrame(
            {
                "open": [1, 1, 1],
                "high": [3, 3, 3],
                "low": [0, 0, 0],
                "close": [1, 1, 1],
            }
        ),
    }

    def fetcher(symbol: str) -> pd.DataFrame:
        return data[symbol]

    ranked = rank_symbols_by_volatility(["A", "B"], period=2, data_fetcher=fetcher)
    assert ranked[0] == "B"
    assert set(ranked) == {"A", "B"}


def test_rank_symbols_by_volatility_handles_errors():
    data = {
        "A": pd.DataFrame(
            {
                "open": [1, 1, 1],
                "high": [2, 2, 2],
                "low": [0, 0, 0],
                "close": [1, 1, 1],
            }
        )
    }

    def fetcher(symbol: str) -> pd.DataFrame:
        if symbol == "B":
            raise ValueError("bad symbol")
        return data[symbol]

    ranked = rank_symbols_by_volatility(["A", "B"], period=2, data_fetcher=fetcher)
    assert ranked == ["A"]
