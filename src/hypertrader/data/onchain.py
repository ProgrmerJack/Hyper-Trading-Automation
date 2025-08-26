import requests
import pandas as pd
from datetime import datetime, timezone

ETHERSCAN_API_URL = "https://api.etherscan.io/api"


def fetch_eth_gas_fees(api_key: str | None = None) -> pd.DataFrame:
    """Fetch current Ethereum gas oracle data from the Etherscan API.

    Parameters
    ----------
    api_key : str | None
        Optional Etherscan API key. If omitted the unauthenticated endpoint is used
        which is subject to stricter rate limits.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``timestamp`` and ``gas`` (gwei) indexed by timestamp.
    """
    params = {
        "module": "gastracker",
        "action": "gasoracle",
    }
    if api_key:
        params["apikey"] = api_key
    response = requests.get(ETHERSCAN_API_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json().get("result", {})
    gas_price = float(data.get("ProposeGasPrice", 0.0))
    ts = datetime.now(timezone.utc)
    df = pd.DataFrame({"timestamp": [ts], "gas": [gas_price]})
    df.set_index("timestamp", inplace=True)
    return df
