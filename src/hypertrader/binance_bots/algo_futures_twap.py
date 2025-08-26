from __future__ import annotations
from typing import Any, Dict
def futures_twap_new_order(http_client, symbol: str, side: str, quantity: float, duration: int, positionSide: str|None=None) -> Dict[str,Any]:
    payload={'symbol':symbol.replace('/',''), 'side':side, 'quantity':quantity, 'duration':duration}
    if positionSide is not None: payload['positionSide']=positionSide
    return {'endpoint':'/fapi/v1/algo/futures/newOrderTwap','data':payload}
