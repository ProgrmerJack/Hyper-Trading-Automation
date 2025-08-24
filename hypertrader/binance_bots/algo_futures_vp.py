from __future__ import annotations
from typing import Any, Dict
def futures_vp_new_order(http_client, symbol: str, side: str, quantity: float, participationRate: float, maxTime: int, positionSide: str|None=None) -> Dict[str,Any]:
    payload={'symbol':symbol.replace('/',''), 'side':side, 'quantity':quantity, 'participationRate':participationRate, 'maxTime':maxTime}
    if positionSide is not None: payload['positionSide']=positionSide
    return {'endpoint':'/fapi/v1/algo/futures/newOrderVp','data':payload}
