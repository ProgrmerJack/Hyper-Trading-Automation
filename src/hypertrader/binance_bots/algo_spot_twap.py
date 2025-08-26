from __future__ import annotations
from typing import Any, Dict
def spot_twap_new_order(http_client, symbol: str, side: str, quantity: float, duration: int, limitPrice: float|None=None, clientAlgoId: str|None=None) -> Dict[str,Any]:
    payload={'symbol':symbol.replace('/',''), 'side':side, 'quantity':quantity, 'duration':duration}
    if limitPrice is not None: payload['limitPrice']=limitPrice
    if clientAlgoId is not None: payload['clientAlgoId']=clientAlgoId
    return {'endpoint':'/sapi/v1/algo/spot/newOrderTwap','data':payload}
def spot_twap_cancel(http_client, algoId: int) -> Dict[str,Any]:
    return {'endpoint':'/sapi/v1/algo/spot/order','params':{'algoId':int(algoId)}, 'method':'DELETE'}
def spot_twap_history(http_client, **params) -> Dict[str,Any]:
    return {'endpoint':'/sapi/v1/algo/spot/historicalOrders','params':params, 'method':'GET'}
