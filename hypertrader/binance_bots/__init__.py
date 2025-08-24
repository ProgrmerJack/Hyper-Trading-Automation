"""Binance-style bot implementations & algo wrappers.
Order intents are dicts: { 'symbol','side','type','price'?, 'qty','tags':{...} }.
"""
from .spot_grid import SpotGrid
from .futures_grid import FuturesGrid
from .rebalancing import RebalancingBot
from .spot_dca import SpotDCA
from .funding_arb import FundingArbBot
from .algo_spot_twap import spot_twap_new_order, spot_twap_cancel, spot_twap_history
from .algo_futures_twap import futures_twap_new_order
from .algo_futures_vp import futures_vp_new_order
