# binance_bots add-on (2025-08-24T09:11:00.749950Z)
Spot Grid, Futures Grid, Rebalancing, Spot DCA, Funding Rate Arb (emulations) +
Algo wrappers: Spot TWAP, Futures TWAP, Futures VP.

Usage example:
```python
from hypertrader_plus.binance_bots import SpotGrid
grid = SpotGrid('BTC/USDT', lower=58000, upper=62000, grids=21, base_qty=0.0006)
for intent in grid.bootstrap(): oms.submit(intent)
```
