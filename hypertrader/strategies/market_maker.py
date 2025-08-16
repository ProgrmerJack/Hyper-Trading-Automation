"""
Market maker strategy using the Avellaneda–Stoikov framework.

This module implements a simplified version of the market making
strategy described by Avellaneda and Stoikov.  The strategy posts
limit orders on both sides of the order book around a reservation
price and manages inventory risk by adjusting the quoted spread
according to the current position.  The theoretical formulas used
here come from the original Avellaneda–Stoikov paper, but many
details have been simplified to make the example self‑contained.

Key Features
------------

* Reservation price shifts mid‑price toward the current inventory to
  manage position risk.
* Optimal spread is computed as a function of risk aversion,
  volatility and order book parameters (``gamma`` and ``kappa``).
* Strategy updates its inventory when orders are filled and adjusts
  quotes accordingly.
* Uses microprice as a short‑horizon estimator for the fundamental
  price, enhancing responsiveness to order book imbalance.

For production deployment, you should integrate this module with a
live order management system (OMS) capable of maintaining queue
position, handling partial fills and managing multiple instruments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ..indicators.microstructure import compute_microprice


@dataclass
class MarketMakerStrategy:
    """Simplified Avellaneda–Stoikov market making strategy.

    Parameters
    ----------
    symbol : str
        Trading pair this strategy operates on (e.g., ``"BTC/USDT"``).
    gamma : float
        Risk aversion parameter.  Higher values result in wider
        spreads and less aggressive quoting.
    kappa : float
        Order book resilience parameter controlling the mean
        reversion speed of the order imbalance.  Typical values are
        small (e.g., 1–5).
    sigma : float
        Estimate of the asset's instantaneous volatility (e.g., one
        second realised volatility).  Used to compute the optimal
        spread.
    base_order_size : float
        Nominal quantity to quote on each side.  Dynamic sizing can
        be layered on top by the bot.
    inventory : float, optional
        Current inventory of the asset.  The strategy will adjust
        its reservation price based on this.  Defaults to zero.
    """

    symbol: str
    gamma: float
    kappa: float
    sigma: float
    base_order_size: float
    inventory: float = 0.0
    # Additional state variables to track outstanding orders and fills
    bid_price: Optional[float] = field(default=None, init=False)
    ask_price: Optional[float] = field(default=None, init=False)

    def update_parameters(self, gamma: Optional[float] = None, kappa: Optional[float] = None, sigma: Optional[float] = None) -> None:
        """Update model parameters on the fly.

        In practice, parameters like volatility and order book
        resilience should be estimated from recent data.  This
        convenience method allows dynamic reconfiguration without
        recreating the strategy object.
        """
        if gamma is not None:
            self.gamma = gamma
        if kappa is not None:
            self.kappa = kappa
        if sigma is not None:
            self.sigma = sigma

    def compute_reservation_price(self, mid_price: float) -> float:
        """Calculate the reservation price (indifference price).

        The reservation price tilts the mid price toward reducing
        inventory.  It is given by:

        .. math::

           r_t = m_t - \frac{I_t}{\gamma} \sigma^2

        where ``m_t`` is the microprice (an estimator of the fair
        value), ``I_t`` is the current inventory, ``gamma`` is the
        risk aversion parameter and ``sigma`` is the volatility.  In
        Avellaneda–Stoikov, this term also depends on the time
        remaining until terminal horizon; here we assume a unit time
        horizon for simplicity.
        """
        return mid_price - (self.inventory * self.sigma ** 2) / self.gamma

    def compute_optimal_spread(self) -> float:
        """Compute the optimal half spread based on model parameters.

        The optimal spread in Avellaneda–Stoikov is:

        .. math::

           s = \frac{\gamma \sigma^2}{\kappa} + \frac{2}{\gamma} \log\left(1 + \frac{\gamma}{\kappa}\right)

        This simplified version omits time‑dependency and assumes
        constant ``sigma`` and ``kappa``.  The full expression also
        includes a term for inventory risk and a factor of 0.5 when
        quoting each side.  Here we return the half‑spread (one side).
        """
        a = (self.gamma * self.sigma ** 2) / self.kappa
        b = (2.0 / self.gamma) * math.log(1 + self.gamma / self.kappa)
        return 0.5 * (a + b)

    def quote(self, order_book: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
        """Compute optimal bid and ask prices given the current order book.

        Parameters
        ----------
        order_book : dict
            Mapping containing lists of bids and asks (see
            :func:`compute_microprice`).  Only the top of book is
            required but additional depth can refine the microprice.

        Returns
        -------
        (bid_price, ask_price) : tuple of float
            The prices at which to post buy and sell orders.
        """
        # Use microprice as a proxy for the fundamental price
        mid = compute_microprice(order_book, depth=1)
        # Determine reservation price and spread
        r_t = self.compute_reservation_price(mid)
        s = self.compute_optimal_spread()
        # Quote around the reservation price
        bid = r_t - s
        ask = r_t + s
        # Persist current quotes
        self.bid_price = bid
        self.ask_price = ask
        return bid, ask

    def update_inventory(self, filled_buy: float = 0.0, filled_sell: float = 0.0) -> None:
        """Update the internal inventory based on filled orders.

        Parameters
        ----------
        filled_buy : float
            Amount bought since last update.
        filled_sell : float
            Amount sold since last update.

        Notes
        -----
        Inventory increases when we buy and decreases when we sell.
        """
        self.inventory += filled_buy - filled_sell

    def generate_orders(self, order_book: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[str, float, float]]:
        """Return a list of orders to send to the connector.

        Each order is represented as a tuple ``(side, price, quantity)``.
        The strategy always posts one buy and one sell order using
        ``base_order_size``.  The caller may choose to adjust the
        quantity using dynamic sizing before executing the orders.
        """
        bid, ask = self.quote(order_book)
        return [
            ("buy", bid, self.base_order_size),
            ("sell", ask, self.base_order_size),
        ]
