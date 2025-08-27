from hypertrader.economics.fees import FeeModel
from hypertrader.economics.slippage import SlippageModel

class CostModel:
    def __init__(self, fee_model: FeeModel, slippage_model: SlippageModel):
        self.fee_model = fee_model
        self.slippage_model = slippage_model

    def estimate_cost(self, symbol: str, side: str, qty: float, price: float) -> float:
        # Estimate fees
        fee = self.fee_model.fee(side, qty, price, maker_fill=False)  # Assume taker for now
        # Estimate slippage
        slippage = self.slippage_model.estimate_slippage(symbol, qty, price)
        return fee + slippage
