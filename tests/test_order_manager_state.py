from hypertrader.execution.order_manager import OrderManager, Order, OrderState


def test_state_transitions() -> None:
    om = OrderManager()
    order = Order(id="1", symbol="BTC/USDT", side="buy", quantity=1)
    om.submit(order)
    assert om.orders["1"].state is OrderState.PENDING
    om.on_ack("1")
    assert om.orders["1"].state is OrderState.OPEN
    om.on_fill("1", 0.4)
    assert om.orders["1"].state is OrderState.PARTIAL
    om.on_fill("1", 0.6)
    assert om.orders["1"].state is OrderState.FILLED


def test_cancel_all() -> None:
    om = OrderManager()
    o1 = Order(id="1", symbol="X", side="buy", quantity=1)
    o2 = Order(id="2", symbol="X", side="buy", quantity=1)
    om.submit(o1)
    om.submit(o2)
    om.on_ack("1")
    om.cancel_all()
    assert om.orders["1"].state is OrderState.CANCELED
    assert om.orders["2"].state is OrderState.CANCELED
