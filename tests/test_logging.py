import json

from hypertrader.utils.logging import get_logger, log_json


def test_log_json(capsys):
    logger = get_logger("test")
    log_json(logger, "event_name", foo="bar")
    out = capsys.readouterr().err.strip()
    data = json.loads(out)
    assert data["event"] == "event_name"
    assert data["foo"] == "bar"
