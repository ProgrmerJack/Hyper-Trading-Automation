import pytest

from hypertrader.utils.component_registry import registry


def test_registry_validates_all_components():
    registry.reset()
    registry.register(registry.required)
    registry.validate()


def test_registry_detects_missing_component():
    registry.reset()
    missing = set(registry.required)
    missing.pop()
    registry.register(missing)
    with pytest.raises(ValueError):
        registry.validate()
