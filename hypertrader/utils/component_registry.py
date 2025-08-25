"""Component registry ensuring all system parts are active.

The registry loads the expected component names from ``components.yaml`` and
allows modules to register the components they activate during runtime.  Before
executing trades the registry can be validated to guarantee that all required
components have been used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Set

import yaml

COMPONENT_FILE = Path(__file__).resolve().parents[1] / "components.yaml"


@dataclass
class ComponentRegistry:
    """Track active components and validate against required set."""

    required: Set[str] = field(default_factory=set)
    active: Set[str] = field(default_factory=set)

    def register(self, names: Iterable[str]) -> None:
        """Mark one or more components as active."""
        self.active.update(str(n) for n in names)

    def reset(self) -> None:
        """Clear active component tracking."""
        self.active.clear()

    def validate(self) -> None:
        """Raise ``ValueError`` if any required components are missing."""
        missing = self.required - self.active
        if missing:
            raise ValueError(f"Inactive components: {sorted(missing)}")


def _load_registry() -> ComponentRegistry:
    data = yaml.safe_load(COMPONENT_FILE.read_text()) or {}
    comps = set(data.get("components", []))
    return ComponentRegistry(required=comps)


registry = _load_registry()

__all__ = ["registry", "ComponentRegistry"]
