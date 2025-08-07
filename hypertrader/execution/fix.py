"""Lightweight FIX execution layer.

This module provides a minimal QuickFIX ``Application`` implementation that
can be extended for live trading. The dependency on :mod:`quickfix` is
optional so unit tests remain lightweight.
"""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import quickfix as fix
except Exception:  # pragma: no cover - quickfix not installed
    fix = None  # type: ignore


class FIXApp(fix.Application if fix else object):
    """Basic FIX application skeleton.

    The class only defines callbacks required by ``quickfix``. If the
    library is not installed an informative :class:`ImportError` is raised
    when attempting to instantiate the class.
    """

    def __init__(self):
        if fix is None:
            raise ImportError("quickfix package is required for FIXApp")
        super().__init__()

    # The following methods simply satisfy the ``quickfix.Application`` API.
    def onCreate(self, sessionID):  # pragma: no cover - behaviourless
        pass

    def onLogon(self, sessionID):  # pragma: no cover - behaviourless
        pass

    def onLogout(self, sessionID):  # pragma: no cover - behaviourless
        pass

    def toAdmin(self, message, sessionID):  # pragma: no cover - behaviourless
        pass

    def fromAdmin(self, message, sessionID):  # pragma: no cover - behaviourless
        pass

    def toApp(self, message, sessionID):  # pragma: no cover - behaviourless
        pass

    def fromApp(self, message, sessionID):  # pragma: no cover - behaviourless
        pass


__all__ = ["FIXApp"]
