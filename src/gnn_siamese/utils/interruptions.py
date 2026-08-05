"""Minimal signal handling for safe-boundary interruption."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from types import FrameType
from typing import Any


class RunInterrupted(KeyboardInterrupt):
    """Raised by the main flow at a safe boundary after a stop request."""

    def __init__(self, *, signum: int | None, reason: str) -> None:
        super().__init__(reason)
        self.signum = signum
        self.reason = reason
        self.exit_code = 143 if signum == signal.SIGTERM else 130


@dataclass
class InterruptionController:
    requested: bool = False
    signum: int | None = None
    signal_name: str | None = None
    request_count: int = 0

    def handler(self, signum: int, _frame: FrameType | None) -> None:
        self.request_count += 1
        if self.request_count == 1:
            self.requested = True
            self.signum = signum
            self.signal_name = signal.Signals(signum).name
            return
        signal.signal(signum, signal.SIG_DFL)

    def raise_if_requested(self) -> None:
        if self.requested:
            raise RunInterrupted(
                signum=self.signum,
                reason=self.signal_name or "KeyboardInterrupt",
            )

    def metadata(self) -> dict[str, Any]:
        return {
            "reason": self.signal_name or "KeyboardInterrupt",
            "signal": self.signum,
            "signal_name": self.signal_name,
            "exit_code": 143 if self.signum == signal.SIGTERM else 130,
        }

    def install(self) -> dict[int, Any]:
        previous: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handler)
        return previous

    @staticmethod
    def restore(previous: dict[int, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
