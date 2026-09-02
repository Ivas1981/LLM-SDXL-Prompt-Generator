from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime


class DebugLog:
    def __init__(self, enabled: bool, path: str | Path | None = None):
        self.enabled = enabled
        self._path = Path(path) if path else Path("debug.log")
        self._file = None
        if self.enabled:
            self._file = open(self._path, "w", encoding="utf-8")
            self._write(f"=== DEBUG session started at {datetime.now().isoformat()} ===\n")

    def _write(self, text: str) -> None:
        if self._file is None:
            return
        try:
            self._file.write(text)
            self._file.flush()
        except Exception:
            pass

    def log(self, label: str, data: str) -> None:
        if not self.enabled:
            return
        header = f"[{datetime.now().strftime('%H:%M:%S')}] {label}\n"
        self._write(header)
        self._write(f"{data}\n\n")

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._write(f"=== DEBUG session ended at {datetime.now().isoformat()} ===\n\n")
            self._file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_debug: DebugLog | None = None


def init(enabled: bool, path: str | Path | None = None) -> DebugLog:
    global _debug
    _debug = DebugLog(enabled=enabled, path=path)
    return _debug


def get() -> DebugLog | None:
    return _debug
