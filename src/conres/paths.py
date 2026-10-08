"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_directory(path_like: str | Path) -> Path:
    """Create a directory if needed and return its absolute path."""
    path = resolve_repo_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_stdout(script: str | Path) -> Path:
    """Copy everything printed from here on to results/diagnostics/<script stem>.log."""
    import sys
    log = ensure_directory("results/diagnostics") / f"{Path(script).stem}.log"
    sys.stdout = _Tee(sys.__stdout__, log.open("w", buffering=1))
    return log
