"""
Clipboard-based automatic item capture (Mode 1).

How it works:
  1. In D4 Companion → Settings → General → enable "Copy item to clipboard"
     (D4 Companion OCRs each tooltip you hover and copies the text automatically)
  2. Run:  python capture_items.py --mode clipboard --output stash.json
  3. Hover items in D4 — they are parsed and saved silently in real time.

No DLL injection. No game memory access. Works via standard OS clipboard API.

Cross-platform clipboard read:
  - Windows: ctypes → user32.OpenClipboard / GetClipboardData
  - macOS:   subprocess → pbpaste
  - Linux:   subprocess → xclip / xsel
"""

from __future__ import annotations

import ctypes
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from d4_optimizer.core.engine import Item
from d4_optimizer.core.tooltip_parser import parse_tooltip, _item_to_dict


# ---------------------------------------------------------------------------
# Cross-platform clipboard reader
# ---------------------------------------------------------------------------

_ITEM_POWER_SIGNAL = re.compile(r"item\s+power\s*:?\s*\d{3,4}", re.IGNORECASE)


def _read_clipboard() -> str:
    """Return current clipboard text, or empty string on any failure."""
    if sys.platform == "win32":
        return _read_clipboard_windows()
    if sys.platform == "darwin":
        return _read_clipboard_macos()
    return _read_clipboard_linux()


def _read_clipboard_windows() -> str:
    CF_UNICODETEXT = 13
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    if not user32.OpenClipboard(None):
        return ""
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return ""
        ptr = kernel32.GlobalLock(h)
        if not ptr:
            return ""
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(h)
    except Exception:
        return ""
    finally:
        user32.CloseClipboard()


def _read_clipboard_macos() -> str:
    import subprocess
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=1)
        return result.stdout
    except Exception:
        return ""


def _read_clipboard_linux() -> str:
    import subprocess
    for cmd in (["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return ""


def _looks_like_d4_tooltip(text: str) -> bool:
    """Quick check whether clipboard text looks like a D4 item tooltip."""
    return bool(_ITEM_POWER_SIGNAL.search(text))


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------


class ClipboardMonitor:
    """
    Polls the clipboard at `interval_ms` and auto-parses any D4 tooltip text.

    Calls `on_item_captured(item)` for each successfully parsed item.
    Calls `on_error(text, exc)` when parsing fails (optional).
    """

    def __init__(
        self,
        on_item_captured: Callable[[Item], None],
        on_error: Optional[Callable[[str, Exception], None]] = None,
        interval_ms: int = 150,
    ) -> None:
        self._on_item = on_item_captured
        self._on_error = on_error
        self._interval = interval_ms / 1000.0
        self._last_text: str = ""
        self._running = False

    def start(self) -> None:
        """Run the monitor loop (blocking). Call stop() from another thread to exit."""
        self._running = True
        # Capture the initial clipboard state so we don't re-parse stale content.
        self._last_text = _read_clipboard()

        print("[ClipboardMonitor] Running — hover items in D4 to capture them.")
        print("[ClipboardMonitor] Press Ctrl+C to stop.\n")

        try:
            while self._running:
                current = _read_clipboard()
                if current != self._last_text:
                    self._last_text = current
                    if _looks_like_d4_tooltip(current):
                        self._process(current)
                time.sleep(self._interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def _process(self, text: str) -> None:
        try:
            item = parse_tooltip(text)
        except ValueError as exc:
            if self._on_error:
                self._on_error(text, exc)
            else:
                print(f"  [skip] Could not parse tooltip: {exc}", file=sys.stderr)
            return

        if item.slot == "unknown":
            print(
                f"  [skip] Slot not detected for '{item.name}'. "
                "Paste it manually with --slot <slot>.",
                file=sys.stderr,
            )
            return

        self._on_item(item)


# ---------------------------------------------------------------------------
# File-writing convenience wrapper
# ---------------------------------------------------------------------------


def monitor_to_file(
    output_path: Path,
    append: bool = False,
    interval_ms: int = 150,
) -> None:
    """
    Run a clipboard monitor that saves captured items to `output_path`.

    Args:
        output_path: JSON file to write (created if absent).
        append:      If True, existing items in the file are preserved.
        interval_ms: Clipboard polling interval.
    """
    items: list[dict] = []

    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            items = json.load(fh)
        print(f"[ClipboardMonitor] Loaded {len(items)} existing item(s) from {output_path}")

    captured_count = 0

    def _save(item: Item) -> None:
        nonlocal captured_count
        entry = _item_to_dict(item)
        items.append(entry)
        captured_count += 1
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2)
        print(
            f"  [{captured_count:>4}] Captured: {item.name} [{item.slot}] "
            f"IP:{item.item_power}  ({len(item.affixes)} affixes)"
        )

    monitor = ClipboardMonitor(on_item_captured=_save, interval_ms=interval_ms)
    monitor.start()

    print(f"\n[ClipboardMonitor] Session ended. {captured_count} item(s) saved to {output_path}")
