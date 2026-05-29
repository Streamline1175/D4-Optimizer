"""
Win32 WinEvent accessibility hook for automatic item capture (Mode 2 — Windows only).

How it works:
  1. Enable Screen Reader in Diablo 4:
       Settings → Accessibility → Screen Reader → On
  2. Run:  python capture_items.py --mode accessibility --output stash.json
  3. Hover items normally — text is captured via Windows accessibility events,
     no clipboard, no extra tools required.

How D4 exposes tooltip text:
  When Screen Reader is enabled, D4 registers each tooltip line as an
  accessible UI object and fires EVENT_OBJECT_NAMECHANGE via WinEvents.
  We use SetWinEventHook (out-of-process, no injection) to receive these
  events, then call AccessibleObjectFromEvent + IAccessible::get_accName
  to read the text. Lines are buffered and flushed as a complete tooltip
  when a 1-second gap is detected (D4 fires all lines within ~200ms).

Requirements:
  - Windows 10/11
  - pip install pywin32 comtypes
  - Diablo 4 running with Screen Reader enabled
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import sys
import threading
import time
from typing import Callable, Optional

if sys.platform != "win32":
    raise ImportError("win_accessibility.py is Windows-only.")

# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------

WINEVENT_OUTOFCONTEXT = 0x0000          # hook runs in our process, not target
EVENT_OBJECT_NAMECHANGE = 0x800C        # accessible name changed
EVENT_OBJECT_SHOW = 0x8002              # object became visible
EVENT_OBJECT_FOCUS = 0x8005

OBJID_WINDOW = 0
OBJID_CLIENT = -4  # 0xFFFFFFFC

# ---------------------------------------------------------------------------
# ctypes declarations
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32
ole32 = ctypes.windll.ole32

WinEventProc = ctypes.WINFUNCTYPE(
    None,
    wintypes.HANDLE,   # hWinEventHook
    wintypes.DWORD,    # event
    wintypes.HWND,     # hwnd
    wintypes.LONG,     # idObject
    wintypes.LONG,     # idChild
    wintypes.DWORD,    # idEventThread
    wintypes.DWORD,    # dwmsEventTime
)


# ---------------------------------------------------------------------------
# IAccessible text extraction
# ---------------------------------------------------------------------------

def _get_accessible_name_comtypes(hwnd: int, id_object: int, id_child: int) -> Optional[str]:
    """
    Use comtypes + oleacc to retrieve the accessible name of an object.
    Returns None on any failure.
    """
    try:
        import comtypes.client
        import comtypes
        from comtypes.gen import Accessibility  # type: ignore[attr-defined]

        oleacc = ctypes.windll.LoadLibrary("oleacc.dll")

        # AccessibleObjectFromEvent signature:
        # HRESULT AccessibleObjectFromEvent(HWND hwnd, DWORD dwId, DWORD dwChildId,
        #                                   IAccessible **ppacc, VARIANT *pvarChild)
        # We use a simplified approach via AccessibleObjectFromWindow for the hwnd.
        iid_iaccessible = comtypes.GUID("{618736E0-3C3D-11CF-810C-00AA00389B71}")

        pacc = ctypes.c_void_p()
        var_child = comtypes.automation.VARIANT()
        var_child.vt = 3  # VT_I4
        var_child.value = id_child

        hr = oleacc.AccessibleObjectFromEvent(
            hwnd,
            ctypes.c_uint(id_object & 0xFFFFFFFF),
            ctypes.c_uint(id_child & 0xFFFFFFFF),
            ctypes.byref(pacc),
            ctypes.byref(var_child),
        )
        if hr != 0 or not pacc:
            return None

        # Cast to IAccessible and call get_accName
        iacc = comtypes.cast(pacc, comtypes.POINTER(Accessibility.IAccessible))
        try:
            name = iacc.accName(var_child)
            return str(name) if name else None
        except comtypes.COMError:
            return None

    except Exception:
        return None


def _get_accessible_name_fallback(hwnd: int) -> Optional[str]:
    """Fallback: read window title text — less accurate but no COM needed."""
    buf = ctypes.create_unicode_buffer(512)
    length = user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value if length > 0 else None


def _get_accessible_name(hwnd: int, id_object: int, id_child: int) -> Optional[str]:
    name = _get_accessible_name_comtypes(hwnd, id_object, id_child)
    if name:
        return name
    return _get_accessible_name_fallback(hwnd)


# ---------------------------------------------------------------------------
# Process finder
# ---------------------------------------------------------------------------

def _find_diablo4_pid() -> Optional[int]:
    """Return the PID of the running Diablo IV process, or None if not found."""
    import subprocess
    try:
        # tasklist is universally available on Windows
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Diablo IV.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 2 and "Diablo" in parts[0]:
                return int(parts[1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Tooltip text buffer
# ---------------------------------------------------------------------------

_TOOLTIP_FLUSH_TIMEOUT = 1.0  # seconds of silence → treat buffered lines as one tooltip


class _TooltipBuffer:
    """
    Accumulates lines fired by WinEvents, flushes them as a block after a
    silence period (D4 fires all tooltip lines within ~200ms).
    """

    def __init__(self, on_tooltip: Callable[[str], None]) -> None:
        self._on_tooltip = on_tooltip
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def push(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._lines.append(text)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(_TOOLTIP_FLUSH_TIMEOUT, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            if not self._lines:
                return
            block = "\n".join(self._lines)
            self._lines = []
            self._timer = None
        self._on_tooltip(block)


# ---------------------------------------------------------------------------
# Main hook class
# ---------------------------------------------------------------------------

class AccessibilityHook:
    """
    Win32 out-of-process WinEvent hook that captures D4 tooltip text.

    Calls `on_tooltip_text(raw_text)` with the complete tooltip block
    when D4 fires it through its Screen Reader channel.
    """

    def __init__(self, on_tooltip_text: Callable[[str], None]) -> None:
        self._on_tooltip = on_tooltip_text
        self._buffer = _TooltipBuffer(on_tooltip=on_tooltip_text)
        self._hook_handle: Optional[int] = None
        self._proc_ref: Optional[WinEventProc] = None  # must keep a reference to prevent GC
        self._target_pid: int = 0

    def start(self) -> None:
        """
        Find D4 process, set up the WinEvent hook, and run a message loop.
        Blocking — runs until stop() is called or the process exits.
        """
        ole32.CoInitialize(None)

        pid = _find_diablo4_pid()
        if pid is None:
            raise RuntimeError(
                "Diablo IV process not found. "
                "Start the game first, then run this script."
            )
        self._target_pid = pid
        print(f"[AccessibilityHook] Attached to Diablo IV (PID {pid})")
        print("[AccessibilityHook] Hover items in D4 to capture them.")
        print("[AccessibilityHook] Press Ctrl+C to stop.\n")

        def _callback(
            hook: int, event: int, hwnd: int,
            id_object: int, id_child: int,
            event_thread: int, event_time: int,
        ) -> None:
            text = _get_accessible_name(hwnd, id_object, id_child)
            if text:
                self._buffer.push(text)

        # Keep strong reference so ctypes doesn't GC the callback
        self._proc_ref = WinEventProc(_callback)

        self._hook_handle = user32.SetWinEventHook(
            EVENT_OBJECT_NAMECHANGE,   # eventMin
            EVENT_OBJECT_NAMECHANGE,   # eventMax
            None,                       # hmodWinEventProc (None = out-of-process)
            self._proc_ref,
            self._target_pid,
            0,                          # idThread (0 = all threads)
            WINEVENT_OUTOFCONTEXT,
        )

        if not self._hook_handle:
            raise OSError("SetWinEventHook failed — ensure D4 has Screen Reader enabled.")

        # Windows message pump — required for WinEvent hooks to fire
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self._hook_handle:
            user32.UnhookWinEvent(self._hook_handle)
            self._hook_handle = None
        ole32.CoUninitialize()
