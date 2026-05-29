"""
Automated item capture for the D4 Optimizer.

Two fully automatic modes — no manual copy-paste required:

────────────────────────────────────────────────────────────────
MODE 1: clipboard  (recommended — works on any OS)
────────────────────────────────────────────────────────────────
Requires D4 Companion with clipboard copy enabled:
  D4 Companion → Settings → General → "Copy item to clipboard" ✓

Then run:
    python capture_items.py --mode clipboard --output stash.json

Hover items in Diablo 4. D4 Companion OCRs each tooltip and copies
the text to your clipboard automatically. This script detects the
change and saves each item to JSON silently.

────────────────────────────────────────────────────────────────
MODE 2: accessibility  (Windows only — no extra tools needed)
────────────────────────────────────────────────────────────────
Requires Screen Reader enabled in Diablo 4:
  D4 → Settings → Accessibility → Screen Reader → On

Then run:
    python capture_items.py --mode accessibility --output stash.json
    pip install pywin32 comtypes  (first time only)

D4 fires tooltip text through the Windows accessibility channel
(EVENT_OBJECT_NAMECHANGE). This script hooks that event out-of-process
(no DLL injection, no game memory access) and parses items automatically
as you hover them.

────────────────────────────────────────────────────────────────
Workflow tips:
  - Run once with --output equipped.json while hovering your 13 equipped slots.
  - Run again with --output stash.json --append while scrolling your stash.
  - Items already saved are preserved across sessions with --append.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from d4_optimizer.core.engine import Item
from d4_optimizer.core.tooltip_parser import parse_tooltip, _item_to_dict


def _make_file_writer(output_path: Path, append: bool):
    """Return a callback that writes each captured item to a JSON file."""
    items: list[dict] = []
    count = 0

    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            items = json.load(fh)
        print(f"Loaded {len(items)} existing item(s) from {output_path}")

    def _on_item(item: Item) -> None:
        nonlocal count
        entry = _item_to_dict(item)
        items.append(entry)
        count += 1
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2)
        print(
            f"  [{count:>4}] {item.name:<40} [{item.slot:<10}]  "
            f"IP:{item.item_power}  affixes:{len(item.affixes)}"
        )

    return _on_item


def _run_clipboard_mode(output_path: Path, append: bool, interval_ms: int) -> None:
    from d4_optimizer.capture.clipboard_monitor import monitor_to_file

    if not append and output_path.exists():
        ans = input(f"{output_path} exists. Overwrite? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
        output_path.unlink()

    print("═" * 60)
    print("  CLIPBOARD MODE")
    print("  D4 Companion must have 'Copy item to clipboard' enabled.")
    print("  Hover items in Diablo 4 — they are captured automatically.")
    print("═" * 60)
    monitor_to_file(output_path=output_path, append=append, interval_ms=interval_ms)


def _run_accessibility_mode(output_path: Path, append: bool) -> None:
    if sys.platform != "win32":
        print("Error: accessibility mode is Windows-only.", file=sys.stderr)
        print("Use --mode clipboard instead.", file=sys.stderr)
        sys.exit(1)

    from d4_optimizer.capture.win_accessibility import AccessibilityHook

    on_item = _make_file_writer(output_path, append)
    captured_count = 0

    def _on_tooltip(raw_text: str) -> None:
        nonlocal captured_count
        try:
            item = parse_tooltip(raw_text)
        except ValueError as exc:
            print(f"  [skip] Parse error: {exc}", file=sys.stderr)
            return
        if item.slot == "unknown":
            print(f"  [skip] Slot not detected for '{item.name}'", file=sys.stderr)
            return
        on_item(item)
        captured_count += 1

    print("═" * 60)
    print("  ACCESSIBILITY MODE")
    print("  Diablo 4 must have Screen Reader enabled in Accessibility settings.")
    print("  Hover items in D4 — they are captured via the Win32 event hook.")
    print("═" * 60)

    hook = AccessibilityHook(on_tooltip_text=_on_tooltip)
    try:
        hook.start()
    finally:
        print(f"\nSession ended. {captured_count} item(s) saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="capture-items",
        description="Automatically capture D4 item data to JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["clipboard", "accessibility"],
        default="clipboard",
        help="Capture mode (default: clipboard).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSON file (e.g. equipped.json or stash.json).",
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        default=False,
        help="Append to existing file instead of overwriting.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=150,
        metavar="MS",
        help="Clipboard polling interval in milliseconds (default: 150, clipboard mode only).",
    )
    args = parser.parse_args()

    if args.mode == "clipboard":
        _run_clipboard_mode(args.output, args.append, args.interval)
    else:
        _run_accessibility_mode(args.output, args.append)


if __name__ == "__main__":
    main()
