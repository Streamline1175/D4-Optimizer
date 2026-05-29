"""
Item import helper for the D4 Optimizer.

Run this script to convert in-game item tooltips into equipped.json or stash.json.

Usage:
    # Interactive session (paste tooltips one-by-one):
    python import_items.py --output equipped.json

    # Batch file (all tooltips in a text file, separated by blank lines):
    python import_items.py --file my_stash_tooltips.txt --output stash.json

    # Append items to an existing file:
    python import_items.py --file more_items.txt --output stash.json --append

How to get your tooltip text:
    1. Open Diablo 4 and hover over an item.
    2. The tooltip shows:
         Ancestral Rare Helm of X
         Item Power 925
         +7.5% Critical Strike Chance
         +1,100 Maximum Life
         ...
    3. Type or copy those lines and paste them here.
    4. If using D4 Companion, enable "Copy Tooltip" in its settings to
       copy the OCR-extracted text to your clipboard automatically.

Separating multiple items in a batch file:
    Leave a blank line (or put "---") between each item tooltip.

Example batch file (stash_items.txt):
    Ancestral Rare Helm
    Item Power 921
    +6.8% Critical Strike Chance
    +980 Maximum Life
    +1380 Armor
    +8.5% Cooldown Reduction

    Ancestral Rare Ring
    Item Power 915
    +7.1% Critical Strike Chance
    +29.0% Critical Strike Damage
    +15.5% Core Skill Damage
    +8.8% Lucky Hit Chance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="import-items",
        description="Convert D4 item tooltips to optimizer JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSON file path (e.g. equipped.json or stash.json).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Batch input: path to a text file containing tooltip blocks separated by blank lines.",
    )
    parser.add_argument(
        "--slot",
        "-s",
        default=None,
        help="Override slot for all items in the batch (useful when the tooltip doesn't name the slot).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append new items to an existing output file instead of overwriting.",
    )
    args = parser.parse_args()

    # Deferred import so errors in the module don't mask the argparse help
    from d4_optimizer.core.tooltip_parser import (
        interactive_import,
        parse_tooltip_batch,
        _item_to_dict,
        _save_items,
    )

    if args.file:
        # Batch mode
        if not args.file.exists():
            print(f"Error: input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        raw_text = args.file.read_text(encoding="utf-8")
        items = parse_tooltip_batch(raw_text, slot_hint=args.slot)

        if not items:
            print("No items parsed. Check that tooltips include 'Item Power NNN'.")
            sys.exit(1)

        # Build output list
        new_entries = [_item_to_dict(i) for i in items]

        existing: list = []
        if args.append and args.output.exists():
            with args.output.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)

        combined = existing + new_entries
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2)

        print(f"Parsed {len(items)} item(s) → saved to {args.output}")
        for item in items:
            print(f"  [{item.slot:10s}] IP:{item.item_power}  {item.name}  ({len(item.affixes)} affixes)")

    else:
        # Interactive mode
        if not args.append and args.output.exists():
            answer = input(
                f"{args.output} already exists. Overwrite? [y/N] "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                sys.exit(0)
            args.output.unlink()

        interactive_import(output_path=args.output)


if __name__ == "__main__":
    main()
