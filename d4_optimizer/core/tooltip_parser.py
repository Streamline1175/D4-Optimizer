"""
Diablo 4 item tooltip text parser.

Converts raw copy-pasted item tooltip text (or OCR output from D4 Companion /
screenshots) into the Item schema used by the optimizer.

Supported tooltip sources:
  - Manual copy-paste from in-game item hover
  - D4 Companion OCR output (same visual format)
  - Any plain-text representation of the standard D4 tooltip layout

Tooltip format reference (what D4 renders):

    Ancestral Rare Helm of Skullbreaker
    Item Power 925

    +7.5% Critical Strike Chance
    +1,100 Maximum Life
    +1,450 Armor
    +9.2% Cooldown Reduction
    ──────────────────────
    Imprinted Aspect: Aspect of Might
    Damage Reduction for 2 seconds after using a Basic Skill is increased by [42.5]%.

Unique example:
    Ancestral Unique Helmet
    The Grandfather
    Item Power 925

    +82.0% Critical Strike Damage
    +1,200 Maximum Life
    +178 Strength
    +68.0% Overpower Damage
    ──────────────────────
    Unique Power: Increases your Critical Strike Damage by +[87.5]%.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from d4_optimizer.core.engine import Item, ItemAffix


# ---------------------------------------------------------------------------
# Regex patterns for tooltip line recognition
# ---------------------------------------------------------------------------

# Tier keyword detection
_TIER_RE = re.compile(r"\b(ancestral|sacred)\b", re.IGNORECASE)
_UNIQUE_KEYWORD_RE = re.compile(r"\bunique\b", re.IGNORECASE)

# Item Power line: "Item Power 925" or "Item Power: 925"
_ITEM_POWER_RE = re.compile(r"item\s+power\s*:?\s*(\d+)", re.IGNORECASE)

# Affix value line: "+7.5% Critical Strike Chance" or "+1,100 Maximum Life"
# Also handles tempered affixes which share the same visual format
_AFFIX_LINE_RE = re.compile(
    r"^\s*\+\s*([\d,]+(?:\.\d+)?)\s*(%?)\s+(.+?)\s*$"
)

# Tempered-affix marker: D4 Companion and some OCR tools prepend [T] or mark in brackets
_TEMPERED_MARKER_RE = re.compile(r"\[T\]|\btempered\b", re.IGNORECASE)

# Aspect / imprint lines
_ASPECT_LINE_RE = re.compile(
    r"(?:imprinted?\s+aspect|aspect\s+imprinted?|imprinted?)\s*:?\s*(.+)",
    re.IGNORECASE,
)
_ASPECT_DESCRIPTION_RE = re.compile(
    r"\[[\d.]+\]",  # values inside brackets → aspect description line
)

# Unique power line
_UNIQUE_POWER_RE = re.compile(
    r"unique\s+power\s*:?\s*(.*)",
    re.IGNORECASE,
)
_UNIQUE_VALUE_RE = re.compile(r"\[([\d.]+)\]")

# Divider lines (horizontal rules in the tooltip)
_DIVIDER_RE = re.compile(r"^[\s\-─━═_]{4,}$")

# Slot keywords that may appear in the item type header
_SLOT_KEYWORD_MAP: dict[str, str] = {
    "helm": "helm",
    "helmet": "helm",
    "chest": "chest",
    "chest armor": "chest",
    "gloves": "gloves",
    "pants": "pants",
    "legs": "pants",
    "boots": "boots",
    "amulet": "amulet",
    "necklace": "amulet",
    "ring": "ring",
    "sword": "weapon",
    "axe": "weapon",
    "mace": "weapon",
    "staff": "weapon",
    "polearm": "weapon",
    "scythe": "weapon",
    "dagger": "weapon",
    "crossbow": "weapon",
    "bow": "weapon",
    "wand": "weapon",
    "focus": "offhand",
    "shield": "offhand",
    "totem": "offhand",
    "offhand": "offhand",
    "off-hand": "offhand",
    "two-handed": "weapon",
    "two handed": "weapon",
}

_SLOT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_SLOT_KEYWORD_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------


def parse_tooltip(raw_tooltip: str, slot_hint: Optional[str] = None) -> Item:
    """
    Parse a single item tooltip text block into an Item dataclass.

    Args:
        raw_tooltip:  Raw text of one item tooltip.
        slot_hint:    If provided, overrides slot detection (e.g. "ring").
                      Required when the slot cannot be inferred from the tooltip
                      (e.g. a rare ring has no "Ring" keyword in the item name).

    Returns:
        A populated Item ready for use in the optimizer.

    Raises:
        ValueError: If item power cannot be found in the tooltip text.
    """
    lines = [l.rstrip() for l in raw_tooltip.strip().splitlines()]
    lines = [l for l in lines if l.strip()]  # drop blank lines

    item_power: Optional[int] = None
    tier = "ancestral"
    is_unique = False
    item_name = ""
    affixes: list[ItemAffix] = []
    aspect_name: Optional[str] = None
    aspect_description: Optional[str] = None
    unique_power_value: Optional[float] = None
    inferred_slot: Optional[str] = None

    next_line_is_unique_name = False
    next_line_is_aspect_desc = False
    in_unique_power = False

    for line in lines:
        stripped = line.strip()

        # ── Item Power ──────────────────────────────────────────────────
        ip_match = _ITEM_POWER_RE.search(stripped)
        if ip_match:
            item_power = int(ip_match.group(1).replace(",", ""))
            continue

        # ── Divider lines ────────────────────────────────────────────────
        if _DIVIDER_RE.match(stripped):
            continue

        # ── Tier / unique keyword detection (header lines) ───────────────
        tier_match = _TIER_RE.search(stripped)
        if tier_match:
            tier = tier_match.group(1).lower()

        if _UNIQUE_KEYWORD_RE.search(stripped):
            is_unique = True
            # The line after the type header for a unique is the unique's name
            next_line_is_unique_name = True

        # ── Slot detection from header line ──────────────────────────────
        slot_match = _SLOT_RE.search(stripped)
        if slot_match and not affixes:
            inferred_slot = _SLOT_KEYWORD_MAP.get(slot_match.group(1).lower(), slot_match.group(1).lower())

        # ── Unique item name (line after unique type header) ─────────────
        if next_line_is_unique_name and not _AFFIX_LINE_RE.match(stripped) and not _ITEM_POWER_RE.search(stripped):
            if not _TIER_RE.search(stripped) and not _UNIQUE_KEYWORD_RE.search(stripped):
                item_name = stripped
                next_line_is_unique_name = False
                continue

        # ── Aspect / imprint lines ────────────────────────────────────────
        aspect_match = _ASPECT_LINE_RE.match(stripped)
        if aspect_match:
            aspect_name = aspect_match.group(1).strip().rstrip(":").strip()
            next_line_is_aspect_desc = True
            in_unique_power = False
            continue

        if next_line_is_aspect_desc:
            aspect_description = stripped
            next_line_is_aspect_desc = False
            continue

        # ── Unique power lines ────────────────────────────────────────────
        unique_power_match = _UNIQUE_POWER_RE.match(stripped)
        if unique_power_match:
            in_unique_power = True
            power_text = unique_power_match.group(1).strip()
            val_match = _UNIQUE_VALUE_RE.search(power_text)
            if val_match:
                unique_power_value = float(val_match.group(1))
            continue

        if in_unique_power:
            val_match = _UNIQUE_VALUE_RE.search(stripped)
            if val_match and unique_power_value is None:
                unique_power_value = float(val_match.group(1))
            continue

        # ── Affix lines ───────────────────────────────────────────────────
        affix_match = _AFFIX_LINE_RE.match(stripped)
        if affix_match:
            raw_value_str = affix_match.group(1).replace(",", "")
            pct_symbol = affix_match.group(2)  # "%" or ""
            affix_name_raw = affix_match.group(3).strip()
            value = float(raw_value_str)

            is_tempered = bool(_TEMPERED_MARKER_RE.search(affix_name_raw))
            clean_name = _TEMPERED_MARKER_RE.sub("", affix_name_raw).strip()

            affixes.append(
                ItemAffix(
                    name=clean_name,
                    value=value,
                    is_tempered=is_tempered,
                )
            )
            continue

        # ── Fallback: first non-header, non-power line = item name ────────
        if not item_name and not is_unique and not affixes:
            # Avoid capturing tier keywords as the name
            if not _TIER_RE.fullmatch(stripped) and not _UNIQUE_KEYWORD_RE.fullmatch(stripped):
                if len(stripped) > 2:
                    item_name = stripped

    if item_power is None:
        raise ValueError(
            "Could not find 'Item Power' in tooltip text. "
            "Ensure the tooltip includes a line like 'Item Power 925'."
        )

    final_slot = slot_hint or inferred_slot or "unknown"
    if not item_name:
        item_name = f"{tier.title()} Item ({final_slot.title()})"

    return Item(
        slot=final_slot,
        name=item_name,
        item_power=item_power,
        tier=tier,
        affixes=affixes,
        aspect_name=aspect_name,
        aspect_description=aspect_description,
        is_unique=is_unique,
        unique_power_value=unique_power_value,
    )


def parse_tooltip_batch(raw_text: str, slot_hint: Optional[str] = None) -> list[Item]:
    """
    Parse multiple item tooltips from a single text block.

    Tooltips must be separated by a blank line or a line containing only
    dashes/equals (e.g. "---" or "==="). Each item's slot will be inferred
    from the tooltip or fall back to slot_hint.

    Args:
        raw_text:   Text containing one or more item tooltips.
        slot_hint:  Default slot used when the tooltip doesn't name one.

    Returns:
        List of parsed Item objects (failed items are skipped with a warning).
    """
    # Split on blank lines or explicit separator lines
    blocks = re.split(r"\n{2,}|\n[=\-]{3,}\n", raw_text.strip())
    items: list[Item] = []

    for idx, block in enumerate(blocks, start=1):
        block = block.strip()
        if not block:
            continue
        try:
            item = parse_tooltip(block, slot_hint=slot_hint)
            items.append(item)
        except ValueError as exc:
            print(
                f"[Warning] Skipping item block {idx} — parse error: {exc}",
                file=sys.stderr,
            )

    return items


# ---------------------------------------------------------------------------
# Interactive CLI helper
# ---------------------------------------------------------------------------


def interactive_import(output_path: Optional[Path] = None) -> list[Item]:
    """
    Interactive item entry session. Prompts the user to paste item tooltips
    one at a time, asking for slot if it cannot be inferred.

    Saves results to output_path as JSON (appending to existing file if present).
    Returns the collected Item list.
    """
    import json

    print("=== D4 Optimizer — Item Import Session ===")
    print("Paste each item tooltip, then type END on a new line.")
    print("Type DONE to finish and save.\n")

    items: list[Item] = []

    while True:
        print("Paste item tooltip (or type DONE to finish):")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().upper() == "DONE":
                if output_path:
                    _save_items(items, output_path)
                    print(f"\nSaved {len(items)} item(s) to {output_path}")
                return items
            if line.strip().upper() == "END":
                break
            lines.append(line)

        raw = "\n".join(lines).strip()
        if not raw:
            continue

        try:
            item = parse_tooltip(raw)
        except ValueError as exc:
            print(f"  ✗ Parse error: {exc}")
            continue

        if item.slot == "unknown":
            valid_slots = list(dict.fromkeys(_SLOT_KEYWORD_MAP.values()))
            print(f"  Slot not detected. Valid slots: {valid_slots}")
            slot_input = input("  Enter slot: ").strip().lower()
            item.slot = slot_input or "unknown"

        items.append(item)
        print(f"  ✓ Parsed: {item.name} [{item.slot}] IP:{item.item_power} — {len(item.affixes)} affixes\n")

    if output_path:
        _save_items(items, output_path)
        print(f"\nSaved {len(items)} item(s) to {output_path}")

    return items


def _save_items(items: list[Item], path: Path) -> None:
    import json

    existing: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)

    new_entries = [_item_to_dict(i) for i in items]
    combined = existing + new_entries

    with path.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)


def _item_to_dict(item: Item) -> dict:
    d: dict = {
        "slot": item.slot,
        "name": item.name,
        "item_power": item.item_power,
        "tier": item.tier,
        "affixes": [
            {
                "name": a.name,
                "value": a.value,
                **({"tempered": True} if a.is_tempered else {}),
            }
            for a in item.affixes
        ],
    }
    if item.is_unique:
        d["is_unique"] = True
    if item.unique_power_value is not None:
        d["unique_power_value"] = item.unique_power_value
    if item.aspect_name:
        d["aspect"] = item.aspect_name
    if item.aspect_description:
        d["aspect_description"] = item.aspect_description
    return d
