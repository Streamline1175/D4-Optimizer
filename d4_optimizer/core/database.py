"""
Lothrik JSON DB loading, indexing, and min/max range resolution.

Loads the static data-mined JSON structures, builds searchable hash maps,
and exposes a clean resolution API for affix/unique rolling ranges.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from d4_optimizer.config import DATA_DIR, ItemTier


# ---------------------------------------------------------------------------
# Type aliases mirroring the Lothrik schema shapes
# ---------------------------------------------------------------------------

AffixEntry = dict[str, Any]
UniqueEntry = dict[str, Any]
RangeResult = dict[str, float]  # {"min": float, "max": float}


class AffixNotFoundError(KeyError):
    """Raised when an affix cannot be located in the loaded database."""


class UniqueNotFoundError(KeyError):
    """Raised when a unique item cannot be located in the loaded database."""


# ---------------------------------------------------------------------------
# Internal normalizer helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"[#\[\]{}()\d.%+]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_affix_key(raw: str) -> str:
    """
    Produce a lowercase, whitespace-collapsed, placeholder-stripped key
    suitable for fuzzy hash-map lookups.
    """
    stripped = _PLACEHOLDER_RE.sub("", raw)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip().lower()
    return collapsed


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class D4Database:
    """
    In-memory indexed representation of the Lothrik game data JSONs.

    Index structures built on load:
      _affix_by_id      : affix_id -> AffixEntry
      _affix_by_name    : normalized_name -> AffixEntry
      _unique_by_id     : unique_id -> UniqueEntry
      _unique_by_name   : normalized_name -> UniqueEntry
    """

    def __init__(
        self,
        affixes_path: Path = DATA_DIR / "affixes.json",
        uniques_path: Path = DATA_DIR / "uniques.json",
    ) -> None:
        self._affixes_path = affixes_path
        self._uniques_path = uniques_path

        self._affix_by_id: dict[str, AffixEntry] = {}
        self._affix_by_name: dict[str, AffixEntry] = {}
        self._unique_by_id: dict[str, UniqueEntry] = {}
        self._unique_by_name: dict[str, UniqueEntry] = {}

        self._load()

    # ------------------------------------------------------------------
    # Loading & indexing
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._load_affixes()
        self._load_uniques()

    def _load_affixes(self) -> None:
        raw = self._read_json(self._affixes_path)
        for entry in raw.get("affixes", []):
            affix_id: str = entry["id"]
            self._affix_by_id[affix_id] = entry

            # Index by canonical name
            name_key = _normalize_affix_key(entry["name"])
            self._affix_by_name[name_key] = entry

            # Also index by description template (stripped of placeholders)
            desc_key = _normalize_affix_key(entry.get("description", ""))
            if desc_key and desc_key not in self._affix_by_name:
                self._affix_by_name[desc_key] = entry

    def _load_uniques(self) -> None:
        raw = self._read_json(self._uniques_path)
        for entry in raw.get("uniques", []):
            uid: str = entry["id"]
            self._unique_by_id[uid] = entry

            name_key = _normalize_affix_key(entry["name"])
            self._unique_by_name[name_key] = entry

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Database file not found: {path}. "
                "Ensure the data/ directory contains affixes.json and uniques.json."
            )
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Public resolution API
    # ------------------------------------------------------------------

    def get_affix_bounds(
        self,
        affix_name: str,
        tier: ItemTier = ItemTier.ANCESTRAL,
    ) -> RangeResult:
        """
        Return the absolute rolling range {min, max} for a named affix at the
        given item tier.

        Args:
            affix_name: Human-readable affix name or description fragment
                        (e.g. "Critical Strike Damage", "+#% Critical Strike Damage").
            tier:       Item quality tier used to select the correct range band.

        Returns:
            {"min": float, "max": float}

        Raises:
            AffixNotFoundError: When no matching affix is found in the DB.
            KeyError:           When the affix exists but has no data for `tier`.
        """
        entry = self._resolve_affix(affix_name)
        tier_key = tier.value if tier not in (ItemTier.NORMAL, ItemTier.MAGIC, ItemTier.RARE) else "sacred"
        ranges: dict[str, Any] = entry.get("ranges", {})

        if tier_key not in ranges:
            # Fall back to the closest available tier
            fallback_order = ["ancestral", "sacred"]
            for fb in fallback_order:
                if fb in ranges:
                    tier_key = fb
                    break
            else:
                raise KeyError(
                    f"Affix '{affix_name}' has no range data for tier '{tier.value}' "
                    f"or any fallback tier. Available: {list(ranges.keys())}"
                )

        band = ranges[tier_key]
        return {"min": float(band["min"]), "max": float(band["max"])}

    def get_unique_affix_bounds(
        self,
        unique_name: str,
        affix_name: str,
        tier: ItemTier = ItemTier.ANCESTRAL,
    ) -> RangeResult:
        """
        Return rolling range for a specific affix on a named unique item.

        Unique items carry per-affix overrides that can exceed normal ranges
        (e.g. The Grandfather).
        """
        unique_entry = self._resolve_unique(unique_name)
        norm_affix = _normalize_affix_key(affix_name)

        for affix in unique_entry.get("affixes", []):
            if _normalize_affix_key(affix["name"]) == norm_affix:
                tier_key = tier.value if tier not in (ItemTier.NORMAL, ItemTier.MAGIC, ItemTier.RARE) else "sacred"
                band = affix.get("ranges", {}).get(tier_key)
                if band:
                    return {"min": float(band["min"]), "max": float(band["max"])}
                # Try ancestral fallback
                band = affix.get("ranges", {}).get("ancestral")
                if band:
                    return {"min": float(band["min"]), "max": float(band["max"])}

        # Affix not found on this unique — fall back to global pool
        return self.get_affix_bounds(affix_name, tier)

    def get_unique_power_range(self, unique_name: str) -> dict[str, RangeResult]:
        """
        Return all named variable ranges for a unique item's special power.

        Returns a dict mapping parameter name -> {min, max}.
        """
        entry = self._resolve_unique(unique_name)
        power = entry.get("unique_power", {})
        return {
            key: {"min": float(val["min"]), "max": float(val["max"])}
            for key, val in power.get("ranges", {}).items()
        }

    def get_affix_slots(self, affix_name: str) -> list[str]:
        """Return the list of equipment slots the named affix can roll on."""
        return list(self._resolve_affix(affix_name).get("slots", []))

    def list_affixes_for_slot(self, slot: str) -> list[str]:
        """Return all affix names eligible for a given equipment slot."""
        slot_lower = slot.lower()
        return [
            entry["name"]
            for entry in self._affix_by_id.values()
            if slot_lower in [s.lower() for s in entry.get("slots", [])]
        ]

    def lookup_unique(self, unique_name: str) -> UniqueEntry:
        """Return the full unique entry dict for a given unique name."""
        return dict(self._resolve_unique(unique_name))

    # ------------------------------------------------------------------
    # Internal resolution helpers
    # ------------------------------------------------------------------

    def _resolve_affix(self, affix_name: str) -> AffixEntry:
        norm = _normalize_affix_key(affix_name)
        if norm in self._affix_by_name:
            return self._affix_by_name[norm]

        # Partial substring match as fallback
        candidates = [
            (key, entry)
            for key, entry in self._affix_by_name.items()
            if norm in key or key in norm
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            # Prefer exact name match over description match
            for key, entry in candidates:
                if _normalize_affix_key(entry["name"]) == norm:
                    return entry
            return candidates[0][1]

        raise AffixNotFoundError(
            f"Affix '{affix_name}' (normalized: '{norm}') not found in database. "
            f"Available affixes: {[e['name'] for e in self._affix_by_id.values()]}"
        )

    def _resolve_unique(self, unique_name: str) -> UniqueEntry:
        norm = _normalize_affix_key(unique_name)
        if norm in self._unique_by_name:
            return self._unique_by_name[norm]

        candidates = [
            (key, entry)
            for key, entry in self._unique_by_name.items()
            if norm in key or key in norm
        ]
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            for key, entry in candidates:
                if _normalize_affix_key(entry["name"]) == norm:
                    return entry
            return candidates[0][1]

        raise UniqueNotFoundError(
            f"Unique '{unique_name}' (normalized: '{norm}') not found in database. "
            f"Available uniques: {[e['name'] for e in self._unique_by_id.values()]}"
        )
