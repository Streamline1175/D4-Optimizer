"""
Main orchestration entry point for the D4 Optimizer pipeline.

Usage:
    python -m d4_optimizer.main \\
        --guide guide_text.txt \\
        --equipped equipped.json \\
        --stash stash.json \\
        [--output report.md]

Or via the Python API:
    from d4_optimizer.main import run_optimizer
    report = run_optimizer(guide_text, equipped_items, stash_items)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from d4_optimizer.config import load_config
from d4_optimizer.core.database import D4Database
from d4_optimizer.core.engine import AuditReport, InferenceEngine, Item, ItemAffix
from d4_optimizer.core.ingestion import parse_guide, ParsedGuide


# ---------------------------------------------------------------------------
# Item deserialization
# ---------------------------------------------------------------------------


def _deserialize_item(raw: dict[str, Any]) -> Item:
    """Convert a raw dict (from JSON input) into a typed Item dataclass."""
    required = ("slot", "name", "item_power")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Item JSON missing required keys: {missing}. Got: {list(raw.keys())}")

    affixes = [
        ItemAffix(
            name=a["name"],
            value=float(a["value"]),
            is_tempered=bool(a.get("tempered", False)),
        )
        for a in raw.get("affixes", [])
    ]

    return Item(
        slot=raw["slot"],
        name=raw["name"],
        item_power=int(raw["item_power"]),
        tier=raw.get("tier", "ancestral"),
        affixes=affixes,
        aspect_name=raw.get("aspect"),
        aspect_description=raw.get("aspect_description"),
        is_unique=bool(raw.get("is_unique", False)),
        unique_power_value=(
            float(raw["unique_power_value"]) if "unique_power_value" in raw else None
        ),
        item_id=raw.get("item_id"),
    )


def _load_items(path: Path) -> list[Item]:
    """Load and deserialize an item list from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Item file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw_list = json.load(fh)
    if not isinstance(raw_list, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw_list).__name__}")
    return [_deserialize_item(item) for item in raw_list]


# ---------------------------------------------------------------------------
# DB context serializer
# ---------------------------------------------------------------------------


def _build_db_context(db: D4Database, equipped: list[Item]) -> str:
    """
    Build a compact JSON string with affix range data relevant to the
    equipped items. This avoids injecting the entire DB into every prompt.
    """
    context: dict[str, Any] = {}
    seen_affixes: set[str] = set()

    for item in equipped:
        for affix in item.affixes:
            if affix.name in seen_affixes:
                continue
            seen_affixes.add(affix.name)
            try:
                bounds = db.get_affix_bounds(affix.name)
                context[affix.name] = bounds
            except (KeyError, Exception):
                pass  # Affix not in static DB — skip silently

    return json.dumps(context, indent=2)


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------


def run_optimizer(
    guide_text: str,
    equipped_items: list[Item],
    stash_items: list[Item],
    config_override: Optional[Any] = None,
) -> AuditReport:
    """
    Execute the full D4 Optimizer pipeline.

    Args:
        guide_text:     Raw clipboard text from a build guide website.
        equipped_items: The player's currently-worn item set.
        stash_items:    All stash items to consider as upgrades.
        config_override: Optional pre-built AppConfig; defaults to env-based config.

    Returns:
        AuditReport containing the Markdown upgrade audit.
    """
    config = config_override if config_override is not None else load_config()

    # 1. Load static game database
    db = D4Database()

    # 2. Parse guide text
    guide: ParsedGuide = parse_guide(guide_text)
    if config.debug:
        print(
            f"[DEBUG] Parsed guide: build='{guide.build_name}', "
            f"class='{guide.class_name}', "
            f"stat_priorities={len(guide.stat_priorities)}, "
            f"aspects={len(guide.required_aspects)}, "
            f"uniques={len(guide.unique_breakdowns)}",
            file=sys.stderr,
        )

    # 3. Build DB context payload (affix ranges for equipped items)
    db_context_json = _build_db_context(db, equipped_items)

    # 4. Run LLM inference
    engine = InferenceEngine(config=config, db_context_json=db_context_json)
    report = engine.run_audit(
        equipped=equipped_items,
        stash=stash_items,
        guide=guide,
    )

    if config.debug:
        print(
            f"[DEBUG] Audit complete: "
            f"model={report.model_used}, "
            f"batches={report.batch_count}, "
            f"prompt_tokens={report.prompt_tokens}, "
            f"completion_tokens={report.completion_tokens}",
            file=sys.stderr,
        )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="d4-optimizer",
        description="Diablo 4 build optimizer — LLM-powered upgrade audit.",
    )
    parser.add_argument(
        "--guide",
        type=Path,
        required=True,
        help="Path to a .txt file containing the raw guide paste.",
    )
    parser.add_argument(
        "--equipped",
        type=Path,
        required=True,
        help="Path to a JSON file containing the equipped item array.",
    )
    parser.add_argument(
        "--stash",
        type=Path,
        required=True,
        help="Path to a JSON file containing the stash inventory array.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the Markdown report (default: stdout).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    guide_path: Path = args.guide
    if not guide_path.exists():
        parser.error(f"Guide file not found: {guide_path}")

    guide_text = guide_path.read_text(encoding="utf-8")
    equipped = _load_items(args.equipped)
    stash = _load_items(args.stash)

    report = run_optimizer(
        guide_text=guide_text,
        equipped_items=equipped,
        stash_items=stash,
    )

    if args.output:
        args.output.write_text(report.raw_markdown, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report.raw_markdown)


if __name__ == "__main__":
    main()
