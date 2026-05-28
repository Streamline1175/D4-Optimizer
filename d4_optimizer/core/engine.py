"""
LLM payload assembly, system prompt orchestration, and API client interface.

Accepts structured game data (equipped items + stash inventory), assembles
the full prompt payload, calls the configured LLM provider, and returns a
structured Markdown upgrade audit report.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from d4_optimizer.config import AppConfig, LLMProvider, get_api_key
from d4_optimizer.core.ingestion import ParsedGuide, guide_to_token_segments


# ---------------------------------------------------------------------------
# Item schema types
# ---------------------------------------------------------------------------


@dataclass
class ItemAffix:
    name: str
    value: float
    is_tempered: bool = False


@dataclass
class Item:
    slot: str
    name: str
    item_power: int
    tier: str                          # "ancestral", "sacred", etc.
    affixes: list[ItemAffix] = field(default_factory=list)
    aspect_name: Optional[str] = None
    aspect_description: Optional[str] = None
    is_unique: bool = False
    unique_power_value: Optional[float] = None
    item_id: Optional[str] = None      # unique stash reference


# ---------------------------------------------------------------------------
# Audit report type
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    raw_markdown: str
    model_used: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    batch_count: int = 1


# ---------------------------------------------------------------------------
# System prompt — production-grade, fully specified
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Diablo 4 build optimizer assistant operating with mathematical precision.
Your role is to analyze a player's equipped items against their stash inventory and produce
a slot-by-slot upgrade audit report in structured Markdown.

## Reasoning Protocol

You MUST follow this exact chain of reasoning for EVERY equipment slot before writing output:

1. **Read the equipped item** for this slot: note each affix name, its rolled value, and
   its theoretical min/max range from the provided database context.
2. **Compute a per-affix efficiency score**: efficiency = (rolled_value - min) / (max - min).
   A score of 1.0 = perfect roll; 0.0 = minimum roll.
3. **Compute the slot's composite quality score**: average of all affix efficiency scores.
4. **Identify missing priority stats**: cross-reference the guide's stat priority list for
   this slot. Flag any priority stat that is absent from the equipped item.
5. **Scan every stash candidate for the same slot**: for each candidate, repeat steps 1–4.
6. **Compare multi-variable combinations**: do NOT compare single affixes in isolation.
   An upgrade is valid only when the stash item's COMBINED weighted stat advantage
   is strictly greater than the equipped item—accounting for all affixes simultaneously.
7. **State an explicit upgrade verdict**: UPGRADE, SIDEGRADE, or NO UPGRADE with a
   one-sentence justification citing specific numerical differences.
8. **Check unique/aspect requirements**: if the guide mandates a specific unique or aspect
   in this slot, flag any mismatch prominently.

## Output Format Rules

- Begin with a `# Upgrade Audit Report` header.
- Use one `## [Slot Name]` section per equipment slot.
- Within each slot section, always include:
  - `### Equipped Item` subsection with affix table and efficiency scores.
  - `### Top Stash Candidates` subsection listing up to 3 best stash alternatives.
  - `### Verdict` with the upgrade decision and justification.
- Use Markdown tables for affix comparisons. Columns: Affix | Rolled | Min | Max | Efficiency%.
- Use **bold** for stat values that exceed the equipped item.
- Flag missing priority stats with ⚠️ prefix.
- Flag guide-mandated uniques/aspects that are absent with 🚨 prefix.
- End the report with a `## Summary` section listing all recommended swaps in priority order.

## Constraints

- Never guess stat values. Use only the values explicitly provided in the stash JSON.
- Never hallucinate affix names. If a name is ambiguous, use the closest match from the
  provided database context and note the uncertainty.
- Do not recommend cosmetic or lore-only improvements.
- Temperature is set low; respond with precise numerical reasoning, not subjective prose.
- If the stash contains no candidates for a slot, state "No stash candidates for this slot."
""").strip()


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


def _serialize_item(item: Item) -> dict[str, Any]:
    """Compact JSON-serializable dict for an Item."""
    d: dict[str, Any] = {
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
    if item.item_id:
        d["item_id"] = item.item_id
    return d


def _build_user_message(
    equipped: list[Item],
    stash_batch: list[Item],
    guide: ParsedGuide,
    db_context: Optional[str],
    batch_index: int,
    total_batches: int,
) -> str:
    """
    Assemble the user-turn message for one batch call.

    Keeps the payload compact:
    - Guide segments use the pre-compressed token format.
    - Items are serialized without redundant keys.
    - DB context is included only for the first batch to avoid duplication.
    """
    parts: list[str] = []

    if batch_index == 0 and db_context:
        parts.append(f"## Database Context (affix/unique ranges)\n```json\n{db_context}\n```")

    guide_segments = guide_to_token_segments(guide)
    if guide_segments:
        guide_lines = []
        for seg in guide_segments:
            guide_lines.append(f"### {seg['section'].replace('_', ' ').title()}\n{seg['content']}")
        parts.append("## Build Guide\n" + "\n\n".join(guide_lines))

    if guide.build_name:
        parts[0] = f"# Build: {guide.build_name}\n\n" + parts[0] if parts else f"# Build: {guide.build_name}"

    equipped_json = json.dumps([_serialize_item(i) for i in equipped], indent=2)
    parts.append(f"## Equipped Items\n```json\n{equipped_json}\n```")

    batch_label = (
        f"(batch {batch_index + 1} of {total_batches})"
        if total_batches > 1
        else ""
    )
    stash_json = json.dumps([_serialize_item(i) for i in stash_batch], indent=2)
    parts.append(f"## Stash Candidates {batch_label}\n```json\n{stash_json}\n```")

    if total_batches > 1 and batch_index < total_batches - 1:
        parts.append(
            "_Note: This is a partial stash batch. Analyze only the candidates provided. "
            "A merged summary will be assembled from all batch results._"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM client wrappers
# ---------------------------------------------------------------------------


def _call_claude(
    system_prompt: str,
    user_message: str,
    config: AppConfig,
) -> tuple[str, Optional[int], Optional[int]]:
    """
    Call the Anthropic Claude API.
    Returns (response_text, prompt_tokens, completion_tokens).
    """
    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for Claude inference. "
            "Install it with: pip install anthropic"
        ) from exc

    api_key = get_api_key(LLMProvider.CLAUDE)
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=config.llm.model,
        max_tokens=config.llm.max_output_tokens,
        temperature=config.llm.temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text: str = message.content[0].text
    prompt_tokens: Optional[int] = getattr(message.usage, "input_tokens", None)
    completion_tokens: Optional[int] = getattr(message.usage, "output_tokens", None)
    return response_text, prompt_tokens, completion_tokens


def _call_gemini(
    system_prompt: str,
    user_message: str,
    config: AppConfig,
) -> tuple[str, Optional[int], Optional[int]]:
    """
    Call the Google Gemini API via the google-generativeai SDK.
    Returns (response_text, prompt_tokens, completion_tokens).
    """
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'google-generativeai' package is required for Gemini inference. "
            "Install it with: pip install google-generativeai"
        ) from exc

    api_key = get_api_key(LLMProvider.GEMINI)
    genai.configure(api_key=api_key)

    generation_config = genai.types.GenerationConfig(  # type: ignore[attr-defined]
        temperature=config.llm.temperature,
        max_output_tokens=config.llm.max_output_tokens,
    )

    model = genai.GenerativeModel(  # type: ignore[attr-defined]
        model_name=config.llm.gemini_model,
        generation_config=generation_config,
        system_instruction=system_prompt,
    )

    response = model.generate_content(user_message)
    response_text: str = response.text

    # Gemini SDK surfaces token counts via usage_metadata
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    return response_text, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------


class InferenceEngine:
    """
    Orchestrates LLM payload construction and inference for upgrade auditing.

    Usage:
        engine = InferenceEngine(config, db_context_json)
        report = engine.run_audit(equipped_items, stash_items, parsed_guide)
    """

    def __init__(
        self,
        config: AppConfig,
        db_context_json: Optional[str] = None,
    ) -> None:
        self._config = config
        self._db_context = db_context_json
        self._provider = config.llm.provider

    def run_audit(
        self,
        equipped: list[Item],
        stash: list[Item],
        guide: ParsedGuide,
    ) -> AuditReport:
        """
        Run the full upgrade audit pipeline.

        Large stash inputs are split into batches. Each batch is sent
        independently; results are merged into a single Markdown report.

        Args:
            equipped: The player's currently equipped item set.
            stash:    All stash items to evaluate as potential upgrades.
            guide:    Parsed build guide with stat priorities and requirements.

        Returns:
            AuditReport with the complete Markdown audit and token usage stats.
        """
        # Pre-filter stash items below the configured item power floor
        viable_stash = [
            item for item in stash
            if item.item_power >= self._config.min_item_power
        ]

        # Split into batches
        batch_size = self._config.stash_batch_size
        batches = [
            viable_stash[i : i + batch_size]
            for i in range(0, max(len(viable_stash), 1), batch_size)
        ]
        if not batches:
            batches = [[]]

        total_batches = len(batches)
        batch_reports: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx, batch in enumerate(batches):
            user_message = _build_user_message(
                equipped=equipped,
                stash_batch=batch,
                guide=guide,
                db_context=self._db_context if idx == 0 else None,
                batch_index=idx,
                total_batches=total_batches,
            )

            response_text, pt, ct = self._call_llm(user_message)
            batch_reports.append(response_text)

            if pt is not None:
                total_prompt_tokens += pt
            if ct is not None:
                total_completion_tokens += ct

        final_markdown = self._merge_batch_reports(batch_reports, total_batches)

        return AuditReport(
            raw_markdown=final_markdown,
            model_used=self._config.llm.model,
            prompt_tokens=total_prompt_tokens or None,
            completion_tokens=total_completion_tokens or None,
            batch_count=total_batches,
        )

    def _call_llm(self, user_message: str) -> tuple[str, Optional[int], Optional[int]]:
        """Dispatch to the configured LLM provider."""
        if self._provider == LLMProvider.CLAUDE:
            return _call_claude(_SYSTEM_PROMPT, user_message, self._config)
        if self._provider == LLMProvider.GEMINI:
            return _call_gemini(_SYSTEM_PROMPT, user_message, self._config)
        raise ValueError(f"Unsupported LLM provider: {self._provider}")

    @staticmethod
    def _merge_batch_reports(reports: list[str], total_batches: int) -> str:
        """
        Merge multiple batch audit reports into one coherent document.

        When only one batch was run, the report is returned as-is.
        For multi-batch runs, duplicate slot sections are collapsed and
        a unified Summary section is synthesized from all batches.
        """
        if len(reports) == 1:
            return reports[0]

        merged_lines: list[str] = ["# Upgrade Audit Report (Merged)\n"]
        seen_slots: set[str] = set()
        summary_lines: list[str] = []

        for report in reports:
            lines = report.split("\n")
            current_slot = ""
            in_summary = False
            slot_lines: list[str] = []

            for line in lines:
                if line.startswith("## ") and not line.startswith("## Summary"):
                    # Flush previous slot
                    if current_slot and current_slot not in seen_slots:
                        merged_lines.extend(slot_lines)
                        seen_slots.add(current_slot)
                    current_slot = line.strip().lstrip("# ")
                    slot_lines = [line]
                    in_summary = False
                elif line.startswith("## Summary"):
                    in_summary = True
                    if current_slot and current_slot not in seen_slots:
                        merged_lines.extend(slot_lines)
                        seen_slots.add(current_slot)
                elif in_summary:
                    summary_lines.append(line)
                else:
                    slot_lines.append(line)

            # Flush last slot of this batch
            if current_slot and current_slot not in seen_slots:
                merged_lines.extend(slot_lines)
                seen_slots.add(current_slot)

        if summary_lines:
            merged_lines.append("\n## Summary (Consolidated)\n")
            merged_lines.extend(summary_lines)

        return "\n".join(merged_lines)
