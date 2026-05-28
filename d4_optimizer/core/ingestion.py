"""
Guide text parsing, sanitization, and regex-filtering.

Accepts raw clipboard-paste text from Maxroll or similar build guide sites
and extracts structured segments: stat priorities, required aspects, and
unique/runeword breakdowns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class StatPriority:
    slot: str
    stats: list[str]
    raw_text: str


@dataclass
class RequiredAspect:
    name: str
    description: str
    slot_recommendation: str
    raw_text: str


@dataclass
class UniqueBreakdown:
    name: str
    slot: str
    why_bis: str
    alternatives: list[str]
    raw_text: str


@dataclass
class RunewordBreakdown:
    name: str
    slot: str
    effect_summary: str
    raw_text: str


@dataclass
class ParsedGuide:
    stat_priorities: list[StatPriority] = field(default_factory=list)
    required_aspects: list[RequiredAspect] = field(default_factory=list)
    unique_breakdowns: list[UniqueBreakdown] = field(default_factory=list)
    runeword_breakdowns: list[RunewordBreakdown] = field(default_factory=list)
    build_name: Optional[str] = None
    class_name: Optional[str] = None
    raw_cleaned: str = ""


# ---------------------------------------------------------------------------
# Noise-stripping patterns
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_SCRIPT_BLOCK_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_URL_TRACKING_RE = re.compile(
    r"https?://\S+utm_[a-z_]+=\S*", re.IGNORECASE
)
_BARE_URL_RE = re.compile(r"https?://\S+")
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_UNICODE_BULLET_RE = re.compile(r"[•·‣⁃◦▸▹▷▶►]")

# Common web-UI noise patterns from Maxroll / Mobalytics
_MAXROLL_NOISE_RE = re.compile(
    r"(Copy\s+build\s+code|Share\s+build|Last\s+Updated|Season\s+\d+|"
    r"Patch\s+[\d.]+|Table\s+of\s+Contents|Back\s+to\s+Top|"
    r"Subscribe\s+to\s+Newsletter|cookie\s+policy|privacy\s+policy)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Section-detection patterns
# ---------------------------------------------------------------------------

_STAT_PRIORITY_HEADER_RE = re.compile(
    r"(?:stat\s+priority|gear\s+stat|affixes?\s+priority|item\s+stat)",
    re.IGNORECASE,
)
_ASPECT_HEADER_RE = re.compile(
    r"(?:aspect|codex\s+of\s+power|legendary\s+power|imprint)",
    re.IGNORECASE,
)
_UNIQUE_HEADER_RE = re.compile(
    r"(?:unique\s+item|bis\s+unique|recommended\s+unique|best\s+in\s+slot\s+unique)",
    re.IGNORECASE,
)
_RUNEWORD_HEADER_RE = re.compile(
    r"(?:runeword|rune\s+word|rune\s+combination)",
    re.IGNORECASE,
)

_SLOT_NAMES = [
    "helm", "chest", "gloves", "pants", "boots",
    "amulet", "ring 1", "ring 2", "ring", "main hand",
    "weapon", "offhand", "off-hand", "focus",
]
_SLOT_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SLOT_NAMES) + r")\b",
    re.IGNORECASE,
)

_STAT_LINE_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*|[-–•]\s*)?(.+)$",
    re.MULTILINE,
)

_D4_CLASS_RE = re.compile(
    r"\b(barbarian|necromancer|sorcerer|sorceress|druid|rogue|spiritborn)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_guide(raw_text: str) -> ParsedGuide:
    """
    Full pipeline: clean the raw guide paste and extract all structured segments.

    Args:
        raw_text: Raw text pasted from a build guide (Maxroll, Mobalytics, etc.).

    Returns:
        A populated ParsedGuide dataclass.
    """
    cleaned = _clean_html(raw_text)
    cleaned = _strip_noise(cleaned)

    guide = ParsedGuide(raw_cleaned=cleaned)

    # Detect class and build name from leading text
    guide.class_name = _detect_class(cleaned)
    guide.build_name = _detect_build_name(cleaned)

    # Segment the text into topic blocks, then parse each
    sections = _split_into_sections(cleaned)
    guide.stat_priorities = _parse_stat_priorities(sections.get("stat_priority", []))
    guide.required_aspects = _parse_aspects(sections.get("aspects", []))
    guide.unique_breakdowns = _parse_uniques(sections.get("uniques", []))
    guide.runeword_breakdowns = _parse_runewords(sections.get("runewords", []))

    return guide


def clean_text(raw_text: str) -> str:
    """Return the sanitized text without further parsing — useful for debug."""
    return _strip_noise(_clean_html(raw_text))


def guide_to_token_segments(guide: ParsedGuide) -> list[dict[str, str]]:
    """
    Convert a ParsedGuide into a list of compact token-efficient text segments
    ready for LLM context assembly.

    Each segment has:
      "section": section label
      "content": tightly formatted text
    """
    segments: list[dict[str, str]] = []

    if guide.stat_priorities:
        lines = []
        for sp in guide.stat_priorities:
            stats_str = " > ".join(sp.stats)
            lines.append(f"{sp.slot}: {stats_str}")
        segments.append({"section": "stat_priorities", "content": "\n".join(lines)})

    if guide.required_aspects:
        lines = []
        for asp in guide.required_aspects:
            lines.append(f"{asp.name} [{asp.slot_recommendation}]: {asp.description}")
        segments.append({"section": "aspects", "content": "\n".join(lines)})

    if guide.unique_breakdowns:
        lines = []
        for u in guide.unique_breakdowns:
            alt_str = (", ".join(u.alternatives)) if u.alternatives else "none"
            lines.append(f"{u.name} [{u.slot}]: {u.why_bis} | alts: {alt_str}")
        segments.append({"section": "uniques", "content": "\n".join(lines)})

    if guide.runeword_breakdowns:
        lines = []
        for rw in guide.runeword_breakdowns:
            lines.append(f"{rw.name} [{rw.slot}]: {rw.effect_summary}")
        segments.append({"section": "runewords", "content": "\n".join(lines)})

    return segments


# ---------------------------------------------------------------------------
# Internal cleaning helpers
# ---------------------------------------------------------------------------


def _clean_html(text: str) -> str:
    text = _SCRIPT_BLOCK_RE.sub("", text)
    text = _STYLE_BLOCK_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(_decode_html_entity, text)
    return text


def _decode_html_entity(match: re.Match) -> str:
    entity = match.group(0)
    mapping = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&apos;": "'", "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
        "&bull;": "•", "&trade;": "™", "&reg;": "®",
    }
    return mapping.get(entity.lower(), " ")


def _strip_noise(text: str) -> str:
    text = _URL_TRACKING_RE.sub("", text)
    text = _BARE_URL_RE.sub("", text)
    text = _MAXROLL_NOISE_RE.sub("", text)
    text = _UNICODE_BULLET_RE.sub("-", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Internal section splitter
# ---------------------------------------------------------------------------


def _split_into_sections(text: str) -> dict[str, list[str]]:
    """
    Heuristically split guide text into named section buckets by scanning for
    header keywords. Returns a dict of section_name -> [paragraph, ...].
    """
    sections: dict[str, list[str]] = {
        "stat_priority": [],
        "aspects": [],
        "uniques": [],
        "runewords": [],
        "other": [],
    }

    current_section = "other"
    paragraphs = re.split(r"\n{2,}", text)

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        # Check all short lines (likely headers) within the paragraph,
        # not just the first, so compact pastes without blank-line separators work.
        header_candidate_lines = [
            l for l in stripped.split("\n") if l.strip() and len(l.strip()) < 60
        ]

        detected: Optional[str] = None
        for candidate in header_candidate_lines:
            if _STAT_PRIORITY_HEADER_RE.search(candidate):
                detected = "stat_priority"
                break
            if _ASPECT_HEADER_RE.search(candidate):
                detected = "aspects"
                break
            if _UNIQUE_HEADER_RE.search(candidate):
                detected = "uniques"
                break
            if _RUNEWORD_HEADER_RE.search(candidate):
                detected = "runewords"
                break

        if detected:
            current_section = detected

        sections[current_section].append(stripped)

    return sections


# ---------------------------------------------------------------------------
# Internal per-section parsers
# ---------------------------------------------------------------------------


def _parse_stat_priorities(paragraphs: list[str]) -> list[StatPriority]:
    """Extract per-slot stat priority lists from guide paragraphs."""
    priorities: list[StatPriority] = []
    current_slot: Optional[str] = None
    current_stats: list[str] = []
    current_raw: list[str] = []

    def _flush() -> None:
        if current_slot and current_stats:
            priorities.append(
                StatPriority(
                    slot=current_slot,
                    stats=list(current_stats),
                    raw_text="\n".join(current_raw),
                )
            )

    for para in paragraphs:
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue

            slot_match = _SLOT_RE.search(line)
            if slot_match and ":" in line:
                _flush()
                current_slot = slot_match.group(1).title()
                current_stats = []
                current_raw = [line]
                # Inline stats after the colon
                after_colon = line.split(":", 1)[1].strip()
                if after_colon:
                    current_stats = [s.strip() for s in re.split(r"[,>/]", after_colon) if s.strip()]
            elif current_slot:
                current_raw.append(line)
                stat_match = _STAT_LINE_RE.match(line)
                if stat_match:
                    stat_text = stat_match.group(1).strip()
                    if stat_text and len(stat_text) < 80:
                        current_stats.append(stat_text)

    _flush()
    return priorities


def _parse_aspects(paragraphs: list[str]) -> list[RequiredAspect]:
    """Extract required aspects from guide paragraphs."""
    aspects: list[RequiredAspect] = []
    aspect_name_re = re.compile(r"(?:aspect|codex)[^:]*:\s*(.+)", re.IGNORECASE)

    for para in paragraphs:
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue

        name = ""
        description = ""
        slot_rec = ""

        for line in lines:
            name_match = aspect_name_re.search(line)
            if name_match and not name:
                name = name_match.group(1).strip()
                continue

            slot_match = _SLOT_RE.search(line)
            if slot_match and not slot_rec:
                slot_rec = slot_match.group(1).title()

            if not description and len(line) > 20:
                description = line

        if not name and lines:
            name = lines[0][:60].strip()

        if name:
            aspects.append(
                RequiredAspect(
                    name=name,
                    description=description,
                    slot_recommendation=slot_rec or "Any",
                    raw_text=para,
                )
            )

    return aspects


def _parse_uniques(paragraphs: list[str]) -> list[UniqueBreakdown]:
    """Extract unique item breakdowns from guide paragraphs."""
    uniques: list[UniqueBreakdown] = []

    for para in paragraphs:
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        name = lines[0]
        slot = ""
        why_bis = ""
        alternatives: list[str] = []

        for line in lines[1:]:
            slot_match = _SLOT_RE.search(line)
            if slot_match and not slot:
                slot = slot_match.group(1).title()

            if re.search(r"\bwhy\b|\bbecause\b|\bprovides\b|\bgrants\b", line, re.IGNORECASE):
                why_bis = line

            alt_match = re.search(
                r"(?:alternative|replace|substitute|swap)[s:]?\s*(.+)", line, re.IGNORECASE
            )
            if alt_match:
                alt_text = alt_match.group(1)
                alternatives = [a.strip() for a in re.split(r"[,/]", alt_text) if a.strip()]

        if name:
            uniques.append(
                UniqueBreakdown(
                    name=name,
                    slot=slot or "Unknown",
                    why_bis=why_bis or "High-impact unique power.",
                    alternatives=alternatives,
                    raw_text=para,
                )
            )

    return uniques


def _parse_runewords(paragraphs: list[str]) -> list[RunewordBreakdown]:
    """Extract runeword combinations from guide paragraphs."""
    runewords: list[RunewordBreakdown] = []
    rune_combo_re = re.compile(r"\b([A-Z][a-z]+)\s*\+\s*([A-Z][a-z]+)\b")

    for para in paragraphs:
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue

        combo_match = rune_combo_re.search(para)
        name = combo_match.group(0) if combo_match else lines[0][:40]

        slot_match = _SLOT_RE.search(para)
        slot = slot_match.group(1).title() if slot_match else "Unknown"

        effect_lines = [l for l in lines if len(l) > 20 and l != name]
        effect_summary = effect_lines[0] if effect_lines else "See guide for details."

        runewords.append(
            RunewordBreakdown(
                name=name,
                slot=slot,
                effect_summary=effect_summary,
                raw_text=para,
            )
        )

    return runewords


# ---------------------------------------------------------------------------
# Utility detectors
# ---------------------------------------------------------------------------


def _detect_class(text: str) -> Optional[str]:
    match = _D4_CLASS_RE.search(text[:500])
    return match.group(1).title() if match else None


def _detect_build_name(text: str) -> Optional[str]:
    """Heuristically extract a build name from the first ~200 chars."""
    first_block = text[:200].split("\n")[0].strip()
    if 5 < len(first_block) < 80:
        return first_block
    return None
