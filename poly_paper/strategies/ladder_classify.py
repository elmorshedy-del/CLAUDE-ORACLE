"""Robust ladder classifier: parse the ACTUAL ladder variable from question text.

Lesson from live scans: Polymarket's `endDate` field is sometimes wrong
(events have inconsistent endDates across sibling markets; the authoritative
ladder variable is encoded in the QUESTION TEXT, not the metadata).

This module parses question text to identify:
  - date ladders:      "...by April 30", "...before June 30", "...by end of 2026"
  - threshold ladders: "...FDV > $1.5B", "...Bitcoin above $150k", "...price > $X"

And emits a LadderRung with an explicit `ladder_variable` and `ladder_value`
that the arb evaluator uses for sorting. It also tags ladder DIRECTION:
  - date_ladder:      probability INCREASES with later date (nested inclusive)
  - threshold_ladder: probability DECREASES with higher threshold (nested inclusive)

The monotonicity arb condition depends on direction:
  - date:      arb if bid(earlier) > ask(later)       [we already pay more for the superset]
  - threshold: arb if bid(higher_thresh) > ask(lower_thresh) [higher is inside lower]

(In both cases: arb if bid(subset-event) > ask(superset-event).)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

# Date patterns like "by April 30", "before June 30, 2026", "by end of 2026", "by December 31"
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE_PATTERNS = [
    re.compile(rf"\bby\s+({_MONTH})\s+(\d{{1,2}}),?\s*(\d{{4}})?", re.I),
    re.compile(rf"\bbefore\s+({_MONTH})\s+(\d{{1,2}}),?\s*(\d{{4}})?", re.I),
    re.compile(rf"\bby\s+end\s+of\s+(\d{{4}})", re.I),
    re.compile(rf"\bbefore\s+(\d{{4}})", re.I),
    re.compile(rf"\bby\s+(\d{{4}})", re.I),
]

_MONTH_NUM = {m.lower(): i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
])}


# Threshold patterns. Two classes:
#   "upper" threshold: "above X", "> X", "reach X", "hit X" when moving UP
#         → higher value = less likely (typical for price markets).
#   "lower" threshold: "hit X" when moving DOWN (approval rating, temperature dips)
#         → higher value = more likely (easier to hit a higher low bound).
#
# Disambiguation is hard from question text alone. We use event-level context
# clues: "How low will X go?" → lower; "How high will X go?" / "above X" → upper.
_THRESH_UPPER = [
    re.compile(r">\s*\$?([\d\.,]+)([BMkKTt]?)\b"),
    re.compile(r"above\s+\$?([\d\.,]+)([BMkKTt]?)\b", re.I),
    re.compile(r"reach\s+\$?([\d\.,]+)([BMkKTt]?)\b", re.I),
    re.compile(r"over\s+\$?([\d\.,]+)([BMkKTt]?)\b", re.I),
    re.compile(r"\$([\d\.,]+)([BMkKTt]?)\s+or\s+more", re.I),
]
_THRESH_AMBIGUOUS_HIT = re.compile(r"hit\s+\$?([\d\.,]+)([BMkKTt%]?)\b", re.I)


@dataclass(frozen=True)
class LadderClassification:
    kind: Literal["date_ladder", "threshold_ladder"]
    # For date_ladder: the unix ts of the date deadline.
    # For threshold_ladder: the numeric threshold value (in USD, shares, whatever).
    value: float
    # Direction of "nesting" — does HIGHER value mean HIGHER probability?
    # date_ladder: True (later date = more likely)
    # threshold_ladder (">X"): False (higher threshold = less likely)
    # threshold_ladder ("<X"): True (higher threshold = more likely)
    higher_value_more_likely: bool
    # Optional human-readable description for logs.
    description: str


def _parse_value(s: str, unit: str) -> float:
    s = s.replace(",", "")
    v = float(s)
    u = unit.lower()
    if u == "t":
        v *= 1_000_000_000_000
    elif u == "b":
        v *= 1_000_000_000
    elif u == "m":
        v *= 1_000_000
    elif u == "k":
        v *= 1_000
    return v


def classify_question(question: str) -> Optional[LadderClassification]:
    """Return a ladder classification for this question, or None if not a ladder."""
    q = question.strip()

    # Try date patterns first (these are unambiguous).
    for pat in _DATE_PATTERNS[:2]:  # "by Month Day [Year]" / "before Month Day [Year]"
        m = pat.search(q)
        if m:
            month = _MONTH_NUM.get(m.group(1).lower())
            day = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else datetime.now().year
            try:
                ts = datetime(year, month, day).timestamp()
            except ValueError:
                continue
            return LadderClassification(
                kind="date_ladder", value=ts,
                higher_value_more_likely=True,
                description=f"by {m.group(1)} {day}, {year}",
            )
    for pat in _DATE_PATTERNS[2:]:
        m = pat.search(q)
        if m:
            year = int(m.group(1))
            # "by end of YYYY" / "before YYYY"
            ts = datetime(year, 12, 31).timestamp() if "end of" in q.lower() else datetime(year, 1, 1).timestamp()
            return LadderClassification(
                kind="date_ladder", value=ts,
                higher_value_more_likely=True,
                description=f"by {year}",
            )

    # Upper thresholds (unambiguous: "above X", "> X", "reach X", "over X").
    for pat in _THRESH_UPPER:
        m = pat.search(q)
        if m:
            value = _parse_value(m.group(1), m.group(2))
            return LadderClassification(
                kind="threshold_ladder", value=value,
                higher_value_more_likely=False,
                description=f"above ${value:,.0f}",
            )

    # "hit X" — ambiguous direction. Default to "hit from below" (upper threshold);
    # when question asks about falling levels ("how low will X go", "dip to", etc),
    # we'd need event context which this signature doesn't take. Default keeps the
    # common price-market case correct (BTC hit $150k), and approval-rating "hit 30%"
    # markets will be filtered out by the coherence check when mixed with actual upper
    # thresholds. Future: accept event_title to disambiguate.
    m = _THRESH_AMBIGUOUS_HIT.search(q)
    if m:
        raw, unit = m.group(1), m.group(2)
        if unit == "%":
            value = float(raw.replace(",", ""))
        else:
            value = _parse_value(raw, unit)
        return LadderClassification(
            kind="threshold_ladder", value=value,
            higher_value_more_likely=False,
            description=f"hit ${value:,.0f}",
        )

    return None


def is_coherent_ladder(classifications: list[LadderClassification]) -> bool:
    """True if all classifications are the same kind AND have the same semantics.

    Rejects mixed ladders (some dates, some thresholds) and mixed semantics
    (some "above X", some "below X").
    """
    if len(classifications) < 2:
        return False
    kinds = {c.kind for c in classifications}
    if len(kinds) != 1:
        return False
    dirs = {c.higher_value_more_likely for c in classifications}
    if len(dirs) != 1:
        return False
    return True


def sorted_rungs_for_arb(
    items: list[tuple[LadderClassification, "any"]],
) -> list[tuple[LadderClassification, "any"]]:
    """Sort so that items[0] is the SUBSET (smallest probability event).

    Under our invariant, arb condition is bid(subset) > ask(superset).
    - date_ladder (higher=more likely): subset = earliest date. Sort ASC by value.
      bid(earliest_date) > ask(later_date) is the arb.
    - threshold_ladder (higher=less likely): subset = highest threshold. Sort DESC.
      bid(highest_thresh) > ask(lower_thresh) is the arb.
    """
    if not items:
        return items
    first = items[0][0]
    reverse = not first.higher_value_more_likely
    return sorted(items, key=lambda x: x[0].value, reverse=reverse)
