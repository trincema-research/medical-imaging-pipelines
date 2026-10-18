"""Silver-label rule. A passing mask is provisional, never gold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SilverRule:
    mean_pairwise_dice: float = 0.97
    min_pairwise_dice: float = 0.95
    anatomy_qc_required: bool = True
    max_relative_area_disagreement: float = 0.05
    left_right_valid: bool = True


def decide_label(
    *,
    mean_dice: float,
    min_dice: float,
    area_disagreement: float,
    anatomy_pass: bool,
    left_right_valid: bool,
    rule: SilverRule | None = None,
    hard_reject: bool = False,
) -> tuple[str, bool]:
    """Return ``(label_status, manual_review_required)``.

    Gold is never returned. Rejected is reserved for hard anatomical failure.
    Everything else that misses the silver bar stays ``pseudo``.
    """
    limits = rule or SilverRule()
    if hard_reject:
        return "rejected", True
    silver = (
        mean_dice >= limits.mean_pairwise_dice
        and min_dice >= limits.min_pairwise_dice
        and area_disagreement <= limits.max_relative_area_disagreement
        and (anatomy_pass or not limits.anatomy_qc_required)
        and (left_right_valid or not limits.left_right_valid)
    )
    if silver:
        return "silver", False
    return "pseudo", True
