"""Two-threshold Jaro-Winkler entity resolution (paper-trail-ph pattern)."""

from __future__ import annotations

import re

import jellyfish

from kb_arena.models.graph import Entity

MERGE_THRESHOLD = 0.92   # auto-merge — proven in paper-trail-ph
REVIEW_THRESHOLD = 0.85  # queue for manual inspection

_NOISE_SUFFIXES = re.compile(
    r"\s*\(\)\s*$|\s+(class|function|method|module|type)\s*$",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    """Strip noise before similarity comparison.

    Upper-cases and removes common suffixes that don't distinguish entities.
    Short result (<3 chars) suggests the original was noise — caller handles.
    """
    name = _NOISE_SUFFIXES.sub("", name).strip().upper()
    return name


def resolve_entities(
    entities: list[Entity],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, float]]]:
    """Return (merged_pairs, review_queue).

    merged_pairs: list of (a.id, b.id, canonical.id) — b was merged into canonical.
    review_queue: list of (name_a, name_b, score) — human inspection needed.

    Rules:
    - Only compare entities of the same type (prevents false cross-type merges)
    - Skip strings shorter than 3 chars after normalization
    - Longer name becomes canonical (preserves more detail)
    - Aliases are appended to the canonical entity for auditability
    """
    merged: list[tuple[str, str, str]] = []
    review_queue: list[tuple[str, str, float]] = []
    # Track which entity IDs have already been absorbed into another
    absorbed: set[str] = set()

    for i, a in enumerate(entities):
        if a.id in absorbed:
            continue
        norm_a = normalize_name(a.name)
        if len(norm_a) < 3:
            continue

        for b in entities[i + 1 :]:
            if b.id in absorbed:
                continue
            if a.type != b.type:
                continue

            norm_b = normalize_name(b.name)
            if len(norm_b) < 3:
                continue

            score = jellyfish.jaro_winkler_similarity(norm_a, norm_b)

            if score >= MERGE_THRESHOLD:
                # Longer name becomes canonical
                canonical, alias = (a, b) if len(a.name) >= len(b.name) else (b, a)
                canonical.aliases.append(alias.name)
                merged.append((a.id, b.id, canonical.id))
                absorbed.add(alias.id)
            elif score >= REVIEW_THRESHOLD:
                review_queue.append((a.name, b.name, score))

    return merged, review_queue
