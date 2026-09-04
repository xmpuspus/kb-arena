"""How much of a result rests on human-reviewed questions, and what that allows.

A question the model drafted is a fine development signal. It is not evidence
a reader can cite, because nobody checked that its answer key is right. Every
result carries the review status of the questions it scored, and a publication
gate refuses to call a number citable while machine drafts sit under it.
"""

from __future__ import annotations

REVIEWED = "human-reviewed"
DRAFT = "machine-assisted-draft"
UNSPECIFIED = "unspecified"
STATUSES = (REVIEWED, DRAFT, UNSPECIFIED)


def count_statuses(records) -> dict[str, int]:
    """Records per review status, every status present."""
    counts = dict.fromkeys(STATUSES, 0)
    for record in records:
        status = _status_of(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_of(record) -> str:
    if isinstance(record, dict):
        status = record.get("question_review_status", UNSPECIFIED)
    else:
        status = getattr(record, "question_review_status", UNSPECIFIED)
    return status if status in STATUSES else UNSPECIFIED


def review_summary(records) -> dict:
    """The counts, the reviewed share, and whether the result is publishable.

    publishable is true only when every scored question is human-reviewed. A
    result with one draft question is a development signal, not evidence.
    """
    counts = count_statuses(records)
    total = sum(counts.values())
    reviewed = counts[REVIEWED]
    return {
        "counts": counts,
        "questions": total,
        "reviewed_share": round(reviewed / total, 4) if total else None,
        "publishable": bool(total) and reviewed == total,
        "note": (
            "publishable is true only when every scored question is human-reviewed. "
            "A machine-assisted draft is a development signal, not citable evidence."
        ),
    }


def publication_blockers(records) -> list[str]:
    """One line per reason the result must not be published, empty when it can be."""
    counts = count_statuses(records)
    total = sum(counts.values())
    if not total:
        return ["the result scored no questions"]
    reasons = []
    if counts[DRAFT]:
        reasons.append(f"{counts[DRAFT]} of {total} questions are machine-assisted drafts")
    if counts[UNSPECIFIED]:
        reasons.append(f"{counts[UNSPECIFIED]} of {total} questions carry no review status")
    return reasons
