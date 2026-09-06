"""The dashboard counts review statuses by the names the backend writes.

`RowProvenance` reads `review.counts` by key to say how many questions a
reviewer checked. A key the backend never writes counts nothing, and the row
then reports zero reviewed questions under a corpus a reviewer signed off.
The three names live in two languages, so this holds them together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kb_arena.benchmark.review import DRAFT, REVIEWED, STATUSES, UNSPECIFIED

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.parametrize(
    ("constant", "status"),
    [
        ("REVIEW_REVIEWED", REVIEWED),
        ("REVIEW_DRAFT", DRAFT),
        ("REVIEW_UNSPECIFIED", UNSPECIFIED),
    ],
)
def test_the_client_names_the_status_the_backend_writes(constant, status):
    api = (WEB / "lib" / "api.ts").read_text()

    assert f'export const {constant} = "{status}";' in api


def test_the_client_carries_every_status_the_backend_defines():
    """A fourth status added in Python must reach the page that counts them."""
    api = (WEB / "lib" / "api.ts").read_text()

    for status in STATUSES:
        assert f'"{status}"' in api, f"the client never names {status}"


def test_the_row_reads_the_counts_through_those_constants():
    """A literal key in the component drifts the moment a status is renamed."""
    provenance = (WEB / "components" / "RowProvenance.tsx").read_text()

    assert "review.counts[REVIEW_REVIEWED]" in provenance
    assert "review.counts[REVIEW_DRAFT]" in provenance
    assert "review.counts[REVIEW_UNSPECIFIED]" in provenance
    assert "publishable" in provenance, "a row says whether it is citable"
