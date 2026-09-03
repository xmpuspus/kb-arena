"""Calibrate the answer judge on a small labelled set before trusting its scores.

Each item pairs a question with a reference answer and a candidate. Correct candidates use
different words from the reference on purpose. A judge that only measures similarity to
the reference scores them low and lands outside the expected band.
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kb_arena.benchmark.evaluator import EvaluationExecutionError, evaluate
from kb_arena.models.benchmark import Constraints, GroundTruth

PACKAGED_SET = "judge_calibration.json"


@dataclass(frozen=True)
class CalibrationItem:
    id: str
    label: str
    question: str
    reference: str
    candidate: str
    expected_min: float
    expected_max: float


@dataclass
class CalibrationOutcome:
    id: str
    label: str
    accuracy: float | None
    expected_min: float
    expected_max: float
    in_band: bool
    error: str = ""


@dataclass
class CalibrationReport:
    outcomes: list[CalibrationOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def in_band(self) -> int:
        return sum(1 for o in self.outcomes if o.in_band)

    @property
    def agreement(self) -> float:
        return self.in_band / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "in_band": self.in_band,
            "agreement": round(self.agreement, 4),
            "outcomes": [asdict(o) for o in self.outcomes],
        }


def load_calibration_items(path: str | Path | None = None) -> list[CalibrationItem]:
    if path is None:
        text = importlib.resources.files("kb_arena.data").joinpath(PACKAGED_SET).read_text()
    else:
        text = Path(path).read_text()
    raw = json.loads(text)
    items = raw["items"] if isinstance(raw, dict) else raw
    return [CalibrationItem(**item) for item in items]


async def run_calibration(llm, items: list[CalibrationItem]) -> CalibrationReport:
    """Score every item with the configured judge and compare with its expected band."""
    report = CalibrationReport()
    for item in items:
        try:
            score = await evaluate(
                item.candidate,
                GroundTruth(answer=item.reference),
                Constraints(),
                llm=llm,
                question_text=item.question,
            )
        except EvaluationExecutionError as exc:
            report.outcomes.append(
                CalibrationOutcome(
                    id=item.id,
                    label=item.label,
                    accuracy=None,
                    expected_min=item.expected_min,
                    expected_max=item.expected_max,
                    in_band=False,
                    error=str(exc),
                )
            )
            continue
        accuracy = float(score.accuracy)
        report.outcomes.append(
            CalibrationOutcome(
                id=item.id,
                label=item.label,
                accuracy=accuracy,
                expected_min=item.expected_min,
                expected_max=item.expected_max,
                in_band=item.expected_min <= accuracy <= item.expected_max,
            )
        )
    return report
