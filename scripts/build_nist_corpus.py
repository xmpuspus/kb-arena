"""Build deterministic KB Arena documents from the NIST SP 800-171r3 HTML snapshot."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Tag

from kb_arena.models.document import CrossRef, Document, Section

CORPUS = "nist-800-171-r3"
SOURCE_URL = (
    "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/800-171r3/" "NIST.SP.800-171r3.html"
)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / CORPUS / "raw" / "nist-sp-800-171r3.html"
OUTPUT = ROOT / "datasets" / CORPUS / "processed" / "documents.jsonl"
QUESTIONS = ROOT / "datasets" / CORPUS / "questions"
QRELS_OUTPUT = QUESTIONS / "expected_chunks.yaml"
CONTROL_HEADING = re.compile(r"^(\d{2}\.\d{2}\.\d{2})\s+(.+)$")
FAMILY_HEADING = re.compile(r"^3\.(\d+)\s+(.+)$")
CONTROL_REFERENCE = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b")
# A trailing \b cannot match after the closing parenthesis, so it would silently
# drop every enhancement and keep only the base control.
SOURCE_CONTROL = re.compile(r"\b[A-Z]{2}-\d{2}(?!\d)(?:\(\d{2}\))?")
# The document bibliography follows the last control and is not part of it.
BIBLIOGRAPHY_CLASSES = {"BiblioHead", "BiblioEntry"}


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def _family(heading: Tag) -> tuple[str, str]:
    family_heading = heading.find_previous("h2")
    if family_heading is None:
        raise ValueError(f"No family heading before {heading.get_text(' ', strip=True)!r}")
    match = FAMILY_HEADING.match(_text(family_heading))
    if match is None:
        raise ValueError(f"Invalid family heading: {_text(family_heading)!r}")
    return f"03.{int(match.group(1)):02d}", match.group(2)


def _anchor(heading: Tag, control_id: str) -> str:
    for link in heading.find_all("a"):
        value = link.get("id", "")
        if value.startswith("sec-sec_"):
            return value
    return f"sec-sec_{control_id}"


def _parts(heading: Tag) -> dict[str, list[str]]:
    parts: dict[str, list[str]] = {"requirement": [], "discussion": [], "references": []}
    part = "requirement"
    for sibling in heading.next_siblings:
        if not isinstance(sibling, Tag):
            continue
        if sibling.name in {"h2", "h3"}:
            break
        if BIBLIOGRAPHY_CLASSES.intersection(sibling.get("class") or ()):
            break
        if sibling.name == "h4":
            label = _text(sibling).lower()
            if label == "discussion":
                part = "discussion"
            elif label == "references":
                part = "references"
            continue
        value = _text(sibling)
        if value:
            parts[part].append(value)
    return parts


def build_documents(source: Path = SOURCE) -> list[Document]:
    """Parse the official snapshot into one document for each control identifier."""
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    soup = BeautifulSoup(source_bytes, "html.parser")
    documents: list[Document] = []

    for heading in soup.find_all("h3"):
        match = CONTROL_HEADING.match(_text(heading))
        if match is None:
            continue

        control_id, title = match.groups()
        family_id, family_title = _family(heading)
        anchor = _anchor(heading, control_id)
        parts = _parts(heading)
        blocks = []
        for label in ("requirement", "discussion", "references"):
            if parts[label]:
                blocks.append(f"{label.title()}:\n" + "\n\n".join(parts[label]))
        content = "\n\n".join(blocks) or "Withdrawn."
        related = sorted(set(CONTROL_REFERENCE.findall(content)) - {control_id})
        source_controls = sorted(set(SOURCE_CONTROL.findall(content)))
        source_anchor = f"{SOURCE_URL}#{anchor}"

        section = Section(
            id=control_id,
            title=f"{control_id} {title}",
            content=content,
            heading_path=[family_title, title],
            links=[
                CrossRef(
                    target=f"nist-{target}::{target}",
                    label=target,
                    ref_type="nist-control",
                )
                for target in related
            ],
            level=2,
        )
        documents.append(
            Document(
                id=f"nist-{control_id}",
                source=source_anchor,
                corpus=CORPUS,
                title=f"{control_id} {title}",
                sections=[section],
                metadata={
                    "control_id": control_id,
                    "control_title": title,
                    "family_id": family_id,
                    "family_title": family_title,
                    "source_anchor": anchor,
                    "source_url": SOURCE_URL,
                    "source_sha256": source_sha256,
                    "publication": "NIST SP 800-171 Revision 3",
                    "publication_date": "2024-05",
                    "license": "Not subject to copyright in the United States",
                    "withdrawn": title.strip().lower() == "withdrawn",
                    "related_control_ids": related,
                    "source_controls": source_controls,
                },
                raw_token_count=len(content.split()),
            )
        )

    return documents


def build_qrels(documents: list[Document], questions_dir: Path = QUESTIONS) -> dict[str, list[str]]:
    """Map each draft question's declared source controls to processed sections."""
    sections_by_document = {
        document.id: [f"{document.id}::{section.id}" for section in document.sections]
        for document in documents
    }
    qrels: dict[str, list[str]] = {}

    for question_path in sorted(questions_dir.glob("*.yaml")):
        if question_path.name == QRELS_OUTPUT.name:
            continue
        for question in yaml.safe_load(question_path.read_text(encoding="utf-8")) or []:
            question_id = question["id"]
            if question_id in qrels:
                raise ValueError(f"Duplicate question identifier: {question_id}")

            targets: list[str] = []
            for source_ref in question["ground_truth"]["source_refs"]:
                try:
                    sections = sections_by_document[source_ref]
                except KeyError as exc:
                    raise ValueError(
                        f"Question {question_id} references unknown document {source_ref}"
                    ) from exc
                targets.extend(section for section in sections if section not in targets)
            if not targets:
                raise ValueError(f"Question {question_id} has no expected sections")
            qrels[question_id] = targets

    return qrels


def main() -> None:
    """Write deterministic processed documents and question relevance labels."""
    documents = build_documents()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as output:
        for document in documents:
            output.write(document.model_dump_json())
            output.write("\n")
    qrels = build_qrels(documents)
    QRELS_OUTPUT.write_text(
        yaml.safe_dump(qrels, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(documents)} controls to {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {len(qrels)} question labels to {QRELS_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
