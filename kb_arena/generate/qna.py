"""Shared Q&A pair generation logic — used by both the qna_pairs strategy and the standalone CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from kb_arena.llm.client import LLMClient
from kb_arena.models.document import Document, Section

QNA_GENERATION_PROMPT = """You are a technical documentation expert. \
Generate 3-5 question-answer pairs from this documentation section.

Rules:
- Questions should be what a developer would actually ask
- Include at least one multi-hop question referencing concepts from other sections
- Answers must be grounded ONLY in the provided text
- Return valid JSON: [{{"question": "...", "answer": "...", "section_ref": "..."}}]
- No explanations outside the JSON

Section heading: {heading}
Section content:
{content}"""


def parse_qna_json(raw: str) -> list[dict]:
    """Extract JSON array from LLM output that may have markdown fences."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


async def generate_pairs_for_section(
    section: Section,
    doc_id: str,
    llm: LLMClient,
) -> list[dict]:
    """Generate 3-5 Q&A pairs for a single documentation section.

    Returns list of dicts: {question, answer, source_id, section_id, section_ref}
    """
    heading = " > ".join(section.heading_path) if section.heading_path else section.title
    content = section.content

    if section.code_blocks:
        code_snippet = section.code_blocks[0].code[:500]
        content = f"{content}\n\nExample:\n{code_snippet}"

    prompt = QNA_GENERATION_PROMPT.format(heading=heading, content=content[:2000])
    resp = await llm.extract(text=prompt, system_prompt="Return only valid JSON. No prose.")
    pairs = parse_qna_json(resp.text)
    if not pairs:
        raise ValueError(f"Q&A generation returned no valid pairs for {doc_id}/{section.id}")

    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"Q&A generation returned a non-object for {doc_id}/{section.id}")
        question = pair.get("question")
        answer = pair.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Q&A generation returned an incomplete pair for {doc_id}/{section.id}"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Q&A generation returned an incomplete pair for {doc_id}/{section.id}"
            )
        pair["source_id"] = doc_id
        pair["section_id"] = section.id
        pair.setdefault("section_ref", heading)

    return pairs


async def generate_pairs_for_documents(
    documents: list[Document],
    llm: LLMClient,
    on_progress: Callable[[str, int], None] | None = None,
) -> list[dict]:
    """Generate Q&A pairs for all sections across all documents.

    on_progress receives (doc_id, pair_count) after each document completes.
    Returns flat list of all generated pairs.
    """
    all_pairs: list[dict] = []

    for doc in documents:
        doc_pairs: list[dict] = []
        for section in doc.sections:
            if not section.content.strip():
                continue
            pairs = await generate_pairs_for_section(section, doc.id, llm)
            doc_pairs.extend(pairs)

        all_pairs.extend(doc_pairs)
        if on_progress:
            on_progress(doc.id, len(doc_pairs))

    return all_pairs
