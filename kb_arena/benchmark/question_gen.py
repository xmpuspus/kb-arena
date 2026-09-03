"""LLM-powered benchmark question generation from processed documents.

Reads JSONL documents for a corpus, generates questions per tier,
writes YAML matching the existing Question schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from kb_arena.llm.client import LLMClient
from kb_arena.settings import settings

TIER_DEFS = {
    1: {
        "name": "Lookup",
        "desc": "Single fact retrieval from one document or topic. One-hop.",
        "type": "factoid",
        "hops": 1,
    },
    2: {
        "name": "How-To",
        "desc": "Step-by-step procedure within one topic. One-hop.",
        "type": "procedural",
        "hops": 1,
    },
    3: {
        "name": "Comparison",
        "desc": "Choosing between two options, services, or configurations. Two-hop.",
        "type": "comparison",
        "hops": 2,
    },
    4: {
        "name": "Integration",
        "desc": "Cross-topic dependencies requiring 3-4 connected concepts. Three-hop.",
        "type": "relational",
        "hops": 3,
    },
    5: {
        "name": "Architecture",
        "desc": "Full system design spanning 3-5+ topics. Four-hop.",
        "type": "multi-hop",
        "hops": 4,
    },
}

SYSTEM_PROMPT = """You generate benchmark questions for evaluating documentation retrieval systems.

Given documentation excerpts, create questions at the specified difficulty tier.
Each question must be answerable from the provided documentation.

Output valid JSON array. Each element:
{{
  "question": "...",
  "answer": "Ground truth answer (2-4 sentences, factual, cite specific details from docs)",
  "source_refs": ["document source paths that contain the answer"],
  "required_entities": ["key terms that must appear in a correct answer"],
  "must_mention": ["terms a correct answer must include"],
  "must_not_claim": ["common misconceptions to watch for"]
}}

Rules:
- Questions must be answerable ONLY from the provided documentation
- Ground truth answers must cite specific facts, numbers, or procedures from the docs
- Higher tiers require synthesizing information across multiple documents
- Avoid yes/no questions — require explanatory answers
- Include diverse question patterns (what, how, why, compare, when)
"""


_SECTION_SLICE = 2000  # the most one section contributes, same as before
_DOC_SEPARATOR = "\n\n---\n\n"
_MIN_DOC_SHARE = 200  # a header plus one sentence; below this a share is useless


def _cut(text: str, limit: int) -> str:
    """Cut text to limit at a word boundary, so an excerpt never ends mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit // 2 else cut


def _load_doc_excerpts(
    corpus: str, max_chars: int = 50000
) -> tuple[list[dict], str, dict[str, int]]:
    """Build the excerpt text the generator reads, with a share for every document.

    The old walk took sections in corpus order until one global cap filled,
    so on the 130-document NIST corpus only the first 36 documents ever
    reached the prompt. Now every document gets an equal share of max_chars
    first. A second pass hands the room that short documents leave unused to
    the documents that still have content, in order. max_chars stays a hard
    cap. When the corpus is so large that an equal share cannot hold a header
    and a sentence, an evenly spaced subset of documents gets a useful share
    instead, and the coverage record says how many.

    Returns the raw documents, the excerpt text, and a coverage record.
    """
    processed_dir = Path(settings.datasets_path) / corpus / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"No processed data at {processed_dir}. Run 'kb-arena ingest' first."
        )

    docs = []
    for jsonl_file in sorted(processed_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

    if not docs:
        raise ValueError(f"No documents found in {processed_dir}")

    # Usable sections per document, each already cut to the section slice.
    usable: list[list[tuple[str, str]]] = []
    for doc in docs:
        title = doc.get("title", "Untitled")
        parts = []
        for section in doc.get("sections", []):
            content = section.get("content", "")
            if not content or len(content) < 50:
                continue
            header = f"[{title} / {section.get('title', '')}]\n"
            parts.append((header, content[:_SECTION_SLICE]))
        usable.append(parts)

    with_content = [i for i, parts in enumerate(usable) if parts]
    selected = list(with_content)
    if selected and max_chars // len(selected) < _MIN_DOC_SHARE:
        per_doc_cost = _MIN_DOC_SHARE + len(_DOC_SEPARATOR)
        stride = -(-len(selected) * per_doc_cost // max_chars)  # ceil
        selected = selected[::stride]
    # The separators between documents count against the cap too.
    budget = max_chars - len(_DOC_SEPARATOR) * max(0, len(selected) - 1)
    share = budget // len(selected) if selected else 0

    # Pass one: an equal share per selected document. Pass two: hand the
    # leftover room to documents that still have content, in order.
    taken: dict[int, list[str]] = {i: [] for i in selected}
    cursor: dict[int, tuple[int, int]] = {i: (0, 0) for i in selected}  # (section, offset)
    used = 0

    def _fill(i: int, room: int) -> int:
        spent = 0
        sec, off = cursor[i]
        parts = usable[i]
        while sec < len(parts) and room - spent > 0:
            header, body = parts[sec]
            rest = body[off:]
            need_header = off == 0
            overhead = len(header) if need_header else 0
            if room - spent - overhead < 50:
                break
            piece = _cut(rest, room - spent - overhead)
            if not piece:
                break
            taken[i].append((header if need_header else "") + piece)
            spent += overhead + len(piece)
            if len(piece) < len(rest):
                off += len(piece)
                # skip the whitespace the word-boundary cut left behind
                while off < len(body) and body[off] == " ":
                    off += 1
            else:
                sec, off = sec + 1, 0
        cursor[i] = (sec, off)
        return spent

    for i in selected:
        used += _fill(i, share)
    leftover = budget - used
    for i in selected:
        if leftover <= 0:
            break
        spent = _fill(i, leftover)
        leftover -= spent
        used += spent

    excerpt_text = _DOC_SEPARATOR.join("".join(taken[i]) for i in selected if taken[i])
    contributed = sum(1 for i in selected if taken[i])
    coverage = {
        "documents": len(docs),
        "documents_with_usable_section": len(with_content),
        "documents_selected": len(selected),
        "documents_in_prompt": contributed,
        "documents_without_usable_section": len(docs) - len(with_content),
        "documents_without_room": len(selected) - contributed,
        "chars": len(excerpt_text),
        "max_chars": max_chars,
        "per_document_share": share,
    }
    return docs, excerpt_text, coverage


async def _generate_tier_questions(
    llm: LLMClient,
    tier: int,
    tier_def: dict,
    excerpt_text: str,
    corpus: str,
    count: int,
) -> list[dict]:
    user_prompt = f"""Generate exactly {count} questions at Tier {tier} ({tier_def["name"]}).

Tier definition: {tier_def["desc"]}

Documentation corpus: {corpus}

Documentation excerpts:
{excerpt_text}

Return a JSON array of {count} question objects. Nothing else — just the JSON array."""

    resp = await llm.generate(
        query=user_prompt,
        context="",
        system_prompt=SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.7,
    )

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"\[[\s\S]*\]", resp.text)
    if not json_match:
        raise ValueError(f"No JSON array found in LLM response for tier {tier}")

    raw_questions = json.loads(json_match.group())

    questions = []
    for i, q in enumerate(raw_questions[:count]):
        qid = f"{corpus}-t{tier}-{i + 1:03d}"
        questions.append(
            {
                "id": qid,
                "tier": tier,
                "type": tier_def["type"],
                "hops": tier_def["hops"],
                # Nobody has read these yet. The label travels with the
                # question so a result can never present a draft as reviewed.
                "review_status": "machine-assisted-draft",
                "reviewed_by": "kb-arena generate-questions draft pass",
                "question": q["question"],
                "ground_truth": {
                    "answer": q.get("answer", ""),
                    "source_refs": q.get("source_refs", []),
                    "required_entities": q.get("required_entities", []),
                },
                "constraints": {
                    "must_mention": q.get("must_mention", []),
                    "must_not_claim": q.get("must_not_claim", []),
                    "max_tokens": 300 if tier <= 2 else 500,
                },
            }
        )

    return questions


async def run_question_generation(corpus: str, count: int = 50) -> None:
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold]Generating {count} questions for corpus: {corpus}[/bold]\n")

    docs, excerpt_text, coverage = _load_doc_excerpts(corpus)
    console.print(
        f"  Loaded {len(docs)} documents, {coverage['chars']:,} of {coverage['max_chars']:,} "
        f"prompt chars used. {coverage['documents_in_prompt']} of {coverage['documents']} "
        f"documents reach the prompt, {coverage['documents_without_usable_section']} have no "
        f"section of 50 chars or more, {coverage['documents_without_room']} got no room."
    )

    llm = LLMClient()
    questions_dir = Path(settings.datasets_path) / corpus / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    (questions_dir / "question_coverage.json").write_text(
        json.dumps({"corpus": corpus, **coverage}, indent=2) + "\n"
    )

    per_tier = count // 5
    remainder = count % 5

    tier_names = {
        1: "tier1_factoid",
        2: "tier2_procedural",
        3: "tier3_comparative",
        4: "tier4_relational",
        5: "tier5_multihop",
    }

    total_generated = 0
    active_tiers = [
        (tier, tier_def, per_tier + (1 if tier <= remainder else 0))
        for tier, tier_def in TIER_DEFS.items()
        if per_tier + (1 if tier <= remainder else 0) > 0
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} tiers"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating questions", total=len(active_tiers))

        for tier, tier_def, tier_count in active_tiers:
            progress.update(
                task,
                description=f"Tier {tier} [bold]{tier_def['name']}[/bold] ({tier_count} questions)",
            )

            questions = await _generate_tier_questions(
                llm, tier, tier_def, excerpt_text, corpus, tier_count
            )

            out_path = questions_dir / f"{tier_names[tier]}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(
                    questions, f, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

            total_generated += len(questions)
            progress.advance(task)

    console.print(f"\n[green]Generated {total_generated} questions in {questions_dir}/[/green]")
