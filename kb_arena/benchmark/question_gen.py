"""LLM-powered benchmark question generation from processed documents.

Reads JSONL documents for a corpus, generates questions per tier,
writes YAML matching the existing Question schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

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


def _load_doc_excerpts(corpus: str, max_chars: int = 50000) -> tuple[list[dict], str]:
    """Load processed documents and return excerpts + source list."""
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

    # Build excerpts within token budget
    excerpts = []
    total_chars = 0
    for doc in docs:
        title = doc.get("title", "Untitled")
        source = doc.get("source", "")
        sections = doc.get("sections", [])

        for section in sections:
            content = section.get("content", "")
            if not content or len(content) < 50:
                continue

            excerpt = f"[{title} / {section.get('title', '')}]\n{content[:2000]}"
            if total_chars + len(excerpt) > max_chars:
                break
            excerpts.append({"text": excerpt, "source": source})
            total_chars += len(excerpt)

        if total_chars >= max_chars:
            break

    excerpt_text = "\n\n---\n\n".join(e["text"] for e in excerpts)
    return docs, excerpt_text


async def _generate_tier_questions(
    llm: LLMClient,
    tier: int,
    tier_def: dict,
    excerpt_text: str,
    corpus: str,
    count: int,
) -> list[dict]:
    """Generate questions for a single tier."""
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

    # Validate and format
    questions = []
    for i, q in enumerate(raw_questions[:count]):
        qid = f"{corpus}-t{tier}-{i + 1:03d}"
        questions.append(
            {
                "id": qid,
                "tier": tier,
                "type": tier_def["type"],
                "hops": tier_def["hops"],
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
    """Generate benchmark questions for a corpus and write YAML files."""
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold]Generating {count} questions for corpus: {corpus}[/bold]\n")

    docs, excerpt_text = _load_doc_excerpts(corpus)
    console.print(f"  Loaded {len(docs)} documents, {len(excerpt_text):,} chars of excerpts")

    llm = LLMClient()
    questions_dir = Path(settings.datasets_path) / corpus / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    # Distribute count across 5 tiers
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
    for tier, tier_def in TIER_DEFS.items():
        tier_count = per_tier + (1 if tier <= remainder else 0)
        if tier_count == 0:
            continue

        console.print(f"  Tier {tier} ({tier_def['name']}): generating {tier_count} questions...")

        questions = await _generate_tier_questions(
            llm, tier, tier_def, excerpt_text, corpus, tier_count
        )

        # Write YAML
        out_path = questions_dir / f"{tier_names[tier]}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(questions, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        console.print(f"    Wrote {len(questions)} questions to {out_path}")
        total_generated += len(questions)

    console.print(f"\n[green]Generated {total_generated} questions in {questions_dir}/[/green]")
    console.print(f"Next: [bold]kb-arena benchmark --corpus {corpus}[/bold]")
