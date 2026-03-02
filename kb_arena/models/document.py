"""Central document interchange model (cloudwright ArchSpec pattern).

Every parser outputs this model. Every strategy reads it.
JSONL intermediate files use Document.model_dump_json().
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Table(BaseModel):
    """Extracted table from documentation."""

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str = ""


class CodeBlock(BaseModel):
    """Extracted code example."""

    language: str = ""
    code: str = ""
    description: str = ""


class CrossRef(BaseModel):
    """Cross-reference link to another section or document."""

    target: str  # fqn or URL
    label: str = ""
    ref_type: str = ""  # "function", "class", "module", "external"


class Section(BaseModel):
    """A section within a document, preserving hierarchy."""

    id: str = Field(pattern=r"^[a-zA-Z0-9_./-]+$")
    title: str
    content: str
    heading_path: list[str] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    links: list[CrossRef] = Field(default_factory=list)
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    level: int = 1


class Document(BaseModel):
    """Unified document model — the central interchange for the entire pipeline.

    Every doc (markdown, HTML, or other format) gets parsed into this model.
    JSONL files in datasets/{corpus}/processed/ contain one Document per line.
    """

    id: str
    source: str  # file path or URL
    corpus: str  # "aws-compute" | "aws-storage" | "aws-networking"
    title: str
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_token_count: int = 0
