"""Graph models for entity/relationship extraction and graph context."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Extracted entity from a document section."""

    id: str
    name: str
    fqn: str  # fully qualified name, e.g. "os.path.join"
    type: str  # must match a NodeType enum value
    description: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    source_section_id: str = ""
    extraction_confidence: float = 1.0
    embedding: list[float] | None = None


class Relationship(BaseModel):
    """Extracted relationship between two entities."""

    source_fqn: str
    target_fqn: str
    type: str  # must match a RelType enum value
    properties: dict[str, Any] = Field(default_factory=dict)
    source_section_id: str = ""
    extraction_confidence: float = 1.0


class ExtractionResult(BaseModel):
    """Result of entity/relationship extraction from a document."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    document_id: str = ""
    section_id: str = ""


class GraphContext(BaseModel):
    """Graph context returned alongside answers for visualization."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    query_path: list[str] = Field(default_factory=list)
    cypher_used: str = ""
