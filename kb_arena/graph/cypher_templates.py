"""Pre-built Cypher query templates for common graph traversal patterns.

Parameters use $name placeholders — always pass via params dict, never interpolate.
"""

from __future__ import annotations

# Find a single entity by name or fqn, returning all its properties.
SINGLE_ENTITY_LOOKUP = """
MATCH (n:KBArenaEntity {fqn: $fqn})
RETURN n.name AS name,
       n.fqn AS fqn,
       head([label IN labels(n) WHERE label <> 'KBArenaEntity']) AS type,
       n.description AS description,
       n.properties AS properties
LIMIT 1
"""

# Find all entities reachable within $depth hops from $target.
# Filtered to $allowed_rel_types so callers can scope traversal.
MULTI_HOP_QUERY = """
MATCH path = (start:KBArenaEntity {fqn: $target})-[*1..$depth]-(connected:KBArenaEntity)
WHERE ALL(n IN nodes(path) WHERE n:KBArenaEntity)
  AND ALL(r IN relationships(path) WHERE type(r) IN $allowed_rel_types)
RETURN connected.name AS name,
       connected.fqn AS fqn,
       head([label IN labels(connected) WHERE label <> 'KBArenaEntity']) AS type,
       length(path) AS hops,
       [r IN relationships(path) | type(r)] AS relationship_chain
ORDER BY hops, connected.name
LIMIT 50
"""

# Compare two entities across shared intermediate nodes.
# Returns shared neighbours plus relationships, then unique neighbours of $entity_a.
COMPARISON_QUERY = """
MATCH (a:KBArenaEntity {fqn: $entity_a})-[r1]-(shared:KBArenaEntity)
  -[r2]-(b:KBArenaEntity {fqn: $entity_b})
RETURN shared.name AS shared_entity,
       head([label IN labels(shared) WHERE label <> 'KBArenaEntity']) AS shared_type,
       type(r1) AS rel_to_a,
       type(r2) AS rel_to_b
UNION
MATCH (a:KBArenaEntity {fqn: $entity_a})-[r]-(unique:KBArenaEntity)
WHERE NOT (unique)--(b:KBArenaEntity {fqn: $entity_b})
RETURN unique.name AS shared_entity,
       head([label IN labels(unique) WHERE label <> 'KBArenaEntity']) AS shared_type,
       type(r) AS rel_to_a,
       null AS rel_to_b
"""

# Trace full dependency chain from $start up to depth 4.
# Uses only valid universal schema relationship types.
DEPENDENCY_CHAIN = """
MATCH path = (source:KBArenaEntity {fqn: $start})
  -[:DEPENDS_ON|CONNECTS_TO|TRIGGERS|EXTENDS|CONFIGURES*1..4]->(dep:KBArenaEntity)
WHERE ALL(n IN nodes(path) WHERE n:KBArenaEntity)
WITH path, dep, length(path) AS depth
RETURN dep.name AS name,
       dep.fqn AS fqn,
       head([label IN labels(dep) WHERE label <> 'KBArenaEntity']) AS type,
       depth,
       [n IN nodes(path) | n.name] AS chain
ORDER BY depth
LIMIT 100
"""

# Find all connections from/to an entity via any relationship.
CROSS_REFERENCE = """
MATCH (entity:KBArenaEntity {fqn: $fqn})-[r]-(other:KBArenaEntity)
RETURN other.name AS name,
       other.fqn AS fqn,
       head([label IN labels(other) WHERE label <> 'KBArenaEntity']) AS type,
       type(r) AS relationship,
       CASE WHEN startNode(r).fqn = $fqn THEN 'outgoing' ELSE 'incoming' END AS direction
ORDER BY direction, other.name
LIMIT 50
"""

# Walk extension/containment hierarchy up and down from $fqn.
TYPE_HIERARCHY = """
MATCH path = (base:KBArenaEntity)-[:EXTENDS|CONTAINS*0..5]->
  (child:KBArenaEntity {fqn: $fqn})
WITH path, base, length(path) AS depth
RETURN base.name AS ancestor,
       base.fqn AS ancestor_fqn,
       depth,
       [n IN nodes(path) | n.name] AS chain
UNION
MATCH path = (target:KBArenaEntity {fqn: $fqn})-[:EXTENDS|CONTAINS*1..5]->
  (descendant:KBArenaEntity)
RETURN descendant.name AS ancestor,
       descendant.fqn AS ancestor_fqn,
       length(path) AS depth,
       [n IN nodes(path) | n.name] AS chain
ORDER BY depth
LIMIT 50
"""

# Full-text search across Concept|Module|Class|Function using the entity_search index.
FULLTEXT_ENTITY_SEARCH = """
CALL db.index.fulltext.queryNodes('entity_search', $query)
YIELD node, score
WHERE node:KBArenaEntity
RETURN node.name AS name,
       node.fqn AS fqn,
       head([label IN labels(node) WHERE label <> 'KBArenaEntity']) AS type,
       node.description AS description,
       score
ORDER BY score DESC
LIMIT $limit
"""
