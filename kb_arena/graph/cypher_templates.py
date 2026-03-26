"""Pre-built Cypher query templates for common graph traversal patterns.

Parameters use $name placeholders — always pass via params dict, never interpolate.
"""

from __future__ import annotations

# Find a single entity by name or fqn, returning all its properties.
SINGLE_ENTITY_LOOKUP = """
MATCH (n {fqn: $fqn})
RETURN n.name AS name,
       n.fqn AS fqn,
       labels(n)[0] AS type,
       n.description AS description,
       n.properties AS properties
LIMIT 1
"""

# Find all entities reachable within $depth hops from $target.
# Filtered to $allowed_rel_types so callers can scope traversal.
MULTI_HOP_QUERY = """
MATCH path = (start {fqn: $target})-[*1..$depth]-(connected)
WHERE ALL(r IN relationships(path) WHERE type(r) IN $allowed_rel_types)
RETURN connected.name AS name,
       connected.fqn AS fqn,
       labels(connected)[0] AS type,
       length(path) AS hops,
       [r IN relationships(path) | type(r)] AS relationship_chain
ORDER BY hops, connected.name
LIMIT 50
"""

# Compare two entities across shared intermediate nodes.
# Returns shared neighbours plus relationships, then unique neighbours of $entity_a.
COMPARISON_QUERY = """
MATCH (a {fqn: $entity_a})-[r1]-(shared)-[r2]-(b {fqn: $entity_b})
RETURN shared.name AS shared_entity,
       labels(shared)[0] AS shared_type,
       type(r1) AS rel_to_a,
       type(r2) AS rel_to_b
UNION
MATCH (a {fqn: $entity_a})-[r]-(unique)
WHERE NOT (unique)--(b {fqn: $entity_b})
RETURN unique.name AS shared_entity,
       labels(unique)[0] AS shared_type,
       type(r) AS rel_to_a,
       null AS rel_to_b
"""

# Trace full dependency chain from $start up to depth 4.
# Uses only valid universal schema relationship types.
DEPENDENCY_CHAIN = """
MATCH path = (source {fqn: $start})
  -[:DEPENDS_ON|CONNECTS_TO|TRIGGERS|EXTENDS|CONFIGURES*1..4]->(dep)
WITH path, dep, length(path) AS depth
RETURN dep.name AS name,
       dep.fqn AS fqn,
       labels(dep)[0] AS type,
       depth,
       [n IN nodes(path) | n.name] AS chain
ORDER BY depth
LIMIT 100
"""

# Find all connections from/to an entity via any relationship.
CROSS_REFERENCE = """
MATCH (entity {fqn: $fqn})-[r]-(other)
RETURN other.name AS name,
       other.fqn AS fqn,
       labels(other)[0] AS type,
       type(r) AS relationship,
       CASE WHEN startNode(r).fqn = $fqn THEN 'outgoing' ELSE 'incoming' END AS direction
ORDER BY direction, other.name
LIMIT 50
"""

# Walk extension/containment hierarchy up and down from $fqn.
TYPE_HIERARCHY = """
MATCH path = (base)-[:EXTENDS|CONTAINS*0..5]->(child {fqn: $fqn})
WITH path, base, length(path) AS depth
RETURN base.name AS ancestor,
       base.fqn AS ancestor_fqn,
       depth,
       [n IN nodes(path) | n.name] AS chain
UNION
MATCH path = (target {fqn: $fqn})-[:EXTENDS|CONTAINS*1..5]->(descendant)
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
RETURN node.name AS name,
       node.fqn AS fqn,
       labels(node)[0] AS type,
       node.description AS description,
       score
ORDER BY score DESC
LIMIT $limit
"""
