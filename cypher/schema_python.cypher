// Python stdlib schema — idempotent, safe to run on every startup

CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT module_fqn IF NOT EXISTS FOR (m:Module) REQUIRE m.fqn IS UNIQUE;
CREATE CONSTRAINT class_fqn IF NOT EXISTS FOR (c:Class) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT function_fqn IF NOT EXISTS FOR (f:Function) REQUIRE f.fqn IS UNIQUE;
CREATE CONSTRAINT parameter_fqn IF NOT EXISTS FOR (p:Parameter) REQUIRE p.fqn IS UNIQUE;
CREATE CONSTRAINT return_type_fqn IF NOT EXISTS FOR (r:ReturnType) REQUIRE r.fqn IS UNIQUE;
CREATE CONSTRAINT exception_fqn IF NOT EXISTS FOR (e:Exception) REQUIRE e.fqn IS UNIQUE;
CREATE CONSTRAINT example_fqn IF NOT EXISTS FOR (e:Example) REQUIRE e.fqn IS UNIQUE;

// Multi-label fulltext index for cross-entity search (single query, all types)
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Concept|Module|Class|Function)
ON EACH [n.name, n.description, n.fqn];

// Vector index for hybrid search — 1536 dims matches text-embedding-3-small
CREATE VECTOR INDEX concept_embeddings IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
