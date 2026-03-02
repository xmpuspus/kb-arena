// Kubernetes schema — idempotent, safe to run on every startup

CREATE CONSTRAINT resource_fqn IF NOT EXISTS FOR (r:Resource) REQUIRE r.fqn IS UNIQUE;
CREATE CONSTRAINT field_fqn IF NOT EXISTS FOR (f:Field) REQUIRE f.fqn IS UNIQUE;
CREATE CONSTRAINT api_group_name IF NOT EXISTS FOR (a:APIGroup) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT controller_fqn IF NOT EXISTS FOR (c:Controller) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT k8s_concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT k8s_example_fqn IF NOT EXISTS FOR (e:Example) REQUIRE e.fqn IS UNIQUE;
CREATE CONSTRAINT k8s_version_name IF NOT EXISTS FOR (v:Version) REQUIRE v.name IS UNIQUE;

// Multi-label fulltext for cross-type search
CREATE FULLTEXT INDEX k8s_entity_search IF NOT EXISTS
FOR (n:Resource|Controller|Concept|APIGroup)
ON EACH [n.name, n.description, n.fqn];

// Vector index for hybrid retrieval
CREATE VECTOR INDEX k8s_resource_embeddings IF NOT EXISTS
FOR (r:Resource) ON (r.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
