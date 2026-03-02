// SEC EDGAR schema — idempotent, safe to run on every startup

CREATE CONSTRAINT company_fqn IF NOT EXISTS FOR (c:Company) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT executive_fqn IF NOT EXISTS FOR (e:Executive) REQUIRE e.fqn IS UNIQUE;
CREATE CONSTRAINT board_member_fqn IF NOT EXISTS FOR (b:BoardMember) REQUIRE b.fqn IS UNIQUE;
CREATE CONSTRAINT subsidiary_fqn IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.fqn IS UNIQUE;
CREATE CONSTRAINT risk_factor_fqn IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.fqn IS UNIQUE;
CREATE CONSTRAINT financial_metric_fqn IF NOT EXISTS FOR (f:FinancialMetric) REQUIRE f.fqn IS UNIQUE;
CREATE CONSTRAINT legal_proceeding_fqn IF NOT EXISTS FOR (l:LegalProceeding) REQUIRE l.fqn IS UNIQUE;
CREATE CONSTRAINT segment_fqn IF NOT EXISTS FOR (s:Segment) REQUIRE s.fqn IS UNIQUE;

// Multi-label fulltext across entity types most likely searched by name
CREATE FULLTEXT INDEX sec_entity_search IF NOT EXISTS
FOR (n:Company|Executive|RiskFactor|Segment)
ON EACH [n.name, n.description, n.fqn];

// Vector index on Company nodes for semantic similarity search
CREATE VECTOR INDEX sec_company_embeddings IF NOT EXISTS
FOR (c:Company) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
