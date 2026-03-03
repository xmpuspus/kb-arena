// Universal documentation schema — idempotent, safe to run on every startup
// Works for any documentation domain (AWS, software docs, wikis, etc.)

CREATE CONSTRAINT topic_fqn IF NOT EXISTS FOR (t:Topic) REQUIRE t.fqn IS UNIQUE;
CREATE CONSTRAINT component_fqn IF NOT EXISTS FOR (c:Component) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT process_fqn IF NOT EXISTS FOR (p:Process) REQUIRE p.fqn IS UNIQUE;
CREATE CONSTRAINT config_fqn IF NOT EXISTS FOR (c:Config) REQUIRE c.fqn IS UNIQUE;
CREATE CONSTRAINT constraint_fqn IF NOT EXISTS FOR (c:Constraint) REQUIRE c.fqn IS UNIQUE;

// Multi-label fulltext index for cross-entity search
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Topic|Component|Process|Config|Constraint)
ON EACH [n.name, n.description, n.fqn];
