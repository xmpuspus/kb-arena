// Universal documentation schema - non-destructive and idempotent
// Works for any documentation domain (AWS, software docs, wikis, etc.)

CREATE CONSTRAINT kb_arena_entity_id IF NOT EXISTS
FOR (n:KBArenaEntity) REQUIRE n.entity_id IS UNIQUE;
CREATE CONSTRAINT topic_entity_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.entity_id IS UNIQUE;
CREATE CONSTRAINT component_entity_id IF NOT EXISTS FOR (c:Component) REQUIRE c.entity_id IS UNIQUE;
CREATE CONSTRAINT process_entity_id IF NOT EXISTS FOR (p:Process) REQUIRE p.entity_id IS UNIQUE;
CREATE CONSTRAINT config_entity_id IF NOT EXISTS FOR (c:Config) REQUIRE c.entity_id IS UNIQUE;
CREATE CONSTRAINT constraint_entity_id IF NOT EXISTS FOR (c:Constraint) REQUIRE c.entity_id IS UNIQUE;

// Queries also require the KBArenaEntity ownership label.
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Topic|Component|Process|Config|Constraint)
ON EACH [n.name, n.description, n.fqn];
