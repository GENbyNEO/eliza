-- Enable pgvector extension

-- -- Drop existing tables and extensions
-- DROP EXTENSION IF EXISTS vector CASCADE;
-- DROP TABLE IF EXISTS relationships CASCADE;
-- DROP TABLE IF EXISTS participants CASCADE;
-- DROP TABLE IF EXISTS logs CASCADE;
-- DROP TABLE IF EXISTS goals CASCADE;
-- DROP TABLE IF EXISTS memories CASCADE;
-- DROP TABLE IF EXISTS rooms CASCADE;
-- DROP TABLE IF EXISTS accounts CASCADE;
-- DROP TABLE IF EXISTS knowledge CASCADE;


CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

-- Create a function to determine vector dimension
CREATE OR REPLACE FUNCTION get_embedding_dimension()
RETURNS INTEGER AS $$
BEGIN
    -- Check for OpenAI first
    IF current_setting('app.use_openai_embedding', TRUE) = 'true' THEN
        RETURN 1536;  -- OpenAI dimension
    -- Then check for Ollama
    ELSIF current_setting('app.use_ollama_embedding', TRUE) = 'true' THEN
        RETURN 1024;  -- Ollama mxbai-embed-large dimension
    -- Then check for GAIANET
    ELSIF current_setting('app.use_gaianet_embedding', TRUE) = 'true' THEN
        RETURN 768;  -- Gaianet nomic-embed dimension
    ELSE
        RETURN 384;   -- BGE/Other embedding dimension
    END IF;
END;
$$ LANGUAGE plpgsql;

BEGIN;

CREATE TABLE IF NOT EXISTS accounts (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT,
    "username" TEXT,
    "email" TEXT NOT NULL,
    "avatarUrl" TEXT,
    "details" JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS rooms (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

DO $$
DECLARE
    vector_dim INTEGER;
BEGIN
    vector_dim := get_embedding_dimension();

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS memories (
            "id" UUID PRIMARY KEY,
            "type" TEXT NOT NULL,
            "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            "content" JSONB NOT NULL,
            "embedding" vector(%s),
            "userId" UUID REFERENCES accounts("id"),
            "agentId" UUID REFERENCES accounts("id"),
            "roomId" UUID REFERENCES rooms("id"),
            "unique" BOOLEAN DEFAULT true NOT NULL,
            CONSTRAINT fk_room FOREIGN KEY ("roomId") REFERENCES rooms("id") ON DELETE CASCADE,
            CONSTRAINT fk_user FOREIGN KEY ("userId") REFERENCES accounts("id") ON DELETE CASCADE,
            CONSTRAINT fk_agent FOREIGN KEY ("agentId") REFERENCES accounts("id") ON DELETE CASCADE
        )', vector_dim);
END $$;

CREATE TABLE IF NOT EXISTS  goals (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID REFERENCES accounts("id"),
    "name" TEXT,
    "status" TEXT,
    "description" TEXT,
    "roomId" UUID REFERENCES rooms("id"),
    "objectives" JSONB DEFAULT '[]'::jsonb NOT NULL,
    CONSTRAINT fk_room FOREIGN KEY ("roomId") REFERENCES rooms("id") ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY ("userId") REFERENCES accounts("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS  logs (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID NOT NULL REFERENCES accounts("id"),
    "body" JSONB NOT NULL,
    "type" TEXT NOT NULL,
    "roomId" UUID NOT NULL REFERENCES rooms("id"),
    CONSTRAINT fk_room FOREIGN KEY ("roomId") REFERENCES rooms("id") ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY ("userId") REFERENCES accounts("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS  participants (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "userId" UUID REFERENCES accounts("id"),
    "roomId" UUID REFERENCES rooms("id"),
    "userState" TEXT,
    "last_message_read" TEXT,
    UNIQUE("userId", "roomId"),
    CONSTRAINT fk_room FOREIGN KEY ("roomId") REFERENCES rooms("id") ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY ("userId") REFERENCES accounts("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS  relationships (
    "id" UUID PRIMARY KEY,
    "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "userA" UUID NOT NULL REFERENCES accounts("id"),
    "userB" UUID NOT NULL REFERENCES accounts("id"),
    "status" TEXT,
    "userId" UUID NOT NULL REFERENCES accounts("id"),
    CONSTRAINT fk_user_a FOREIGN KEY ("userA") REFERENCES accounts("id") ON DELETE CASCADE,
    CONSTRAINT fk_user_b FOREIGN KEY ("userB") REFERENCES accounts("id") ON DELETE CASCADE,
    CONSTRAINT fk_user FOREIGN KEY ("userId") REFERENCES accounts("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS  cache (
    "key" TEXT NOT NULL,
    "agentId" TEXT NOT NULL,
    "value" JSONB DEFAULT '{}'::jsonb,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP,
    PRIMARY KEY ("key", "agentId")
);

DO $$
DECLARE
    vector_dim INTEGER;
BEGIN
    vector_dim := get_embedding_dimension();

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS knowledge (
            "id" UUID PRIMARY KEY,
            "agentId" UUID REFERENCES accounts("id"),
            "content" JSONB NOT NULL,
            "embedding" vector(%s),
            "createdAt" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            "isMain" BOOLEAN DEFAULT FALSE,
            "originalId" UUID REFERENCES knowledge("id"),
            "chunkIndex" INTEGER,
            "isShared" BOOLEAN DEFAULT FALSE,
            CHECK(("isShared" = true AND "agentId" IS NULL) OR ("isShared" = false AND "agentId" IS NOT NULL))
        )', vector_dim);
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw ("embedding" vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memories_type_room ON memories("type", "roomId");
CREATE INDEX IF NOT EXISTS idx_participants_user ON participants("userId");
CREATE INDEX IF NOT EXISTS idx_participants_room ON participants("roomId");
CREATE INDEX IF NOT EXISTS idx_relationships_users ON relationships("userA", "userB");
CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge("agentId");
CREATE INDEX IF NOT EXISTS idx_knowledge_agent_main ON knowledge("agentId", "isMain");
CREATE INDEX IF NOT EXISTS idx_knowledge_original ON knowledge("originalId");
CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge("agentId", "createdAt");
CREATE INDEX IF NOT EXISTS idx_knowledge_shared ON knowledge("isShared");
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding ON knowledge USING ivfflat (embedding vector_cosine_ops);

-- Add metadata to knowledge tables
CREATE INDEX IF NOT EXISTS idx_knowledge_metadata_likes ON knowledge ((content ->> 'metadata' ->> 'likes'));
CREATE INDEX IF NOT EXISTS idx_knowledge_metadata_created_at ON knowledge ((content ->> 'metadata' ->> 'createdAt'));

-- TF-IDF
CREATE OR REPLACE FUNCTION calculate_idf(term text, contents text[])
RETURNS float AS $$
DECLARE
    N float;  -- Total number of documents
    n float;  -- Number of documents containing the term
BEGIN
    N := array_length(contents, 1);
    n := (SELECT count(*) FROM unnest(contents) c WHERE c ILIKE '%' || term || '%');
    RETURN ln((N - n + 0.5)/(n + 0.5) + 1);
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION bm25_score(
    query text,
    document text,
    all_documents text[],
    k1 float DEFAULT 1.2,
    b float DEFAULT 0.75
) RETURNS float AS $$
DECLARE
    score float := 0;
    query_terms text[];
    term text;
    tf float;
    idf float;
    doc_length float;
    avg_doc_length float;
BEGIN
    query_terms := regexp_split_to_array(lower(query), '\s+');

    doc_length := array_length(regexp_split_to_array(document, '\s+'), 1);

    avg_doc_length := (
        SELECT avg(array_length(regexp_split_to_array(d, '\s+'), 1))
        FROM unnest(all_documents) d
    );

    FOREACH term IN ARRAY query_terms LOOP
        tf := (
            SELECT count(*)::float
            FROM regexp_matches(lower(document), lower(term), 'g')
        );

        idf := calculate_idf(term, all_documents);

        score := score + (
            idf * (
                (tf * (k1 + 1)) /
                (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            )
        );
    END LOOP;

    SELECT
        MAX(bm25_raw_score),
        MIN(bm25_raw_score)
    INTO max_score, min_score
    FROM (
        SELECT bm25_score(
            query,
            d,
            all_documents,
            k1,
            b
        ) as bm25_raw_score
        FROM unnest(all_documents) d
    ) scores;

    IF max_score = min_score THEN
        RETURN CASE
            WHEN score > 0 THEN 1.0
            ELSE 0.0
        END;
    ELSE
        RETURN (score - min_score) / (max_score - min_score);
    END IF;

    RETURN score;
END;
$$ LANGUAGE plpgsql;

COMMIT;
