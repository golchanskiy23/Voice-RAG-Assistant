from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE documents (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            filename text NOT NULL,
            sha256 char(64) NOT NULL UNIQUE,
            mime text NOT NULL,
            pages integer CHECK (pages >= 0),
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE chunks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            ord integer NOT NULL CHECK (ord >= 0),
            text text NOT NULL,
            page integer,
            section text,
            tokens integer NOT NULL CHECK (tokens > 0),
            embedding vector({EMBEDDING_DIM}),
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED,
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            UNIQUE (document_id, ord)
        )
        """
    )
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
