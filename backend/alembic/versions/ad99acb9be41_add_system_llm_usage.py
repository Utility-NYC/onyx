"""add system llm usage

Revision ID: ad99acb9be41
Revises: 947b94d2ebf1
Create Date: 2026-09-02 16:59:23.117473

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ad99acb9be41"
down_revision = "947b94d2ebf1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("attribution", sa.String(), nullable=False),
        sa.Column(
            "window_start", sa.DateTime(timezone=True), nullable=False, index=True
        ),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("flow", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False, server_default=""),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column(
            "cache_read_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "cache_creation_tokens",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("cost_cents", sa.Numeric(18, 6), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attribution IN ('SYSTEM', 'UNATTRIBUTED')",
            name="ck_system_usage_attribution",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_system_usage_dims",
        "system_usage",
        ["attribution", "window_start", "model", "flow", "provider"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_system_usage_dims", table_name="system_usage")
    op.drop_index("ix_system_usage_window_start", table_name="system_usage")
    op.drop_table("system_usage")
