from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db.enums import SystemUsageAttribution
from onyx.db.models import SystemUsage
from onyx.utils.datetime import datetime_to_utc

_CONFLICT_COLUMNS = [
    "attribution",
    "window_start",
    "model",
    "flow",
    "provider",
]


class SystemTokenUsageBucket(BaseModel):
    window_start: datetime
    tokens: int


def record_system_usage(
    db_session: Session,
    attribution: SystemUsageAttribution,
    model: str,
    flow: str,
    provider: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cost_cents: float,
    window_start: datetime,
    *,
    cache_creation_tokens: int = 0,
) -> None:
    """Accumulate one non-user generation into its daily rollup."""
    statement = pg_insert(SystemUsage).values(
        attribution=attribution,
        window_start=window_start,
        model=model,
        flow=flow,
        provider=provider or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_cents=cost_cents,
    )
    statement = statement.on_conflict_do_update(
        index_elements=_CONFLICT_COLUMNS,
        set_={
            "input_tokens": SystemUsage.input_tokens + statement.excluded.input_tokens,
            "output_tokens": (
                SystemUsage.output_tokens + statement.excluded.output_tokens
            ),
            "cache_read_tokens": (
                SystemUsage.cache_read_tokens + statement.excluded.cache_read_tokens
            ),
            "cache_creation_tokens": (
                SystemUsage.cache_creation_tokens
                + statement.excluded.cache_creation_tokens
            ),
            "cost_cents": SystemUsage.cost_cents + statement.excluded.cost_cents,
        },
    )
    db_session.execute(statement)
    db_session.flush()


def get_system_cost_cents_since(db_session: Session, cutoff: datetime) -> float:
    total = db_session.scalar(
        select(func.coalesce(func.sum(SystemUsage.cost_cents), 0.0)).where(
            SystemUsage.window_start >= cutoff
        )
    )
    return float(total or 0.0)


def get_system_cost_cents_buckets_since(
    db_session: Session, cutoff: datetime
) -> list[tuple[datetime, float]]:
    rows = db_session.execute(
        select(
            SystemUsage.window_start,
            func.coalesce(func.sum(SystemUsage.cost_cents), 0.0),
        )
        .where(SystemUsage.window_start >= cutoff)
        .group_by(SystemUsage.window_start)
    ).all()
    return [(datetime_to_utc(window_start), float(cost)) for window_start, cost in rows]


def get_system_token_buckets_since(
    db_session: Session, cutoff: datetime
) -> list[SystemTokenUsageBucket]:
    rows = db_session.execute(
        select(
            SystemUsage.window_start,
            func.sum(SystemUsage.input_tokens + SystemUsage.output_tokens),
        )
        .where(SystemUsage.window_start >= cutoff)
        .group_by(SystemUsage.window_start)
        .order_by(SystemUsage.window_start)
    ).all()
    return [
        SystemTokenUsageBucket(
            window_start=datetime_to_utc(window_start), tokens=int(tokens)
        )
        for window_start, tokens in rows
    ]
