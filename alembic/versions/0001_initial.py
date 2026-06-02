"""initial — tabla forecasts

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecasts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # Parámetros de la solicitud
        sa.Column("horizon_days", sa.Integer, nullable=False),
        sa.Column("best_model", sa.String(30), nullable=False),
        # Métricas del modelo seleccionado en test
        sa.Column("rmse", sa.Float, nullable=False),
        sa.Column("mae", sa.Float, nullable=False),
        sa.Column("mape", sa.Float, nullable=False),
        sa.Column("nse", sa.Float, nullable=False),
        sa.Column("r2", sa.Float, nullable=False),
        sa.Column("mda", sa.Float, nullable=False),
        # Resultado del pronóstico
        sa.Column("forecast_dates", postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("forecast_values", postgresql.ARRAY(sa.Float), nullable=False),
        sa.Column("forecast_mean", sa.Float, nullable=False),
        sa.Column("forecast_min", sa.Float, nullable=False),
        sa.Column("forecast_max", sa.Float, nullable=False),
        # Nivel de alerta hidrológica
        sa.Column("alert_level", sa.String(10), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("forecasts")
