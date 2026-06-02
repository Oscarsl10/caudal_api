import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, ARRAY, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class Forecast(Base):
    """Registro de cada pronóstico de caudal generado por la API."""

    __tablename__ = "forecasts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Parámetros de entrada
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    best_model: Mapped[str] = mapped_column(String(30), nullable=False)

    # Métricas del modelo seleccionado en test
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    nse: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    mda: Mapped[float] = mapped_column(Float, nullable=False)

    # Resultado del pronóstico serializado como lista de valores
    forecast_dates: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    forecast_values: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)

    # Estadísticas del pronóstico
    forecast_mean: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_min: Mapped[float] = mapped_column(Float, nullable=False)
    forecast_max: Mapped[float] = mapped_column(Float, nullable=False)

    # Nivel de alerta hidrológica derivado del pronóstico
    alert_level: Mapped[str] = mapped_column(String(10), nullable=False)
