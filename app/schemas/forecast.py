# app/schemas/forecast.py
import uuid
from datetime import date
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator

class HistoricRecord(BaseModel):
    fecha: date = Field(..., description="Fecha del registro (AAAA-MM-DD)")
    caudal: float = Field(..., description="Caudal medio diario en m³/s", ge=0.0)

class ForecastRequest(BaseModel):
    horizon_days: int = Field(30, description="Horizonte a pronosticar", ge=1, le=90)
    model_override: Optional[str] = Field(None, description="Forzar un modelo específico")
    data: List[HistoricRecord] = Field(..., description="Registros históricos continuos")

    # 👇 Desactiva la advertencia del prefijo model_ para este esquema
    model_config = {"protected_namespaces": ()}

    @field_validator("data")
    @classmethod
    def validate_minimum_records(cls, v: List[HistoricRecord]) -> List[HistoricRecord]:
        if len(v) < 35:
            raise ValueError("Se requieren mínimo 35 registros diarios continuos para calcular variables de rezago.")
        return v

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    nse: float
    r2: float
    mda: float

class ForecastResponse(BaseModel):
    forecast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    best_model: str
    horizon_days: int
    forecast_dates: List[str]
    forecast_values: List[float]
    forecast_mean: float
    forecast_min: float
    forecast_max: float
    alert_level: str
    metrics: ModelMetrics
    model_comparison: Dict[str, ModelMetrics]

    # 👇 Desactiva la advertencia del prefijo model_ para este esquema
    model_config = {"protected_namespaces": ()}