"""
forecast_service.py
-------------------
Orquesta el flujo completo:
    serie histórica → ensemble → clasificación hidrológica → persistencia → respuesta

Es el único módulo que conoce todas las capas (ML, ORM, schemas).
"""

import uuid
import pandas as pd
from sqlalchemy.orm import Session

from app.ml import ensemble, classifier
from app.models.forecast import Forecast
from app.schemas.forecast import ForecastRequest, ForecastResponse, ModelMetrics


def create_forecast(
    body: ForecastRequest,
    db: Session,
    models: dict,
) -> ForecastResponse:
    """
    Parameters
    ----------
    body : ForecastRequest
        Parámetros de la petición (horizon_days, model_override).
    db : Session
        Sesión de SQLAlchemy inyectada por Depends(get_db).
    models : dict
        Artefactos ML cargados en app.state.models durante el lifespan.

    Returns
    -------
    ForecastResponse con el pronóstico, métricas y nivel de alerta.
    """
    # 1. Recuperar la serie histórica completa desde los artefactos
    serie: pd.Series = models["serie_historica"]

    # 2. Ejecutar el ensemble y seleccionar el mejor modelo
    result = ensemble.run_forecast(
        serie=serie,
        horizon=body.horizon_days,
        models=models,
        model_override=body.model_override,
    )

    best_model: str = result["best_model"]
    forecast_series: pd.Series = result["forecast_series"]
    metrics_by_model: dict = result["metrics_by_model"]

    forecast_values = [round(v, 4) for v in forecast_series.tolist()]
    forecast_dates = [d.strftime("%Y-%m-%d") for d in forecast_series.index]

    # 3. Clasificar nivel de alerta hidrológica
    alert_level = classifier.classify_alert(forecast_values)

    # 4. Extraer métricas del modelo seleccionado
    best_metrics = metrics_by_model[best_model]

    # 5. Persistir en base de datos
    forecast_id = uuid.uuid4()
    record = Forecast(
        id=forecast_id,
        horizon_days=body.horizon_days,
        best_model=best_model,
        rmse=best_metrics["rmse"],
        mae=best_metrics["mae"],
        mape=best_metrics["mape"],
        nse=best_metrics["nse"],
        r2=best_metrics["r2"],
        mda=best_metrics["mda"],
        forecast_dates=forecast_dates,
        forecast_values=forecast_values,
        forecast_mean=round(float(forecast_series.mean()), 4),
        forecast_min=round(float(forecast_series.min()), 4),
        forecast_max=round(float(forecast_series.max()), 4),
        alert_level=alert_level,
    )
    db.add(record)
    db.commit()

    # 6. Construir y retornar la respuesta Pydantic
    return ForecastResponse(
        forecast_id=str(forecast_id),
        best_model=best_model,
        horizon_days=body.horizon_days,
        forecast_dates=forecast_dates,
        forecast_values=forecast_values,
        forecast_mean=record.forecast_mean,
        forecast_min=record.forecast_min,
        forecast_max=record.forecast_max,
        alert_level=alert_level,
        metrics=ModelMetrics(**best_metrics),
        model_comparison={
            m: {k: round(v, 4) for k, v in met.items()}
            for m, met in metrics_by_model.items()
        },
    )
